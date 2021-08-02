import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CLASS = ['SingleGaze', 'MutualGaze', 'AvertGaze', 'GazeFollow', 'JointAtt']


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)
        f.close()
        self.pad_dim = 50

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        if len(sequence['data']) <= self.pad_dim:
            padded = sequence['data'] + [0 for _ in range(self.pad_dim - len(sequence['data']))]
        else:
            padded = sequence['data'][:self.pad_dim]

        if len(sequence['len']) <= self.pad_dim:
            padded_len = sequence['len'] + [0 for _ in range(self.pad_dim - len(sequence['len']))]
        else:
            padded_len = sequence['len'][:self.pad_dim]
        return {'label': torch.tensor(CLASS.index(sequence['label'])), 'data': torch.tensor(padded).float(), 'len': torch.tensor(padded_len).float()}


class EDNet(nn.Module):
    def __init__(self):
        super(EDNet, self).__init__()
        self.encoder_1 = nn.Linear(50, 50)
        self.encoder_2 = nn.Linear(50, 50)
        self.decoder = nn.Linear(100, 5)

    def forward(self, x_1, x_2):
        latent_1 = F.dropout(F.relu(self.encoder_1(x_1)), 0.8)
        latent_2 = F.dropout(F.relu(self.encoder_2(x_2)), 0.8)
        x = self.decoder(torch.cat((latent_1, latent_2), 1))
        return x


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc_1 = nn.Linear(100, 5)

    def forward(self, x_1, x_2):
        return self.fc_1(torch.cat((x_1, x_2), 1))
        # return self.fc_2(F.dropout(F.relu(self.fc_1(torch.cat((x_1, x_2), 1))), 0.8))


def train_loader():
    return DataLoader(dataset=EventDataset('train_seq.pickle'), num_workers=4, batch_size=64, shuffle=True)


def test_loader():
    return DataLoader(dataset=EventDataset('test_seq_without_gt.pickle'), num_workers=4, batch_size=1, shuffle=False)


def get_metric_from_confmat(confmat):

    N=5

    recall=np.zeros(N)
    precision=np.zeros(N)
    F_score=np.zeros(N)

    correct_cnt=0.
    total_cnt=0.

    for i in range(N):

        recall[i]=confmat[i,i]/(np.sum(confmat[i,:])+1e-7)

        precision[i]=confmat[i,i]/(np.sum(confmat[:,i])+1e-7)

        F_score[i]=2*precision[i]*recall[i]/(precision[i]+recall[i]+1e-7)

        correct_cnt+=confmat[i,i]

        total_cnt+=np.sum(confmat[i,:])

    acc=correct_cnt/total_cnt

    print('===> Confusion Matrix for Event Label: \n {}'.format(confmat.astype(int)))

    print('===> Precision: \n  [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(precision[0]*100, precision[1]*100, precision[2]*100, precision[3]*100, precision[4]*100))

    print('===> Recall: \n [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(recall[0]*100, recall[1]*100, recall[2]*100, recall[3]*100, recall[4]*100))

    print('===> F score: \n [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(F_score[0]*100, F_score[1]*100, F_score[2]*100, F_score[3]*100, F_score[4]*100))

    print('===> Accuracy: {} %'.format(acc*100))

trainloader = train_loader()
testloader = test_loader()
criterion = nn.CrossEntropyLoss()
net = EDNet()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


total_acc_top1= AverageMeter()
total_acc_top2=AverageMeter()


for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        input_1, input_2, labels = data['data'], data['len'], data['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input_1, input_2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0


print('Finished Training')
correct = 0.0
correct_2 = 0.0
total = 0.0
confmat = np.zeros((5, 5))
with torch.no_grad():
    for data in testloader:
        input_1, input_2, labels = data['data'], data['len'], data['label']
        outputs = net(input_1, input_2)
        outputs.data = torch.rand((1, 5))
        _, predicted = torch.max(outputs.data, 1)
        valuse, ind = torch.topk(outputs.data, 2)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(labels.size(0)):
            confmat[predicted[i], labels[i]] += 1
            if labels[i] in ind.squeeze().numpy().tolist():
                correct_2 += 1
get_metric_from_confmat(confmat)


print('Top-1 Accuracy of the network on the test images: %f %%' % (
        100 * correct / total))

print('Top-2 Accuracy of the network on the test images: %f %%' % (
        100 * correct_2 / total))






