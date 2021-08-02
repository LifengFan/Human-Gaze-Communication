import sys
sys.path.append("/home/lfan/Dropbox/Projects/ICCV19/RunComm/src/")

import torch
import torch.nn
import torch.autograd
import units
import os.path as op

class Nodes2SL(torch.nn.Module):
    def __init__(self, args):
        super(Nodes2SL, self).__init__()

        self.args = args
        self.resnet = units.Resnet.ResNet50(3)

        # load headpose-finetuned resnet50 weights
        self._load_pretrained_weight(op.join(args.tmp_root,'checkpoints','finetune_headpose','model_best.pth'))

        # replace the last fc layer of resnet with a new one
        num_ftrs = self.resnet.resnet.fc.in_features
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 256)

        self.resnet.resnet.fc.weight.data.normal_(0,0.01)
        self.resnet.resnet.fc.bias.data.zero_()


        self.fc1_1=torch.nn.Linear(6*262,1024)
        self.bn1_1=torch.nn.BatchNorm1d(1024)
        self.fc2_1=torch.nn.Linear(1024,512)
        self.bn2_1 = torch.nn.BatchNorm1d(512)
        self.fc3_1=torch.nn.Linear(512,7)

        self.fc1_2=torch.nn.Linear(6*262,1024)
        self.bn1_2 = torch.nn.BatchNorm1d(1024)
        self.fc2_2=torch.nn.Linear(1024,512)
        self.bn2_2 = torch.nn.BatchNorm1d(512)
        self.fc3_2=torch.nn.Linear(512,7)

        self.fc1_3=torch.nn.Linear(6*262,1024)
        self.bn1_3 = torch.nn.BatchNorm1d(1024)
        self.fc2_3=torch.nn.Linear(1024,512)
        self.bn2_3 = torch.nn.BatchNorm1d(512)
        self.fc3_3=torch.nn.Linear(512,7)

        self.fc1_4=torch.nn.Linear(6*262,1024)
        self.bn1_4 = torch.nn.BatchNorm1d(1024)
        self.fc2_4=torch.nn.Linear(1024,512)
        self.bn2_4 = torch.nn.BatchNorm1d(512)
        self.fc3_4=torch.nn.Linear(512,7)

        self.fc1_5=torch.nn.Linear(6*262,1024)
        self.bn1_5 = torch.nn.BatchNorm1d(1024)
        self.fc2_5=torch.nn.Linear(1024,512)
        self.bn2_5 = torch.nn.BatchNorm1d(512)
        self.fc3_5=torch.nn.Linear(512,7)

        self.fc1_6=torch.nn.Linear(6*262,1024)
        self.bn1_6 = torch.nn.BatchNorm1d(1024)
        self.fc2_6=torch.nn.Linear(1024,512)
        self.bn2_6 = torch.nn.BatchNorm1d(512)
        self.fc3_6=torch.nn.Linear(512,7)

        self.relu=torch.nn.ReLU()

        self.dropout=torch.nn.Dropout(p=0.5)


    def _load_pretrained_weight(self, model_path):

        model_dict=self.resnet.state_dict()
        pretrained_model=torch.load(model_path)['state_dict']

        pretrained_dict={}

        for k,v in pretrained_model.items():
            if k[len('module.resnet.'):] in model_dict:
                pretrained_dict[k[len('module.resnet.'):]]=v

        print(len(model_dict))

        print(len(pretrained_dict))

        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)


    def forward(self, nodes, pos):
        # nodes [N, max_node_num, 3, 224, 224]
        # pos [N, max_node_num, 6]
        # suppose max_node_num=6

        N=nodes.shape[0]
        max_node_num=nodes.shape[1]
        nodes=nodes.view(N*max_node_num, 3, 224, 224)
        nodes_feature=self.resnet(nodes).view(N,max_node_num,256)

        # out [N, max_node_num,262]
        out=torch.cat((nodes_feature,pos),2)

        feat=out.view(N, max_node_num*(256+6))

        #brach for node 1
        n1=self.fc1_1(feat)
        n1=self.relu(n1)
        n1=self.dropout(n1)
        n1=self.fc2_1(n1)
        n1=self.relu(n1)
        n1=self.dropout(n1)
        n1=self.fc3_1(n1)

        #brach for node 2
        n2=self.fc1_2(feat)
        n2=self.relu(n2)
        n2=self.dropout(n2)
        n2=self.fc2_2(n2)
        n2=self.relu(n2)
        n2=self.dropout(n2)
        n2=self.fc3_2(n2)

        #brach for node 1
        n3=self.fc1_3(feat)
        n3=self.relu(n3)
        n3=self.dropout(n3)
        n3=self.fc2_3(n3)
        n3=self.relu(n3)
        n3=self.dropout(n3)
        n3=self.fc3_3(n3)

        #brach for node 1
        n4=self.fc1_4(feat)
        n4=self.relu(n4)
        n4=self.dropout(n4)
        n4=self.fc2_4(n4)
        n4=self.relu(n4)
        n4=self.dropout(n4)
        n4=self.fc3_4(n4)

        #brach for node 1
        n5=self.fc1_5(feat)
        n5=self.relu(n5)
        n5=self.dropout(n5)
        n5=self.fc2_5(n5)
        n5=self.relu(n5)
        n5=self.dropout(n5)
        n5=self.fc3_5(n5)

        #brach for node 1
        n6=self.fc1_6(feat)
        n6=self.relu(n6)
        n6=self.dropout(n6)
        n6=self.fc2_6(n6)
        n6=self.relu(n6)
        n6=self.dropout(n6)
        n6=self.fc3_6(n6)

        return [n1,n2,n3,n4,n5,n6]

def main():
    pass

if __name__ == '__main__':

    main()



