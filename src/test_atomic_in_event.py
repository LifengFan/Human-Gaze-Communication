import os
import argparse
import numpy as np
import torch
import torch.autograd
from os.path import isdir
import get_data
import models
import dataset.metadata
import dataset.utils
import utils
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from utils import get_metric_from_confmat
import matplotlib.pyplot as plt
# from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback

event_label_dict= {'SingleGaze': 0, 'GazeFollow': 1, 'AvertGaze': 2, 'MutualGaze': 3, 'JointAtt': 4}

f = open(os.path.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/', 'event', "clean_event_seq.pkl"), "rb")
gt_rec = pickle.load(f)
pred_rec=gt_rec

f2=open(os.path.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/', 'test_atomic_in_event.pkl'), "rb")
pred_list=pickle.load(f2)

correct_cnt=0.
total_cnt=0.

for i in range(len(pred_list)):

    tmp=pred_list[i]
    pred, ev, rec_ind, sq_ind=tmp

    if ev == 0:
        event = 'SingleGaze'
    elif ev == 1:
        event = 'GazeFollow'
    elif ev == 2:
        event = 'AvertGaze'
    elif ev == 3:
        event = 'MutualGaze'
    elif ev == 4:
        event = 'JointAtt'

    if pred==0:

        if gt_rec[event][rec_ind][1][sq_ind]=='single':
            correct_cnt+=1

        pred_rec[event][rec_ind][1][sq_ind]='single'
    elif pred==1:
        if gt_rec[event][rec_ind][1][sq_ind]=='mutual':
            correct_cnt+=1
        pred_rec[event][rec_ind][1][sq_ind] = 'mutual'
    elif pred==2:
        if gt_rec[event][rec_ind][1][sq_ind]=='avert':
            correct_cnt+=1
        pred_rec[event][rec_ind][1][sq_ind] = 'avert'
    elif pred==3:
        if gt_rec[event][rec_ind][1][sq_ind]=='refer':
            correct_cnt+=1
        pred_rec[event][rec_ind][1][sq_ind] = 'refer'
    elif pred==4:
        if gt_rec[event][rec_ind][1][sq_ind]=='follow':
            correct_cnt+=1
        pred_rec[event][rec_ind][1][sq_ind] = 'follow'
    elif pred==5:
        if gt_rec[event][rec_ind][1][sq_ind]=='share':
            correct_cnt+=1
        pred_rec[event][rec_ind][1][sq_ind] = 'share'

    total_cnt+=1

print(correct_cnt/total_cnt)
    #{'single': 0, 'mutual': 1, 'avert': 2, 'refer': 3, 'follow': 4, 'share': 5}

with open(os.path.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/', 'pred_res_for_atomic_in_event.pkl'), 'w') as f:
        pickle.dump(pred_rec, f)


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


def main(args):

    args.cuda = args.use_cuda and torch.cuda.is_available()

    test_set, test_loader = get_data.get_data_test_atomic_in_event(args)

    #model = models.Atomic(args)
    model=models.Atomic_node_only(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    #{'single': 0, 'mutual': 1, 'avert': 2, 'refer': 3, 'follow': 4, 'share': 5}
    criterion = [torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 0.05, 0.25, 0.25, 0.25, 0.15])), torch.nn.MSELoss()]

    # {'NA': 0, 'single': 1, 'mutual': 2, 'avert': 3, 'refer': 4, 'follow': 5, 'share': 6}

#--------------------------------------------------
    # ------------------------
    # use multi-gpu

    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")

        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
        #model=model.cuda()
        criterion[0] = criterion[0].cuda()
        criterion[1] = criterion[1].cuda()

    elif args.cuda:
        model = model.cuda()
        criterion[0] = criterion[0].cuda()
        criterion[1] = criterion[1].cuda()

    # ----------------------------------------------------------------------------------------------------------
    # test

    checkpoint_dir = args.resume
    best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')

    if os.path.isfile(best_model_file):
        print("====> loading best model {}".format(best_model_file))

        checkpoint = torch.load(best_model_file)
        args.start_epoch = checkpoint['epoch']
        best_epoch_error = checkpoint['best_epoch_acc']

        try:
            avg_epoch_error = checkpoint['avg_epoch_acc']
        except KeyError:
            avg_epoch_error = np.inf

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        #model.cuda()

        print("===> loaded best model {} (epoch {})".format(best_model_file, checkpoint['epoch']))


    #test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
    #test_loss, test_acc, confmat, top2_acc, pred_list = test(test_loader, model, criterion, args)
    pred_list = test(test_loader, model, criterion, args)


    # print("Test Acc {}".format(test_acc))
    # print("Top 2 Test Acc {}".format(top2_acc))
    #
    # # todo: need to change the mode here!
    # get_metric_from_confmat(confmat, 'atomic')

    with open(os.path.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/', 'test_atomic_in_event.pkl'), 'w') as f:
        pickle.dump(pred_list, f)


def test(test_loader, model, criterion, args):
    model.eval()

    # total_acc = AverageMeter()
    # total_loss = AverageMeter()
    # confmat=np.zeros((6,6))

    pred_list=[]

    #confmat_transient=np.zeros((4,4))

    #total_acc_top2=AverageMeter()


    for i, (head_batch, pos_batch, attmat_batch, atomic_label_batch, ID_rec) in enumerate(test_loader):

        print(i)

        batch_size = head_batch.shape[0]

        if args.cuda:
            heads = (torch.autograd.Variable(head_batch)).cuda()
            poses = (torch.autograd.Variable(pos_batch)).cuda()
            attmat_gt=(torch.autograd.Variable(attmat_batch)).cuda()
            atomic_gt = (torch.autograd.Variable(atomic_label_batch)).cuda()
            ID = (torch.autograd.Variable(ID_rec)).cuda()

            # event_label = (torch.autograd.Variable(event_label_batch)).cuda()
            # pos_in_sq = (torch.autograd.Variable(pos_in_sq_batch)).cuda()

        with torch.set_grad_enabled(False):

            pred_atomic = model(heads, poses, attmat_gt) #[N, 6, 1,1,1]

            #test_loss = 0

            for bid in range(batch_size):
                # todo:check pre_atomic dim [N,6,1,1,1]??
                #tmp_loss = criterion[0](pred_atomic[bid, :].unsqueeze(0), atomic_gt[bid].unsqueeze(0))

                # print('label loss', criterion[0](sl_pred[nid][bid, :].unsqueeze(0), sl_gt[bid, nid].unsqueeze(0)))
                # print('attmat loss', criterion[1](attmat_pred, attmat_gt))
                #
                # total_loss.update(tmp_loss.item(), 1)
                # test_loss = test_loss + tmp_loss

                pred = torch.argmax(pred_atomic[bid, :], dim=0)
                # bv = (pred == atomic_gt[bid].data)
                # bv = bv.type(torch.cuda.FloatTensor)
                # total_acc.update(bv.item(), 1)

                pred_list.append([pred.cpu().numpy(), ID[bid][5].cpu().numpy(), ID[bid][6].cpu().numpy(), ID[bid][7].cpu().numpy()])
                # # todo: use top2 acc here!
                # _, sort_ind = torch.sort(pred_atomic[bid, :], descending=True)
                # pred_2 = sort_ind[1]
                #
                # bv2 = (pred == atomic_gt[bid].data or pred_2 == atomic_gt[bid].data)
                # bv2 = bv2.type(torch.cuda.FloatTensor)
                # total_acc_top2.update(bv2.item(), 1)
                #
                # confmat[atomic_gt[bid].data, pred]+=1

            #print('Iter: {} Testing Loss: {:.4f} Total Avg Acc: {:.4f}, Top 2 Avg Acc: {:.4f}'.format( i,test_loss.item(),total_acc.avg, total_acc_top2.avg))


    #return total_loss.avg, total_acc.avg, confmat, total_acc_top2.avg, pred_list
    return pred_list


def parse_arguments():

    path = dataset.utils.Paths()

    project_name = 'train_atomic_node_only_iter2'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('--project-name', default=project_name, help='project name')


    # path settings
    parser.add_argument('--project-root', default=path.project_root, help='project root path')
    parser.add_argument('--tmp-root', default=path.tmp_root, help='checkpoint path')
    parser.add_argument('--data-root', default='/media/ramdisk/', help='data path')
    parser.add_argument('--log-root', default=path.log_root, help='log files path')
    parser.add_argument('--resume', default=os.path.join(path.tmp_root, 'checkpoints', project_name),help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default=os.path.join(path.tmp_root, 'test_results', project_name),help='path to save test metrics')

    # optimization options
    parser.add_argument('--load-last-checkpoint', default=False, help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint', default=False,help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--batch-size', type=int, default=24, help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda', default=True, help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom', default=False, help='use visdom to visualize loss curve')

    parser.add_argument('--device-ids', default=[0,1], help='gpu ids')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()


    #    if args.visdom:
    #        vis = visdom.Visdom()
    #        assert vis.check_connection()

    #main(args)



