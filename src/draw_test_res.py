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

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

    plt.show()


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

    train_set, validate_set, test_set, train_loader, validate_loader, test_loader = get_data.get_data_atomic(args)

    #model = models.Atomic(args)
    model=models.Atomic_2branch(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #{'single': 0, 'mutual': 1, 'avert': 2, 'refer': 3, 'follow': 4, 'share': 5}
    criterion = [torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 0.05, 0.25, 0.25, 0.25, 0.15])), torch.nn.MSELoss()]

    # {'NA': 0, 'single': 1, 'mutual': 2, 'avert': 3, 'refer': 4, 'follow': 5, 'share': 6}

    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=1, verbose=True, mode='max')
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

    loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)

    if loaded_checkpoint:
        args, best_epoch_acc, avg_epoch_acc, model, optimizer = loaded_checkpoint

    test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
    test_loss, test_acc, confmat, top2_acc, correct_rec, error_rec= test(test_loader, model, criterion, args)

    #fw = open(os.path.join(args.tmp_root, 'correct_list.txt'), 'w')
    fc_single=open(os.path.join(args.tmp_root, 'correct_single.txt'), 'w')
    fc_mutual = open(os.path.join(args.tmp_root, 'correct_mutual.txt'), 'w')
    fc_avert = open(os.path.join(args.tmp_root, 'correct_avert.txt'), 'w')
    fc_refer = open(os.path.join(args.tmp_root, 'correct_refer.txt'), 'w')
    fc_follow = open(os.path.join(args.tmp_root, 'correct_follow.txt'), 'w')
    fc_share = open(os.path.join(args.tmp_root, 'correct_share.txt'), 'w')

    fe_single=open(os.path.join(args.tmp_root, 'error_single.txt'), 'w')
    fe_mutual = open(os.path.join(args.tmp_root, 'error_mutual.txt'), 'w')
    fe_avert = open(os.path.join(args.tmp_root, 'error_avert.txt'), 'w')
    fe_refer = open(os.path.join(args.tmp_root, 'error_refer.txt'), 'w')
    fe_follow = open(os.path.join(args.tmp_root, 'error_follow.txt'), 'w')
    fe_share = open(os.path.join(args.tmp_root, 'error_share.txt'), 'w')


    for item in correct_rec:

        if item[1]=='0':
            for i in range(5):
                fc_single.write(str(item[0][i].cpu().numpy()))
                fc_single.write(' ')
            fc_single.write(str(item[1].cpu().numpy()))
            fc_single.write(' ')
            fc_single.write(str(item[2].cpu().numpy()))
            fc_single.write('\n')

        elif item[1]=='1':
            for i in range(5):
                fc_mutual.write(str(item[0][i].cpu().numpy()))
                fc_mutual.write(' ')
            fc_mutual.write(str(item[1].cpu().numpy()))
            fc_mutual.write(' ')
            fc_mutual.write(str(item[2].cpu().numpy()))
            fc_mutual.write('\n')

        elif item[1]=='2':
            for i in range(5):
                fc_avert.write(str(item[0][i].cpu().numpy()))
                fc_avert.write(' ')
            fc_avert.write(str(item[1].cpu().numpy()))
            fc_avert.write(' ')
            fc_avert.write(str(item[2].cpu().numpy()))
            fc_avert.write('\n')

        elif item[1]=='3':

            for i in range(5):
                fc_refer.write(str(item[0][i].cpu().numpy()))
                fc_refer.write(' ')
            fc_refer.write(str(item[1].cpu().numpy()))
            fc_refer.write(' ')
            fc_refer.write(str(item[2].cpu().numpy()))
            fc_refer.write('\n')

        elif item[1] == '4':

            for i in range(5):
                fc_follow.write(str(item[0][i].cpu().numpy()))
                fc_follow.write(' ')
            fc_follow.write(str(item[1].cpu().numpy()))
            fc_follow.write(' ')
            fc_follow.write(str(item[2].cpu().numpy()))
            fc_follow.write('\n')

        elif item[1] == '5':

            for i in range(5):
                fc_share.write(str(item[0][i].cpu().numpy()))
                fc_share.write(' ')
            fc_share.write(str(item[1].cpu().numpy()))
            fc_share.write(' ')
            fc_share.write(str(item[2].cpu().numpy()))
            fc_share.write('\n')



    #fw2 = open(os.path.join(args.tmp_root, 'error_list.txt'), 'w')

    # for item in error_rec:
    #     for i in range(5):
    #         fw2.write(str(item[0][i].cpu().numpy()))
    #         fw2.write(' ')
    #     fw2.write(str(item[1].cpu().numpy()))
    #     fw2.write(' ')
    #     fw2.write(str(item[2].cpu().numpy()))
    #     fw2.write('\n')

    for item in error_rec:

        if item[1] == '0':
            for i in range(5):
                fe_single.write(str(item[0][i].cpu().numpy()))
                fe_single.write(' ')
            fe_single.write(str(item[1].cpu().numpy()))
            fe_single.write(' ')
            fe_single.write(str(item[2].cpu().numpy()))
            fe_single.write('\n')

        elif item[1] == '1':
            for i in range(5):
                fe_mutual.write(str(item[0][i].cpu().numpy()))
                fe_mutual.write(' ')
            fe_mutual.write(str(item[1].cpu().numpy()))
            fe_mutual.write(' ')
            fe_mutual.write(str(item[2].cpu().numpy()))
            fe_mutual.write('\n')

        elif item[1] == '2':
            for i in range(5):
                fe_avert.write(str(item[0][i].cpu().numpy()))
                fe_avert.write(' ')
            fe_avert.write(str(item[1].cpu().numpy()))
            fe_avert.write(' ')
            fe_avert.write(str(item[2].cpu().numpy()))
            fe_avert.write('\n')

        elif item[1] == '3':

            for i in range(5):
                fe_refer.write(str(item[0][i].cpu().numpy()))
                fe_refer.write(' ')
            fe_refer.write(str(item[1].cpu().numpy()))
            fe_refer.write(' ')
            fe_refer.write(str(item[2].cpu().numpy()))
            fe_refer.write('\n')

        elif item[1] == '4':

            for i in range(5):
                fe_follow.write(str(item[0][i].cpu().numpy()))
                fe_follow.write(' ')
            fe_follow.write(str(item[1].cpu().numpy()))
            fe_follow.write(' ')
            fe_follow.write(str(item[2].cpu().numpy()))
            fe_follow.write('\n')

        elif item[1] == '5':

            for i in range(5):
                fe_share.write(str(item[0][i].cpu().numpy()))
                fe_share.write(' ')
            fe_share.write(str(item[1].cpu().numpy()))
            fe_share.write(' ')
            fe_share.write(str(item[2].cpu().numpy()))
            fe_share.write('\n')


def test(test_loader, model, criterion, args):

    model.eval()

    total_acc = AverageMeter()
    total_loss = AverageMeter()
    confmat=np.zeros((6,6))
    correct_rec=list()
    error_rec=list()

    total_acc_top2=AverageMeter()

    for i, (head_batch, pos_batch, attmat_batch, atomic_label_batch, ID_rec) in enumerate(test_loader):


        batch_size = head_batch.shape[0]

        if args.cuda:
            heads = (torch.autograd.Variable(head_batch)).cuda()
            poses = (torch.autograd.Variable(pos_batch)).cuda()
            attmat_gt=(torch.autograd.Variable(attmat_batch)).cuda()
            atomic_gt = (torch.autograd.Variable(atomic_label_batch)).cuda()
            ID_rec=(torch.autograd.Variable(ID_rec)).cuda()


        with torch.set_grad_enabled(False):

            pred_atomic = model(heads, poses, attmat_gt) #[N, 6, 1,1,1]

            test_loss = 0

            for bid in range(batch_size):
                # todo:check pre_atomic dim [N,6,1,1,1]??
                tmp_loss = criterion[0](pred_atomic[bid, :].unsqueeze(0), atomic_gt[bid].unsqueeze(0))

                # print('label loss', criterion[0](sl_pred[nid][bid, :].unsqueeze(0), sl_gt[bid, nid].unsqueeze(0)))
                # print('attmat loss', criterion[1](attmat_pred, attmat_gt))

                total_loss.update(tmp_loss.item(), 1)
                test_loss = test_loss + tmp_loss

                pred = torch.argmax(pred_atomic[bid, :], dim=0)
                bv = (pred == atomic_gt[bid].data)
                bv = bv.type(torch.cuda.FloatTensor)
                if bv==1:

                    correct_rec.append((ID_rec[bid, ...],atomic_gt[bid].data,pred))

                else:
                    error_rec.append((ID_rec[bid, ...],atomic_gt[bid].data,pred))


                total_acc.update(bv.item(), 1)

                # todo: use top2 acc here!
                _, sort_ind = torch.sort(pred_atomic[bid, :], descending=True)
                pred_2 = sort_ind[1]

                bv2 = (pred == atomic_gt[bid].data or pred_2 == atomic_gt[bid].data)
                bv2 = bv2.type(torch.cuda.FloatTensor)
                total_acc_top2.update(bv2.item(), 1)

                confmat[atomic_gt[bid].data, pred]+=1

            print('Iter: {} Testing Loss: {:.4f} Total Avg Acc: {:.4f}, Top 2 Avg Acc: {:.4f}'.format( i,test_loss.item(),total_acc.avg, total_acc_top2.avg))


    return total_loss.avg, total_acc.avg, confmat, total_acc_top2.avg, correct_rec, error_rec


def parse_arguments():

    path = dataset.utils.Paths()

    project_name = 'train_atomic_2branch'
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
    parser.add_argument('--batch-size', type=int, default=48, help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda', default=True, help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.5, help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom', default=False, help='use visdom to visualize loss curve')

    parser.add_argument('--device-ids', default=[0,1], help='gpu ids')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()


    #    if args.visdom:
    #        vis = visdom.Visdom()
    #        assert vis.check_connection()

    main(args)



