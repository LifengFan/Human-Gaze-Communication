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
import random
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

    print('start validate')
    val_epoch_acc, confmat = validate(validate_loader, args)
    print('validate confmat')
    get_metric_from_confmat(confmat, 'atomic')

    test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
    print('start test')
    test_epoch_acc, confmat_test = validate(test_loader, args)
    print('test confmat')
    get_metric_from_confmat(confmat_test, 'atomic')


def validate(validate_loader, args):        
    total_acc = AverageMeter()
    confmat = np.zeros((6, 6))
    for i, (head_batch, pos_batch, attmat_batch, atomic_label_batch,_,_) in enumerate(validate_loader):

        batch_size=head_batch.shape[0]
        
        for bid in range(batch_size):
            value = random.uniform(0,1)
            if value > 0 and value <0.1667:
                pred = 0
            elif value >0.1667 and value <0.3333:
                pred = 1
            elif value > 0.3333 and value <0.5:
                pred = 2
            elif value > 0.5 and value <0.6667:
                pred = 3
            elif value > 0.6667 and value <0.8333:
                pred = 4
            elif value > 0.8333 and value < 1:
                pred = 5
            if bid == 255:
                a = 1
            bv = int(pred == atomic_label_batch[bid])
            total_acc.update(bv,1) 
            confmat[int(atomic_label_batch[bid]), pred] +=1

        if i % 50 ==0:
            print('Iter: {} Total Avg Acc: {:.4f}'.format( i, total_acc.avg))

    return total_acc.avg, confmat



def parse_arguments():

    path = dataset.utils.Paths()

    project_name = 'train_atomic_3'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('--project-name', default=project_name, help='project name')

    # path settings
    parser.add_argument('--project-root', default=path.project_root, help='project root path')
    parser.add_argument('--tmp-root', default=path.tmp_root, help='checkpoint path')
    parser.add_argument('--data-root', default=path.data_root, help='data path')
    parser.add_argument('--log-root', default=path.log_root, help='log files path')
    parser.add_argument('--resume', default=os.path.join(path.tmp_root, 'checkpoints', project_name),help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default=os.path.join(path.tmp_root, 'test_results', project_name),help='path to save test metrics')

    # optimization options
    parser.add_argument('--load-last-checkpoint', default=False, help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint', default=False,help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--batch-size', type=int, default=256, help='Input batch size for training (default: 10)')
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



