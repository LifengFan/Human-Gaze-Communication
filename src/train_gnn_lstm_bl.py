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

    train_set, validate_set, test_set, train_loader, validate_loader, test_loader = get_data.get_data_pos_bl_lstm(args)

    model = models.GNN_lstm(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = [torch.nn.CrossEntropyLoss(weight=torch.Tensor([0, 0.05, 0.15, 0.15,0.25, 0.25, 0.15]), ignore_index=0), torch.nn.MSELoss()]

    # {'NA': 0, 'single': 1, 'mutual': 2, 'avert': 3, 'refer': 4, 'follow': 5, 'share': 6}

    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=1, verbose=True, mode='max')
#--------------------------------------------------
# test  fastai

    # ------------------------
    # use multi-gpu

    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")

        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
        criterion[0] = criterion[0].cuda()
        criterion[1] = criterion[1].cuda()

    elif args.cuda:
        model = model.cuda()
        criterion[0] = criterion[0].cuda()
        criterion[1] = criterion[1].cuda()

    if args.load_best_checkpoint:
        loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)

        if loaded_checkpoint:
            args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    if args.load_last_checkpoint:
        loaded_checkpoint = utils.load_last_checkpoint(args, model, optimizer, path=args.resume,
                                                       version=args.model_load_version)

        if loaded_checkpoint:
            args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

            # ------------------------------------------------------------------------------
            # Start Training!

    since = time.time()

    train_epoch_acc_all = []
    val_epoch_acc_all = []

    best_acc = 0
    avg_epoch_acc = 0

    for epoch in range(args.start_epoch, args.epochs):

        train_epoch_loss, train_epoch_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        train_epoch_acc_all.append(train_epoch_acc)

        val_epoch_loss, val_epoch_acc = validate(validate_loader, model, criterion, epoch, args)
        val_epoch_acc_all.append(val_epoch_acc)

        print('Epoch {}/{} Training Acc: {:.4f} Validation Acc: {:.4f}'.format(epoch, args.epochs - 1, train_epoch_acc,
                                                                               val_epoch_acc))
        print('*' * 15)

        scheduler.step(val_epoch_acc)

        is_best = val_epoch_acc > best_acc

        if is_best:
            best_acc = val_epoch_acc

        avg_epoch_acc = np.mean(val_epoch_acc_all)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_epoch_acc': best_acc,
            'avg_epoch_acc': avg_epoch_acc,
            'optimizer': optimizer.state_dict(), 'args': args}, is_best=is_best, directory=args.resume,
            version='epoch_{}'.format(str(epoch)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {},  Final Avg Val Acc: {}'.format(best_acc, avg_epoch_acc))

    # ----------------------------------------------------------------------------------------------------------
    # test

    loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)

    if loaded_checkpoint:
        args, best_epoch_acc, avg_epoch_acc, model, optimizer = loaded_checkpoint

    test_loss, test_acc, test_acc_top2, confmat = test(test_loader, model, criterion, args)

    # save test results
    if not isdir(args.save_test_res):
        os.mkdir(args.save_test_res)

    with open(os.path.join(args.save_test_res, 'raw_test_results.pkl'), 'w') as f:
        pickle.dump([test_loss, test_acc, confmat], f)

    print("Test Acc {}".format(test_acc))

    # todo: need to change the mode here!
    get_metric_from_confmat(confmat, 'b')



def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()

    total_loss = AverageMeter()
    total_acc = AverageMeter()


    # todo: check the round cnt is correct!
    train_loader.dataset.round_cnt={'SingleGaze': 0, 'GazeFollow': 0, 'AvertGaze': 0, 'MutualGaze': 0, 'JointAtt': 0}

    #print(train_loader.dataset.round_cnt)
    # ------------------------------------------------
    # start iteration over current epoch
    for i, (patches, poses, sl_gt, num_rec, attmat_gt) in enumerate(train_loader):
        # sl_gt [N, 10, max_node_num]

        batch_size=sl_gt.shape[0]
        #assert batch_size==args.batch_size, 'wrong batch size!{} {}'.format(batch_size, args.batch_size)

        optimizer.zero_grad()

        if args.cuda:
            patches = (torch.autograd.Variable(patches)).cuda()
            poses = (torch.autograd.Variable(poses)).cuda()
            sl_gt= (torch.autograd.Variable(sl_gt)).cuda()
            num_rec=(torch.autograd.Variable(num_rec)).cuda()
            attmat_gt=(torch.autograd.Variable(attmat_gt)).cuda()

        with torch.set_grad_enabled(True):
            # forward and calculate loss
            sl_pred, sl_pred0= model(patches, poses, num_rec, attmat_gt)
            # sl_pred [N, 10, 6 , 7] attmat_pred [N, 10, 6, 6]


            train_loss=0
            for bid in range(batch_size):
                for sq_id in range(10):
                    valid_node_num = num_rec[bid,sq_id]
                    for nid in range(valid_node_num):
                        tmp_loss = criterion[0](sl_pred[bid, sq_id, nid, :].unsqueeze(0), sl_gt[bid, sq_id, nid].unsqueeze(0)) \
                                   + criterion[0](sl_pred0[bid, sq_id, nid, :].unsqueeze(0), sl_gt[bid, sq_id, nid].unsqueeze(0))

                        # print('label loss', criterion[0](sl_pred[nid][bid, :].unsqueeze(0), sl_gt[bid, nid].unsqueeze(0)))
                        # print('attmat loss', criterion[1](attmat_pred, attmat_gt))

                        total_loss.update(tmp_loss.item(), 1)
                        train_loss = train_loss + tmp_loss

                        pred = torch.argmax(sl_pred[bid, sq_id, nid, :], dim=-1)
                        bv = (pred == sl_gt[bid, sq_id, nid].data)
                        bv = bv.type(torch.cuda.FloatTensor)
                        total_acc.update(bv.item(), 1)

                        # auc=utils.MAUC(sl_gt[bid, nid].data, sl_pred[nid][bid, :], 7)

                        # total_auc.update(auc.item(), 1)

                        # assert len(sl_pred[bid, nid, :])==7, "wrong class number!"

            train_loss.backward()
            #plot_grad_flow(model.named_parameters())
            optimizer.step()

            print('Epoch: {}/{} Iter: {} Training Loss: {:.4f} Total Avg Acc: {:.4f}'.format(epoch, args.epochs - 1, i,
                                                                                            train_loss.item(),
                                                                                            total_acc.avg))

            # save tmp checkpoint
            if i % 300 == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_epoch_acc': [],
                    'avg_epoch_acc': [],
                    'optimizer': optimizer.state_dict(), 'args': args}, is_best=False, directory=args.resume,
                    version='batch_{}'.format(str(i)))

    #print(train_loader.dataset.round_cnt)

    return total_loss.avg, total_acc.avg


def validate(validate_loader, model, criterion, epoch, args):
    model.eval()

    total_acc = AverageMeter()
    total_loss = AverageMeter()

    for i, (patches, poses, sl_gt, num_rec, attmat_gt) in enumerate(validate_loader):

        batch_size=sl_gt.shape[0]
        #assert batch_size==args.batch_size, 'wrong batch size!'

        if args.cuda:

            patches = (torch.autograd.Variable(patches)).cuda()
            poses = (torch.autograd.Variable(poses)).cuda()
            sl_gt= (torch.autograd.Variable(sl_gt)).cuda()
            num_rec=(torch.autograd.Variable(num_rec)).cuda()
            attmat_gt=(torch.autograd.Variable(attmat_gt)).cuda()

        with torch.set_grad_enabled(False):
            # forward and calculate loss
            sl_pred, sl_pred0 = model(patches, poses, num_rec, attmat_gt)

            #max_node_num=sl_pred.shape[0]
            #assert max_node_num==6, 'max node num is not 6!'

            val_loss=0
            for bid in range(batch_size):
                for sq_id in range(10):
                    valid_node_num = num_rec[bid, sq_id]
                    for nid in range(valid_node_num):

                        tmp_loss = criterion[0](sl_pred[bid, sq_id, nid, :].unsqueeze(0),sl_gt[bid, sq_id, nid].unsqueeze(0)) \
                                   +  criterion[0](sl_pred0[bid, sq_id, nid, :].unsqueeze(0), sl_gt[bid, sq_id, nid].unsqueeze(0))

                        total_loss.update(tmp_loss.item(), 1)
                        val_loss=val_loss+tmp_loss

                        pred = torch.argmax(sl_pred[bid, sq_id,nid, :], dim=-1)
                        bv = (pred == sl_gt[bid, sq_id, nid].data)
                        bv = bv.type(torch.cuda.FloatTensor)
                        total_acc.update(bv.item(),1)

            print('Epoch: {}/{} Iter: {} Validation Loss: {:.4f} Total Avg Acc: {:.4f}'.format(epoch, args.epochs - 1, i,
                                                                                          val_loss.item(),
                                                                                          total_acc.avg))

    return total_loss.avg, total_acc.avg


def test(test_loader, model, criterion, args):

    model.eval()

    total_acc = AverageMeter()
    total_loss = AverageMeter()
    confmat=np.zeros((7,7))
    total_acc_top2=AverageMeter()


    for i, (patches, poses, sl_gt, num_rec, attmat_gt) in enumerate(test_loader):

        batch_size=sl_gt.shape[0]

        if args.cuda:
            patches = (torch.autograd.Variable(patches)).cuda()
            poses = (torch.autograd.Variable(poses)).cuda()
            sl_gt= (torch.autograd.Variable(sl_gt)).cuda()
            num_rec=(torch.autograd.Variable(num_rec)).cuda()
            attmat_gt=(torch.autograd.Variable(attmat_gt)).cuda()


        with torch.set_grad_enabled(False):

            sl_pred, sl_pred0 = model(patches, poses, num_rec, attmat_gt)

            test_loss = 0
            for bid in range(batch_size):
                for sq_id in range(10):
                    valid_node_num = num_rec[bid,sq_id]
                    for nid in range(valid_node_num):
                        tmp_loss = criterion[0](sl_pred[bid, sq_id, nid, :].unsqueeze(0), sl_gt[bid, sq_id, nid].unsqueeze(0)) + \
                                   criterion[0](sl_pred0[bid, sq_id, nid, :].unsqueeze(0),sl_gt[bid, sq_id, nid].unsqueeze(0))

                        total_loss.update(tmp_loss.item(), 1)
                        test_loss += tmp_loss

                        pred = torch.argmax(sl_pred[bid, sq_id, nid, :], dim=-1)
                        bv = (pred == sl_gt[bid, sq_id, nid].data)
                        bv = bv.type(torch.cuda.FloatTensor)
                        total_acc.update(bv.item(), 1)

                        # todo: use top2 acc here!
                        _, sort_ind=torch.sort(sl_pred[bid, sq_id, nid, :], descending=True)
                        pred_2=sort_ind[1]

                        bv2 = (pred == sl_gt[bid, sq_id, nid].data or pred_2 == sl_gt[bid, sq_id, nid].data)
                        bv2 = bv2.type(torch.cuda.FloatTensor)
                        total_acc_top2.update(bv2.item(), 1)

                        confmat[sl_gt[bid, sq_id, nid].data, pred]+=1


            print('Iter: {}, Test Loss: {:.4F}, Top 1 Total Avg Accuracy: {:.4F}, Top 2 Total Avg Accuracy: {:.4F}'.format(i, test_loss.item(), total_acc.avg, total_acc_top2.avg))

    return total_loss.avg, total_acc.avg, total_acc_top2, confmat


def parse_arguments():

    path = dataset.utils.Paths()

    project_name = 'GNN_LSTM_lowdim_iter2_bl'
    parser = argparse.ArgumentParser(description=project_name)

    parser.add_argument('--project-name', default=project_name, help='project name')

    # path settings
    parser.add_argument('--project-root', default=path.project_root, help='project root path')
    parser.add_argument('--tmp-root', default=path.tmp_root, help='checkpoint path')
    parser.add_argument('--data-root', default=path.data_root, help='data path')
    parser.add_argument('--log-root', default=path.log_root, help='log files path')
    parser.add_argument('--resume', default=os.path.join(path.tmp_root, 'checkpoints', project_name),
                        help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default=os.path.join(path.tmp_root, 'test_results', project_name),
                        help='path to save test metrics')

    # optimization options
    parser.add_argument('--load-last-checkpoint', default=False,
                        help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint', default=False,
                        help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda', default=True, help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom', default=False, help='use visdom to visualize loss curve')

    parser.add_argument('--device-ids', default=[0, 1], help='gpu ids')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    #    if args.visdom:
    #        vis = visdom.Visdom()
    #        assert vis.check_connection()

    main(args)



