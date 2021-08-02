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
#from utils import visdom_viz
#import visdom
import pickle
import timeit
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(args):


    args.cuda=args.use_cuda and torch.cuda.is_available()

    train_set, validate_set, test_set, train_loader, validate_loader, test_loader = get_data.get_data_attmat(args)

    model = models.AttMat(args)

    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)

    criterion=torch.nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=1, verbose=True, mode='max')

    #------------------------
    # use multi-gpu

    if args.cuda and torch.cuda.device_count()>1:
        print("Now Using ", len(args.device_ids), " GPUs!")

        model=torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
        criterion = criterion.cuda()

    elif args.cuda:
        model=model.cuda()
        criterion=criterion.cuda()


    if args.load_best_checkpoint:
        loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)

        if loaded_checkpoint:
            args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    if args.load_last_checkpoint:
        loaded_checkpoint = utils.load_last_checkpoint(args, model, optimizer, path=args.resume,version=args.model_load_version)

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


        train_epoch_loss,train_epoch_acc=train(train_loader,model,criterion,optimizer,epoch,args)
        train_epoch_acc_all.append(train_epoch_acc)

        val_epoch_loss, val_epoch_acc=validate(validate_loader,model,criterion,epoch, args)
        val_epoch_acc_all.append(val_epoch_acc)

        print('Epoch {}/{} Training Acc: {:.4f} Validation Acc: {:.4f}'.format(epoch, args.epochs - 1, train_epoch_acc, val_epoch_acc))
        print('*' * 15)

        scheduler.step(val_epoch_acc)

        is_best=val_epoch_acc>best_acc

        if is_best:
            best_acc = val_epoch_acc

        avg_epoch_acc = np.mean(val_epoch_acc_all)

        utils.save_checkpoint({
            'epoch':epoch+1,
            'state_dict':model.state_dict(),
            'best_epoch_acc':best_acc,
            'avg_epoch_acc':avg_epoch_acc,
            'optimizer':optimizer.state_dict(),'args':args},is_best=is_best,directory=args.resume,version='epoch_{}'.format(str(epoch)))

    time_elapsed=time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best Val Acc: {},  Final Avg Val Acc: {}'.format(best_acc, avg_epoch_acc))


#----------------------------------------------------------------------------------------------------------
    # test

    loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)

    if loaded_checkpoint:
        args, best_epoch_acc, avg_epoch_acc, model, optimizer = loaded_checkpoint

    #
    test_loss, test_acc, one_acc, zero_acc = test(test_loader, model, criterion, args)

    print("Test Acc {}, One Acc {}, Zero Acc {}".format(test_acc, one_acc, zero_acc))

    # save test results
    if not isdir(args.save_test_res):
        os.mkdir(args.save_test_res)

    with open(os.path.join(args.save_test_res, 'raw_test_results.pkl'), 'w') as f:
        pickle.dump([test_loss, test_acc, one_acc, zero_acc],f)


def train(train_loader,model,criterion,optimizer,epoch,args):

    model.train()

    total_loss=AverageMeter()
    total_acc=AverageMeter()

    #------------------------------------------------
    # start iteration over current epoch
    for i, (head_patch, pos, att_gt) in enumerate(train_loader):

        optimizer.zero_grad()

        if args.cuda:
            head_patch = (torch.autograd.Variable(head_patch)).cuda()
            pos = (torch.autograd.Variable(pos)).cuda()
            att_gt = (torch.autograd.Variable(att_gt)).cuda()

        with torch.set_grad_enabled(True):
            # forward and calculate loss
            pred_att = model(head_patch, pos)
            train_loss = criterion(pred_att, att_gt)

            train_loss.backward()
            optimizer.step()

            total_loss.update(train_loss.item(), att_gt.shape[0])

            pred=torch.argmax(pred_att, dim=-1)

            bv= (pred == att_gt.data)
            bv=bv.type(torch.cuda.FloatTensor)

            total_acc.update(torch.mean(bv).item(), att_gt.shape[0])

            #assert att_gt.shape[0] ==args.batch_size, 'batch size wrong!'

            print('Epoch: {}/{} Iter: {} Training Loss: {:.4f} Training Acc: {:.4f}'.format(epoch, args.epochs-1, i, train_loss.item(),
                                                                                            torch.mean(bv)))

            # save tmp checkpoint
            if i % 300 == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_epoch_acc': [],
                    'avg_epoch_acc': [],
                    'optimizer': optimizer.state_dict(), 'args':args}, is_best=False, directory=args.resume,
                    version='batch_{}'.format(str(i)))

    return total_loss.avg, total_acc.avg


def validate(validate_loader,model,criterion, epoch, args):

    model.eval()

    total_acc = AverageMeter()
    total_loss= AverageMeter()

    for i, (head_patch, pos, att_gt) in enumerate(validate_loader):

        if args.cuda:
            head_patch = (torch.autograd.Variable(head_patch)).cuda()
            pos = (torch.autograd.Variable(pos)).cuda()
            att_gt = (torch.autograd.Variable(att_gt)).cuda()

        with torch.set_grad_enabled(False):
            # forward and calculate loss
            pred_att = model(head_patch, pos)
            val_loss = criterion(pred_att, att_gt)

            total_loss.update(val_loss.item(), att_gt.shape[0])

            pred=torch.argmax(pred_att, dim=-1)

            bv= (pred == att_gt.data)
            bv=bv.type(torch.cuda.FloatTensor)
            total_acc.update(torch.mean(bv).item(), att_gt.shape[0])

            #assert att_gt.shape[0] ==args.batch_size, 'batch size wrong!'

            print('Epoch: {}/{} Iter: {} Validation Loss: {:.4f} Validation Acc: {:.4f}'.format(epoch, args.epochs-1, i,
                                                                val_loss.item(),torch.mean(bv)))

    return total_loss.avg, total_acc.avg


def test(test_loader, model, criterion, args):

    model.eval()

    total_acc=AverageMeter()
    total_loss=AverageMeter()
    one_acc=AverageMeter()
    zero_acc=AverageMeter()


    for i, (head_patch, pos, att_gt) in enumerate(test_loader):


        if args.cuda:
            head_patch = (torch.autograd.Variable(head_patch)).cuda()
            pos = (torch.autograd.Variable(pos)).cuda()
            att_gt = (torch.autograd.Variable(att_gt)).cuda()

        with torch.set_grad_enabled(False):

            pred_att = model(head_patch, pos)
            test_loss = criterion(pred_att, att_gt)

            total_loss.update(test_loss.item(), att_gt.shape[0])
            
            pred=torch.argmax(pred_att, dim=-1)

            bv= (pred == att_gt.data)
            bv=bv.type(torch.cuda.FloatTensor)

            for j in range(att_gt.shape[0]):
                    if att_gt[j]==1:
                        one_acc.update(pred[j].item(), 1)
                    elif att_gt[j]==0:
                        zero_acc.update(1-pred[j].item(), 1)


            total_acc.update(torch.mean(bv).item(), att_gt.shape[0])

            #assert att_gt.shape[0] == args.batch_size, 'batch size wrong!'

            print('Iter: {} Test Loss: {:.4F} Test Accuracy: {:.4F}'.format(i, test_loss.item(), torch.mean(bv)))


    return total_loss.avg, total_acc.avg, one_acc.avg, zero_acc.avg


def parse_arguments():

    path=dataset.utils.Paths()

    project_name='attmat'

    parser=argparse.ArgumentParser(description=project_name)

    parser.add_argument('--project-name', default=project_name, help='project name')

    # path settings
    parser.add_argument('--project-root',default = path.project_root, help='project root path')
    parser.add_argument('--tmp-root', default= path.tmp_root, help='checkpoint path')
    parser.add_argument('--data-root', default= path.data_root, help='data path')
    parser.add_argument('--log-root', default= path.log_root, help='log files path')
    parser.add_argument('--resume', default= os.path.join(path.tmp_root,'checkpoints', project_name), help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default=os.path.join(path.tmp_root, 'test_results', project_name), help='path to save test metrics')


    # optimization options
    parser.add_argument('--load-last-checkpoint', default=False,help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint',default=False,help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--batch-size',type=int,default=32,help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda',default=True,help='Enables CUDA training')
    parser.add_argument('--epochs',type=int,default=0, help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch',type=int,default=0,help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr',type=float,default=0.001,help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay',type=float,default=0.1,help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--momentum',type=float,default=0.9,help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom',default=False,help='use visdom to visualize loss curve')

    parser.add_argument('--device-ids', default=[0,1], help='gpu ids')

    return parser.parse_args()


if __name__=='__main__':

    args=parse_arguments()


#    if args.visdom:
#        vis = visdom.Visdom()
#        assert vis.check_connection()

    main(args)

    pass


