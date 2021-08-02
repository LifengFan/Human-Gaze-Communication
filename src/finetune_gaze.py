import sys
#sys.path.append("/home/tangxy/RunGaze/src/")
#sys.path.append("/home/lfan/Dropbox/Projects/ICCV19/RunComm/src/")
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
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import math
import pickle
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau



print("PyTorch Version: ", torch.__version__)



def adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // 50))
    elif args.lr_mode == 'poly':
        current_step = epoch * iters_per_epoch + i_iter
        max_step = args.epochs * iters_per_epoch
        lr = args.lr * (1 - current_step / max_step) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    optimizer.param_groups[0]['lr'] = lr
    return lr


def main(args):

    args.cuda = args.use_cuda and torch.cuda.is_available()
    train_set, validate_set, test_set, train_loader, validate_loader, test_loader = get_data.get_data_headpose(args)
    model = models.Gaze(args)
    # TODO: try to use the step policy for Adam, also consider the step interval
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # TODO: related to line 99 (e.g. max for auc, min for loss; try to check the definition of this method)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=1, verbose=True, mode='min')
    # TODO; double check this loss, also the output of the network
    criterion = torch.nn.MSELoss()

    #------------------------
    # use multi-gpu

    if args.cuda and torch.cuda.device_count()>1:
        print("Now Using ", len(args.device_ids), " GPUs!")

        #model=model.to(device_ids[0])
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

    #------------------------------------------------------------------------------
    # Train

    since=time.time()

    train_epoch_loss_all = []
    val_epoch_loss_all = []


    best_loss=np.inf
    avg_epoch_loss=np.inf

    for epoch in range(args.start_epoch, args.epochs):

        train_epoch_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        train_epoch_loss_all.append(train_epoch_loss)
        #visdom_viz(vis, train_epoch_loss_all, win=0, ylabel='Training Epoch Loss', title=args.project_name, color='green')

        val_epoch_loss=validate(validate_loader, model, criterion, epoch, args)
        val_epoch_loss_all.append(val_epoch_loss)
        #visdom_viz(vis, val_epoch_loss_all, win=1, ylabel='Validation Epoch Loss', title=args.project_name,color='blue')

        print('Epoch {}/{} Training Loss: {:.4f} Validation Loss: {:.4f}'.format(epoch, args.epochs - 1, train_epoch_loss, val_epoch_loss))
        print('*' * 15)


        #TODO: reducing lr when there is no gains on validation metric results (e.g. auc, loss)
        scheduler.step(val_epoch_loss)

        is_best = val_epoch_loss < best_loss

        if is_best:
            best_loss = val_epoch_loss

        avg_epoch_loss = np.mean(val_epoch_loss_all)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_epoch_error': best_loss,
            'avg_epoch_error': avg_epoch_loss,
            'optimizer': optimizer.state_dict(), 'args':args}, is_best=is_best, directory=args.resume,version='epoch_{}'.format(str(epoch)))


    time_elapsed=time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best Val Loss: {},  Final Avg Val Loss: {}'.format(best_loss, avg_epoch_loss))


    #-------------------------------------------------------------------------------------------------------------
    # test
    loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    pred_rpy, gt_rpy, test_loss = test(test_loader, model, criterion, args)

    print("Test Epoch Loss {}".format(test_loss))

    # save test results
    if not isdir(args.save_test_res):
        os.mkdir(args.save_test_res)

    with open(os.path.join(args.save_test_res, 'raw_test_results.pkl'), 'w') as f:
        pickle.dump([pred_rpy, gt_rpy, test_loss], f)



def train(train_loader, model, criterion, optimizer, epoch, args):

    model.train()

    #train_error_rate_all = list()
    running_loss=0
    node_cnt=0
    loss_on=[]

    # iterate over training data
    # head_patch (N,3,224,224) rpy (N,3)
    for i, (head_patch, gt_rpy) in enumerate(train_loader):

        if args.cuda:
            head_patch = torch.autograd.Variable(head_patch.cuda())
            gt_rpy = torch.autograd.Variable(gt_rpy.cuda())

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward and calculate loss
            pred_rpy = model(head_patch)

            #print("Outside: input_size", node_feature.size(), "output_size", pred_label.size())

            # error_rate = evaluation(pred_label.unsqueeze(0), gt_label.unsqueeze(0))
            # train_error_rate_all.append(error_rate)
            #todo: check the dimension change, see if it's valid or not

            train_loss = criterion(pred_rpy, gt_rpy)

            train_loss.backward()
            optimizer.step()

            loss_on.append(train_loss.item())
            print('Epoch: {}/{} Iter: {} Training Loss: {:.4F}'.format(epoch, args.epochs,i, train_loss.item()))
            # if args.visdom:
            #     visdom_viz(vis, loss_on, win=2, ylabel='Train Online Loss', title=args.project_name, color='pink')

        # train_loss = criterion(pred_label.view(-1, 6), gt_label)
        running_loss+=train_loss.item()*pred_rpy.shape[0]
        node_cnt+=pred_rpy.shape[0]

        # save tmp checkpoint
        if i % 300 == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_epoch_error': [],
                'avg_epoch_error': [],
                'optimizer': optimizer.state_dict(), 'args':args}, is_best=False, directory=args.resume,version='batch_{}'.format(str(i)))

        # reducing lr during iterations, there is no need to reducing the lr every epoch
        # below is the 'poly' lr policy always used for changing lr during iteration
        # update lr during iteration
        # todo: lr update criterion
        # if True:
        #     iters_per_epoch = len(train_loader)
        #     lr = adjust_learning_rate(optimizer, epoch, i, iters_per_epoch)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr


    epoch_loss=running_loss/node_cnt


    return epoch_loss


def validate(validate_loader, model, criterion, epoch, args):

    model.eval()

    running_loss=0
    node_cnt=0
    loss_on=[]

    # iterate over validation data
    for i, (head_patch, gt_rpy) in enumerate(validate_loader):

        if args.cuda:
            head_patch = torch.autograd.Variable(head_patch.cuda())
            gt_rpy = torch.autograd.Variable(gt_rpy.cuda())

        with torch.set_grad_enabled(False):
            pred_rpy = model(head_patch)
            val_loss = criterion(pred_rpy, gt_rpy)

            loss_on.append(val_loss.item())
            print('Epoch: {}/{} Iter: {} Validation Loss: {:.4F}'.format(epoch, args.epochs,i, val_loss.item()))
            # if args.visdom:
            #    visdom_viz(vis, loss_on, win=3, ylabel='Validate Online Loss', title=args.project_name, color='red')



        running_loss+=val_loss.item()*pred_rpy.shape[0]
        node_cnt+=pred_rpy.shape[0]

    epoch_loss = running_loss / node_cnt

    return epoch_loss


def test(test_loader, model, criterion, args):

    model.eval()

    pred_rpy_all=[]
    gt_rpy_all=[]

    running_loss=0
    node_cnt=0
    loss_on=[]
    for i, (head_patch, gt_rpy) in enumerate(test_loader):

        if args.cuda:
            head_patch = torch.autograd.Variable(head_patch.cuda())
            gt_rpy = torch.autograd.Variable(gt_rpy.cuda())

        with torch.set_grad_enabled(False):
            pred_rpy = model(head_patch)
            test_loss = criterion(pred_rpy, gt_rpy)
            loss_on.append(test_loss.item())
            print('Iter: {} Test Loss: {:.4F}'.format( i, test_loss.item()))
            # if args.visdom:
            #     visdom_viz(vis, loss_on, win=4, ylabel='Test Online Loss', title=args.project_name, color='red')

        running_loss += test_loss.item() * pred_rpy.shape[0]
        node_cnt += pred_rpy.shape[0]

        if i == 0:
            pred_rpy_all = pred_rpy.cpu().numpy()
            gt_rpy_all = gt_rpy.cpu().numpy()

        else:
            pred_rpy_all = np.vstack((pred_rpy_all, pred_rpy.cpu().numpy()))
            gt_rpy_all = np.vstack((gt_rpy_all, gt_rpy.cpu().numpy()))

    epoch_loss = running_loss / node_cnt


    # err_rate = evaluation(pred_label.unsqueeze(0), gt_label.unsqueeze(0))
    # test_err_rate_all.append(err_rate.data.cpu())
    # print('batch [{}]  error rate: {}'.format(i, err_rate))

    return pred_rpy_all, gt_rpy_all, epoch_loss


def parse_arguments():
    path = dataset.utils.Paths()

    project_name='finetune_headpose'

    parser = argparse.ArgumentParser(description=project_name)
    # path settings
    parser.add_argument('--project-name', default=project_name, help='project name')
    parser.add_argument('--project-root', default=path.project_root, help='project root path')
    parser.add_argument('--tmp-root', default=path.tmp_root, help='intermediate result path')
    parser.add_argument('--data-root', default=path.data_root, help='data path')
    parser.add_argument('--log-root', default=path.log_root, help='log files path')
    parser.add_argument('--resume', default=os.path.join(path.tmp_root, 'checkpoints', project_name),help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default=os.path.join(path.tmp_root, 'test_results', project_name),help='path to save test metrics')
    # model settings
    parser.add_argument('--small-attr-class-num', type=int, default=7, help='small attribute class number')
    parser.add_argument('--big-attr-class-num', type=int, default=6, help='big attribute class number')
    parser.add_argument('--roi-feature-size', type=int, default=500, help='node and edge feature size')
    parser.add_argument('--message-size', type=int, default=500, help='message size of the message function')
    parser.add_argument('--link-hidden-size', type=int, default=1024, help='link hidden size of the link function')
    parser.add_argument('--link-hidden-layers', type=int, default=2, help='link hidden layers of the link function')
    parser.add_argument('--propagate-layers', type=int, default=2, help='propagate layers for message passing')

    # optimization options
    parser.add_argument('--load-last-checkpoint', default=False,help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint', default=False,help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda', default=True, help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--lr-mode', default='poly', help='Learning rate decay mode (default : poly)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom', default=False, help='use visdom to visualize loss curve')
    parser.add_argument('--model_load_version', default='batch_300', help='model load version')

    parser.add_argument('--device-ids', default=[0,1,2,3], help='gpu ids')

    return parser.parse_args()


if __name__ == '__main__':

    #tmp=np.load('/home/lfan/Dropbox/RunComm/data/train/ant_processed/vid_1_ant_all.npy')

    torch.cuda.empty_cache()

    args = parse_arguments()
    print args

    # if args.visdom:
    #     vis = visdom.Visdom()
    #     assert vis.check_connection()

    main(args)
