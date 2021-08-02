import sys
sys.path.append("/home/lfan/Dropbox/RunComm/src/")

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
from utils import evaluation, visdom_viz

import visdom
import pickle


def main(args):


    args.cuda = args.use_cuda and torch.cuda.is_available()

    train_set, validate_set, test_set, train_loader, validate_loader, test_loader = get_data.get_data_resnet_msgpassing_balanced_lstm(args)

    model_args = {'roi_feature_size':args.roi_feature_size,'edge_feature_size': args.roi_feature_size, 'node_feature_size': args.roi_feature_size,
                  'message_size': args.message_size, 'link_hidden_size': args.link_hidden_size,
                  'link_hidden_layers': args.link_hidden_layers, 'propagate_layers': args.propagate_layers,
                   'big_attr_classes': args.big_attr_class_num, 'lstm_hidden_size':args.lstm_hidden_size}


    model = models.HGNN_resnet_msgpassing_balanced_lstm(model_args)

    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)

    criterion=torch.nn.CrossEntropyLoss()


    if args.cuda:
        model=model.cuda()
        criterion=criterion.cuda( )

    if args.load_best_checkpoint:
        loaded_checkpoint=utils.load_best_checkpoint(args,model,optimizer,path=args.resume)

        if loaded_checkpoint:
             args, best_epoch_error, avg_epoch_error, model, optimizer=loaded_checkpoint

    if args.load_last_checkpoint:
        loaded_checkpoint=utils.load_last_checkpoint(args,model,optimizer,path=args.resume)

        if loaded_checkpoint:
             args, best_epoch_error, avg_epoch_error, model, optimizer=loaded_checkpoint


    train_error_history=list()
    train_loss_history=list()
    val_error_history=list()
    val_loss_history=list()

    best_epoch_error=np.inf


    for epoch in range(args.start_epoch, args.epochs):


        train_error_rate_cur_epoch,  train_loss_cur_epoch=train(train_loader,model,criterion,optimizer,epoch,args)
        train_error_history.append(train_error_rate_cur_epoch)
        train_loss_history.append(train_loss_cur_epoch)

        val_error_rate_cur_epoch, val_loss_cur_epoch=validate(validate_loader,model,criterion,args)
        val_error_history.append(val_error_rate_cur_epoch)
        val_loss_history.append(val_loss_cur_epoch)


        if epoch>0 and epoch%1==0:
            args.lr*=args.lr_decay

            for param_group in optimizer.param_groups:
                param_group['lr']=args.lr


        is_best=val_error_rate_cur_epoch<best_epoch_error

        best_epoch_error=min(val_error_rate_cur_epoch,best_epoch_error)

        avg_epoch_error=np.mean(val_error_history)

        utils.save_checkpoint({
            'epoch':epoch+1,
            'state_dict':model.state_dict(),
            'best_epoch_error':best_epoch_error,
            'avg_epoch_error':avg_epoch_error,
            'optimizer':optimizer.state_dict(),},is_best=is_best,directory=args.resume)

        print('best_epoch_error: {}, avg_epoch_error: {}'.format(best_epoch_error, avg_epoch_error))


    # test
    #loaded_checkpoint=utils.load_best_checkpoint(args,model,optimizer,path=args.resume)
    loaded_checkpoint = utils.load_last_checkpoint(args, model, optimizer, path=args.resume)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer=loaded_checkpoint

    test(test_loader,model,args)



def train(train_loader,model,criterion,optimizer,epoch,args):


    model.train()

    train_loss_all=list()
    train_error_rate_all=list()

    for i, (node_feature, edge_feature, gt_label,node_num_rec) in enumerate(train_loader):

        optimizer.zero_grad()


        if args.cuda:

            node_feature = torch.autograd.Variable(node_feature.cuda())
            edge_feature = torch.autograd.Variable(edge_feature.cuda())
            gt_label = torch.autograd.Variable(gt_label.cuda())
            node_num_rec=torch.autograd.Variable(node_num_rec.cuda())


        pred_label=model(node_feature, edge_feature,node_num_rec,args)

        for sq_idx in range(pred_label.size()[1]):

            valid_node_num=node_num_rec[0,sq_idx]

            if sq_idx==0:
                pred_label_all=pred_label[0,sq_idx,:valid_node_num,:]
                gt_label_all=gt_label[0,sq_idx,:valid_node_num]

            else:
                pred_label_all=torch.cat((pred_label_all,pred_label[0,sq_idx,:valid_node_num,:]),dim=0)
                gt_label_all=torch.cat((gt_label_all,gt_label[0,sq_idx,:valid_node_num]),dim=0)


        error_rate = evaluation(pred_label.unsqueeze(0), gt_label.unsqueeze(0))
        train_error_rate_all.append(error_rate)

        train_loss=criterion(pred_label_all, gt_label_all)

        train_loss_all.append(train_loss.data.cpu().numpy().item())

        visdom_viz(vis, train_loss_all, win=0, ylabel='training loss over batch', title='HGNN Resnet Msgpassing balanced lstm', color='green')

        print('epoch [{}], batch [{}], training loss: {}, training error rate: {}, lr [{}]'.format(epoch, i,train_loss,error_rate, optimizer.param_groups[0]['lr']))

        train_loss.backward()
        optimizer.step()

        if i > 0 and i % 300 == 0:
            args.lr *= args.lr_decay

            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        if i%200==0:
            utils.save_checkpoint({
            'epoch':epoch+1,
            'state_dict':model.state_dict(),
            'best_epoch_error':[],
            'avg_epoch_error':[],
            'optimizer':optimizer.state_dict(),},is_best=False,directory=args.resume)

        del node_feature, edge_feature, gt_label,node_num_rec

    return np.mean(train_error_rate_all), np.mean(train_loss_all)


def validate(validate_loader,model,criterion,args):

    model.eval()

    val_error_rate_all=list()
    val_loss_all=list()


    for i, (node_feature, edge_feature, gt_label,node_num_rec) in enumerate(validate_loader):

        if args.cuda:

            node_feature = torch.autograd.Variable(node_feature.cuda())
            edge_feature = torch.autograd.Variable(edge_feature.cuda())
            gt_label = torch.autograd.Variable(gt_label.cuda())
            node_num_rec = torch.autograd.Variable(node_num_rec.cuda())

        pred_label= model(node_feature, edge_feature, node_num_rec,args)

        for sq_idx in range(pred_label.size()[1]):

            valid_node_num = node_num_rec[0, sq_idx]

            if sq_idx == 0:
                pred_label_all = pred_label[0, sq_idx, :valid_node_num, :]
                gt_label_all = gt_label[0, sq_idx, :valid_node_num]

            else:
                pred_label_all = torch.cat((pred_label_all, pred_label[0, sq_idx, :valid_node_num, :]), dim=0)
                gt_label_all = torch.cat((gt_label_all, gt_label[0, sq_idx, :valid_node_num]), dim=0)


        error_rate = evaluation(pred_label_all.unsqueeze(0), gt_label_all.unsqueeze(0))
        val_error_rate_all.append(error_rate)

        val_loss = criterion(pred_label_all, gt_label_all)
        val_loss_all.append(val_loss.data.cpu().numpy().item())

        visdom_viz(vis, val_loss_all, win=0, ylabel='validation loss over batch', title='HGNN Resnet Msgpassing balanced lstm', color='red')

        print('batch [{}], validation loss: {}, validation error rate: {}'.format(i, val_loss, error_rate))

        del node_feature, edge_feature, gt_label, node_num_rec

    return np.mean(val_error_rate_all), np.mean(val_loss_all)


def test(test_loader,model,args):

    model.eval()
    confmat = np.zeros((6, 6))

    for i, (node_feature, edge_feature, gt_label, node_num_rec) in enumerate(test_loader):

        print('batch {}'.format(i))

        if args.cuda:

            node_feature = torch.autograd.Variable(node_feature.cuda())
            edge_feature = torch.autograd.Variable(edge_feature.cuda())
            gt_label = torch.autograd.Variable(gt_label.cuda())
            node_num_rec = torch.autograd.Variable(node_num_rec.cuda())

        pred_label = model(node_feature, edge_feature, node_num_rec, args)

        pred_label=pred_label.data.cpu().numpy()
        gt_label=gt_label.data.cpu().numpy()

        for sq_idx in range(pred_label.shape[1]):
            valid_node_num = node_num_rec[0,sq_idx]
            for v_idx in range(valid_node_num):
                confmat[int(gt_label[0,sq_idx,v_idx]),int(np.argmax(pred_label[0,sq_idx,v_idx,:]))]+=1

        del node_feature, edge_feature, gt_label, node_num_rec, pred_label


    if not isdir(args.save_test_res):
            os.mkdir(args.save_test_res)

    with open(os.path.join(args.save_test_res, 'raw_test_results.pkl'), 'w') as f:
        pickle.dump(confmat, f)

    utils.get_metric_from_confmat(confmat)

    #plot_confusion_matrix(confmat, classes=dataset.metadata.BigAtt, normalize=False, title='Confusion matrix')


def parse_arguments():

    path=dataset.utils.Paths()

    parser=argparse.ArgumentParser(description='run HGNN Resnet Msgpassing balanced lstm')

    # path settings
    parser.add_argument('--project-root',default = path.project_root, help='project root path')
    parser.add_argument('--tmp-root', default= path.tmp_root, help='intermediate result path')
    parser.add_argument('--data-root', default= path.data_root, help='data path')
    parser.add_argument('--log-root', default= path.log_root, help='log files path')
    parser.add_argument('--resume', default= os.path.join(path.tmp_root,'checkpoints','graph','hgnn_resnet_msgpassing_balanced_lstm'), help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default=os.path.join(path.tmp_root, 'test_results', 'hgnn_resnet_msgpassing_balanced_lstm'), help='path to save test metrics')
    # model settings
    parser.add_argument('--small-attr-class-num', type=int, default=7, help='small attribute class number')
    parser.add_argument('--big-attr-class-num', type=int, default=6, help='big attribute class number')
    parser.add_argument('--roi-feature-size', type=int, default=512, help='node and edge feature size')
    parser.add_argument('--message-size', type=int, default=512, help='message size of the message function')
    parser.add_argument('--link-hidden-size', type=int, default=512, help='link hidden size of the link function')
    parser.add_argument('--link-hidden-layers', type=int, default=2, help='link hidden layers of the link function')
    parser.add_argument('--propagate-layers', type=int, default=2, help='propagate layers for message passing')
    parser.add_argument('--lstm-seq-size', type=int, default=5, help='lstm sequence length')
    parser.add_argument('--lstm-hidden-size', type=int, default=256, help='hiddden state size of lstm')

    # optimization options
    parser.add_argument('--load-last-checkpoint', default=False,help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint',default=False,help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--batch-size',type=int,default=1,help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda',default=True,help='Enables CUDA training')
    parser.add_argument('--epochs',type=int,default=0,help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch',type=int,default=0,help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr',type=float,default=1e-7,help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay',type=float,default=0.6,help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--momentum',type=float,default=0.9,help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom',default=True,help='use visdom to visualize loss curve')


    return parser.parse_args()



if __name__=='__main__':

    args=parse_arguments()


    if args.visdom:
        vis = visdom.Visdom()
        assert vis.check_connection()

    main(args)



