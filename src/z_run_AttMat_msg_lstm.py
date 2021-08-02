import argparse
import os
import pickle
from os.path import isdir

import numpy as np
import torch
import torch.autograd
import visdom

import dataset.metadata
import dataset.utils
import get_data
import models
import utils


def criterion(output_am, target_am, output_label, target_label, args):
    weight_mask = torch.autograd.Variable(torch.ones(target_am.size()))
    if hasattr(args, 'cuda') and args.cuda:
        weight_mask = weight_mask.cuda()
    link_weight = args.link_weight
    weight_mask += target_am * link_weight

    # TODO: why use MSE
    loss_am = torch.mean(weight_mask * ((output_am - target_am) ** 2))

    # TODO: wrong loss
    loss_label = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0, 0.2, 0.2, 0.2, 0.2])).cuda()(output_label,
                                                                                                target_label)

    # if args.freeze_base_model:
    #
    #     return loss_label
    # else:
    # TODO: the weights for different loss
    return 0.3 * loss_am + 0.7 * loss_label


def main(args):
    args.cuda = args.use_cuda and torch.cuda.is_available()

    train_set, validate_set, test_set, train_loader, validate_loader, test_loader = get_data.get_data_AttMat_msg_lstm(
        args)

    model_args = {'roi_feature_size': args.roi_feature_size, 'edge_feature_size': args.roi_feature_size,
                  'node_feature_size': args.roi_feature_size,
                  'message_size': args.message_size, 'link_hidden_size': args.link_hidden_size,
                  'link_hidden_layers': args.link_hidden_layers, 'propagate_layers': args.propagate_layers,
                  'big_attr_classes': args.big_attr_class_num, 'lstm_hidden_size': args.lstm_hidden_size}

    model = models.AttMat_msg_lstm(model_args, args)

    # TODO: check grads and then to set the learning rate for Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # criterion=torch.nn.CrossEntropyLoss()

    if args.cuda:
        model = model.cuda()
        # criterion=criterion.cuda( )

    if args.load_best_checkpoint:
        loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)

        if loaded_checkpoint:
            args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    if args.load_last_checkpoint:
        loaded_checkpoint = utils.load_last_checkpoint(args, model, optimizer, path=args.resume,
                                                       version=args.model_load_version)
        if loaded_checkpoint:
            args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    train_error_history = list()
    train_loss_history = list()
    val_error_history = list()
    val_loss_history = list()

    best_epoch_error = np.inf

    for epoch in range(args.start_epoch, args.epochs):

        train_error_rate_cur_epoch, train_loss_cur_epoch = train(train_loader, model, criterion, optimizer, epoch, args)
        train_error_history.append(train_error_rate_cur_epoch)
        train_loss_history.append(train_loss_cur_epoch)

        val_error_rate_cur_epoch, val_loss_cur_epoch = validate(validate_loader, model, criterion, args)
        val_error_history.append(val_error_rate_cur_epoch)
        val_loss_history.append(val_loss_cur_epoch)

        # TODO: why use this schedule for adjusting learning rate, there is no need to decrease lr for Adam every epoch
        if epoch > 0 and epoch % 1 == 0:
            args.lr *= args.lr_decay

            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        is_best = val_error_rate_cur_epoch < best_epoch_error

        best_epoch_error = min(val_error_rate_cur_epoch, best_epoch_error)

        avg_epoch_error = np.mean(val_error_history)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_epoch_error': best_epoch_error,
            'avg_epoch_error': avg_epoch_error,
            'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.resume,
            version='epoch_{}'.format(str(epoch)))

        print('best_epoch_error: {}, avg_epoch_error: {}'.format(best_epoch_error, avg_epoch_error))

    # test
    # loaded_checkpoint=utils.load_best_checkpoint(args,model,optimizer,path=args.resume)
    loaded_checkpoint = utils.load_last_checkpoint(args, model, optimizer, path=args.resume,
                                                   version=args.model_load_version)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    test(test_loader, model, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()

    train_loss_all = list()

    for i, (node_feature, edge_feature, AttMat, gt_label, node_num_rec) in enumerate(train_loader):

        optimizer.zero_grad()

        if args.cuda:
            node_feature = torch.autograd.Variable(node_feature.cuda())
            edge_feature = torch.autograd.Variable(edge_feature.cuda())
            AttMat = torch.autograd.Variable(AttMat.cuda())
            gt_label = torch.autograd.Variable(gt_label.cuda())
            node_num_rec = torch.autograd.Variable(node_num_rec.cuda())

        sigmoid_pred_adj_mat, pred_label = model(node_feature, edge_feature, AttMat, node_num_rec, args)

        train_loss = 0

        for sq_idx in range(pred_label.size()[1]):
            valid_node_num = node_num_rec[0, sq_idx]

            train_loss += criterion(sigmoid_pred_adj_mat[0, sq_idx, :valid_node_num, :valid_node_num],
                                    AttMat[0, sq_idx, :valid_node_num, :valid_node_num],
                                    pred_label[0, sq_idx, :valid_node_num, :], gt_label[0, sq_idx, :valid_node_num],
                                    args)

        train_loss_all.append(train_loss.cpu().item())

        # visdom_viz(vis, train_loss_all, win=0, ylabel='training loss over batch', title='HGNN AttMat msg lstm', color='green')

        print('epoch [{}], batch [{}], training loss: {}, lr [{}]'.format(epoch, i, train_loss,
                                                                          optimizer.param_groups[0]['lr']))

        train_loss.backward()
        optimizer.step()

        # TODO: why to decrease the lr every 300 iters when you have decreased the lr every epoch
        if i > 0 and i % 300 == 0:
            args.lr *= args.lr_decay

            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        if i % 100 == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_epoch_error': [],
                'avg_epoch_error': [],
                'optimizer': optimizer.state_dict(), }, is_best=False, directory=args.resume,
                version='batch_{}'.format(str(i)))

    return np.mean(train_loss_all)


def validate(validate_loader, model, criterion, args):
    # TODO: the validation process should directly compute the performance used for final results (e.g. AUC)

    model.eval()

    val_loss_all = list()

    for i, (node_feature, edge_feature, AttMat, gt_label, node_num_rec) in enumerate(validate_loader):

        if args.cuda:
            node_feature = torch.autograd.Variable(node_feature.cuda())
            edge_feature = torch.autograd.Variable(edge_feature.cuda())
            AttMat = torch.autograd.Variable(AttMat.cuda())
            gt_label = torch.autograd.Variable(gt_label.cuda())
            node_num_rec = torch.autograd.Variable(node_num_rec.cuda())

        sigmoid_pred_adj_mat, pred_label = model(node_feature, edge_feature, AttMat, node_num_rec, args)

        val_loss = 0

        for sq_idx in range(pred_label.size()[1]):
            valid_node_num = node_num_rec[0, sq_idx]

            val_loss += criterion(sigmoid_pred_adj_mat[0, sq_idx, :valid_node_num, :valid_node_num],
                                  AttMat[0, sq_idx, :valid_node_num, :valid_node_num],
                                  pred_label[0, sq_idx, :valid_node_num, :], gt_label[0, sq_idx, :valid_node_num],
                                  args)

        val_loss_all.append(val_loss.cpu().item())

        # visdom_viz(vis, val_loss_all, win=0, ylabel='validation loss over batch', title='HGNN AttMat msg lstm',
        #            color='red')

        print('batch [{}], validation loss: {}'.format(i, val_loss))

    return np.mean(val_loss_all)


def test(test_loader, model, args):
    model.eval()

    total_edge_cnt_pos = 0.
    correct_edge_cnt_pos = 0.

    total_edge_cnt_neg = 0.
    correct_edge_cnt_neg = 0.

    confmat = np.zeros((5, 5))

    for i, (node_feature, edge_feature, AttMat, gt_label, node_num_rec) in enumerate(test_loader):

        print('batch {}'.format(i))

        if args.cuda:
            node_feature = torch.autograd.Variable(node_feature.cuda())
            edge_feature = torch.autograd.Variable(edge_feature.cuda())
            AttMat = torch.autograd.Variable(AttMat.cuda())
            gt_label = torch.autograd.Variable(gt_label.cuda())
            node_num_rec = torch.autograd.Variable(node_num_rec.cuda())

        sigmoid_pred_adj_mat, pred_label = model(node_feature, edge_feature, AttMat, node_num_rec, args)

        pred_label = pred_label.data.cpu().numpy()
        gt_label = gt_label.data.cpu().numpy()

        for sq_idx in range(node_num_rec.size()[1]):
            valid_node_num = node_num_rec[0, sq_idx]

            for idx1 in range(valid_node_num):

                confmat[int(gt_label[0, sq_idx, idx1]), int(np.argmax(pred_label[0, sq_idx, idx1, :]))] += 1

                for idx2 in range(valid_node_num):
                    if idx2 == idx1:
                        continue

                    if int(AttMat[0, sq_idx, idx1, idx2]) == 1:

                        correct_edge_cnt_pos += int(
                            int(sigmoid_pred_adj_mat[0, sq_idx, idx1, idx2] > args.attmat_threshold) == int(
                                AttMat[0, sq_idx, idx1, idx2]))
                        # print('pred {} true {} cnt {} (thres {})'.format(pred_AttMat[b_idx,idx1,idx2],int(AttMat[b_idx,idx1, idx2]),
                        #                                       int(int(pred_AttMat[b_idx,idx1,idx2]>args.attmat_threshold)==int(AttMat[b_idx,idx1, idx2])),
                        #                                                  args.attmat_threshold))

                        total_edge_cnt_pos += 1

                    else:

                        correct_edge_cnt_neg += int(
                            int(sigmoid_pred_adj_mat[0, sq_idx, idx1, idx2] > args.attmat_threshold) == int(
                                AttMat[0, sq_idx, idx1, idx2]))
                        # print('pred {} true {} cnt {} (thres {})'.format(pred_AttMat[b_idx,idx1,idx2],int(AttMat[b_idx,idx1, idx2]),
                        #                                       int(int(pred_AttMat[b_idx,idx1,idx2]>args.attmat_threshold)==int(AttMat[b_idx,idx1, idx2])),
                        #                                                  args.attmat_threshold))

                        total_edge_cnt_neg += 1

        print('[AttMat] Testing {}, pos acc now: {} neg acc now: {} (threshold: {})'.format(i,
                                                                                            correct_edge_cnt_pos / total_edge_cnt_pos,
                                                                                            correct_edge_cnt_neg / total_edge_cnt_neg,
                                                                                            args.attmat_threshold))

        # utils.get_metric_from_confmat(confmat)

    print('[AttMat] Final, pos acc: {} neg acc: {}  total acc {} (threshold: {})'.format(
        correct_edge_cnt_pos / total_edge_cnt_pos,
        correct_edge_cnt_neg / total_edge_cnt_neg,
        (correct_edge_cnt_neg + correct_edge_cnt_pos) / (total_edge_cnt_neg + total_edge_cnt_pos),
        args.attmat_threshold))

    utils.get_metric_from_confmat(confmat)

    if not isdir(args.save_test_res):
        os.mkdir(args.save_test_res)

    with open(os.path.join(args.save_test_res, 'raw_test_results.pkl'), 'w') as f:
        pickle.dump([correct_edge_cnt_pos, total_edge_cnt_pos, correct_edge_cnt_neg, total_edge_cnt_neg, confmat], f)


def parse_arguments():
    path = dataset.utils.Paths()

    parser = argparse.ArgumentParser(description='run AttMat msg lstm')

    # path settings
    parser.add_argument('--project-root', default=path.project_root, help='project root path')
    parser.add_argument('--tmp-root', default=path.tmp_root, help='intermediate result path')
    parser.add_argument('--data-root', default=path.data_root, help='data path')
    parser.add_argument('--log-root', default=path.log_root, help='log files path')

    # model settings
    parser.add_argument('--small-attr-class-num', type=int, default=7, help='small attribute class number')
    parser.add_argument('--big-attr-class-num', type=int, default=5, help='big attribute class number')
    parser.add_argument('--roi-feature-size', type=int, default=512, help='node and edge feature size')
    parser.add_argument('--message-size', type=int, default=512, help='message size of the message function')
    parser.add_argument('--link-hidden-size', type=int, default=512, help='link hidden size of the link function')
    parser.add_argument('--link-hidden-layers', type=int, default=2, help='link hidden layers of the link function')
    parser.add_argument('--propagate-layers', type=int, default=2, help='propagate layers for message passing')
    parser.add_argument('--lstm-seq-size', type=int, default=5, help='lstm sequence length')
    parser.add_argument('--lstm-hidden-size', type=int, default=256, help='hiddden state size of lstm')
    parser.add_argument('--attmat-threshold', type=float, default=0.6, help='attention matrix threshold')
    parser.add_argument('--link-weight', type=float, default=5, metavar='N', help='Loss weight of existing edges')

    parser.add_argument('--resume', default=os.path.join(path.tmp_root, 'checkpoints', 'graph', 'attmat_msg_lstm3'),
                        help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default=os.path.join(path.tmp_root, 'test_results', 'attmat_msg_lstm3'),
                        help='path to save test metrics')
    parser.add_argument('--load-base-model', default=True, help=' to load pretrained base model')
    parser.add_argument('--base-model-weight',
                        default=os.path.join(path.tmp_root, 'checkpoints', 'graph', 'attmat_msg_new1',
                                             'checkpoint_batch_300.pth'), metavar='N',
                        help='pretrained base model checkpoint file')

    parser.add_argument('--freeze-base-model', default=False, help='to freeze the base model')
    parser.add_argument('--load-last-checkpoint', default=False,
                        help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint', default=False,
                        help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--model_load_version', default='batch_300', help='model load version')

    # optimization options
    # TODO: try to use larger batch size for optimizing BN layers (at least 8)
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda', default=True, help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.8, help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom', default=True, help='use visdom to visualize loss curve')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    if args.visdom:
        vis = visdom.Visdom()
        assert vis.check_connection()

    main(args)
