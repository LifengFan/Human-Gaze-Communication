import sys
sys.path.append("/Users/txy15/Dropbox/Summer18/RunComm/src/")
import shutil
import torch
import torch.utils.data
import os
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from dataset import metadata
import random

import itertools
import pickle
import os.path as op
from PIL import Image,  ImageFont
import PIL.ImageDraw as ImageDraw


"""
    MAUCpy
    ~~~~~~
    Contains two equations from Hand and Till's 2001 paper on a multi-class
    approach to the AUC. The a_value() function is the probabilistic approximation
    of the AUC found in equation 3, while MAUC() is the pairwise averaging of this
    value for each of the classes. This is equation 7 in their paper.
"""


def a_value(y_true, y_pred_prob, zero_label=0, one_label=1):
    """
    Approximates the AUC by the method described in Hand and Till 2001,
    equation 3.

    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the predicted
    probability list.

    Args:
        y_true: actual labels of test data
        y_pred_prob: predicted class probability
        zero_label: label for positive class
        one_label: label for negative class
    Returns:
        The A-value as a floating point.
    """

    idx = np.isin(y_true, [zero_label, one_label])
    labels = y_true[idx]
    prob = y_pred_prob[idx, zero_label]
    sorted_ranks = labels[np.argsort(prob)]

    n0, n1, sum_ranks = 0, 0, 0
    n0 = np.count_nonzero(sorted_ranks == zero_label)
    n1 = np.count_nonzero(sorted_ranks == one_label)
    sum_ranks = np.sum(np.where(sorted_ranks == zero_label)) + n0

    return (sum_ranks - (n0 * (n0 + 1) / 2.0)) / float(n0 * n1)  # Eqn 3


def MAUC(y_true, y_pred_prob, num_classes):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.

    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.

    Args:
        y_true: actual labels of test data
        y_pred_prob: predicted class probability
        zero_label: label for positive class
        one_label: label for negative class
        num_classes (int): The number of classes in the dataset.

    Returns:
        The MAUC as a floating point value.
    """
    # Find all pairwise comparisons of labels
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]

    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(y_true, y_pred_prob, zero_label=pairing[0], one_label=pairing[1]) +
                      a_value(y_true, y_pred_prob, zero_label=pairing[1], one_label=pairing[0])) / 2.0

    return sum_avals * (2 / float(num_classes * (num_classes - 1)))  # Eqn 7


def load_best_checkpoint(args,model,optimizer,path):

    if path:
       checkpoint_dir=path
       best_model_file=os.path.join(checkpoint_dir,'model_best.pth')

       if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
       if os.path.isfile(best_model_file):
            print("====> loading best model {}".format(best_model_file))

            checkpoint=torch.load(best_model_file)
            args.start_epoch=checkpoint['epoch']
            best_epoch_error=checkpoint['best_epoch_acc']

            try:
                avg_epoch_error=checkpoint['avg_epoch_acc']
            except KeyError:
                avg_epoch_error=np.inf

            model_dict = model.state_dict()
            pretrained_model = checkpoint['state_dict']

            pretrained_dict = {}

            for k, v in pretrained_model.items():
                if k[len('module.'):] in model_dict:
                    pretrained_dict[k[len('module.'):]] = v

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            optimizer.load_state_dict(checkpoint['optimizer'])

            if args.cuda:
                model.cuda()

            print("===> loaded best model {} (epoch {})".format(best_model_file,checkpoint['epoch']))

            return args, best_epoch_error, avg_epoch_error, model, optimizer

       else:
           print('===> no best model found at {}'.format(best_model_file))

    else:

        return None

def load_last_checkpoint(args,model,optimizer,path,version):

    if path:
       checkpoint_dir=path
       best_model_file=os.path.join(checkpoint_dir,'checkpoint_'+version+'.pth')

       if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
       if os.path.isfile(best_model_file):

            checkpoint=torch.load(best_model_file)
            args.start_epoch=checkpoint['epoch']
            best_epoch_error=checkpoint['best_epoch_error']

            try:
                avg_epoch_error=checkpoint['avg_epoch_error']
            except KeyError:
                avg_epoch_error=np.inf

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            if args.cuda:
                model.cuda()

            print("===> loaded last model {} (epoch {})".format(best_model_file,checkpoint['epoch']))

            return args, best_epoch_error, avg_epoch_error, model, optimizer

       else:
           print('===> no last model found at {}'.format(best_model_file))

    else:

        return None

def save_checkpoint(state,is_best,directory, version):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file=os.path.join(directory,'checkpoint_'+version+'.pth')
    best_model_file=os.path.join(directory,'model_best.pth')

    torch.save(state,checkpoint_file)

    if is_best:

        shutil.copyfile(checkpoint_file,best_model_file)


#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')

#    print(cm)

#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)

#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.tight_layout()

def get_saved_test_res(save_path):

    # save_path  args.save_test_res, 'raw_test_results.pkl'

    pickle_in = open(save_path, "rb")
    test_loss, test_acc, confmat = pickle.load(pickle_in)

    get_metric_from_confmat(confmat, 's')



def get_test_metric(pred_labels,gt_labels,args):

    # pred_labels [N, class_num] probability array
    # gt_labels [N, class_num] one-hot array

    class_num=gt_labels.shape[-1]

    TP_cnt=np.zeros(class_num)
    TN_cnt=np.zeros(class_num)
    FP_cnt=np.zeros(class_num)
    FN_cnt=np.zeros(class_num)


    for idx in range(gt_labels.shape[0]):

        true_c=np.argmax(gt_labels[idx,:])
        pred_c=np.argmax(pred_labels[idx,:])

        if pred_c==true_c:

            TP_cnt[true_c]+=1

            TN_cnt+=1
            TN_cnt[true_c]-=1
        else:

            FN_cnt[true_c]+=1
            FP_cnt[pred_c] += 1

            TN_cnt+=1
            TN_cnt[true_c]-=1
            TN_cnt[pred_c]-=1


    # recall=tp/(tp+fn)
    recall=np.zeros(class_num)
    for c_id in range(class_num):
        try:
            recall[c_id]=float(TP_cnt[c_id])/float(TP_cnt[c_id]+FN_cnt[c_id])
        except:
            recall[c_id]=-1

    # precision=tp/(tp+fp)
    precision=np.zeros(class_num)
    for c_id in range(class_num):
        try:
            precision[c_id]=float(TP_cnt[c_id])/float(TP_cnt[c_id]+FP_cnt[c_id])
        except:
            precision[c_id]=-1

    # F1 score=2*precision*recall/(precision+recall)
    F_one_score=np.zeros(class_num)
    for c_id in range(class_num):
        try:
            F_one_score[c_id]=2*precision[c_id]*recall[c_id]/(precision[c_id]+recall[c_id])
        except:
            F_one_score[c_id]=-1

    # Average accuracy
    acc=np.zeros(class_num)
    for c_id in range(class_num):
        try:
            acc[c_id]=float(TP_cnt[c_id]+TN_cnt[c_id])/float(TP_cnt[c_id]+TN_cnt[c_id]+FP_cnt[c_id]+FN_cnt[c_id])
        except:
            acc[c_id]=-1

    avg_acc=np.mean(acc)

    # confusion matrix
    ConfMat=confusion_matrix(np.argmax(gt_labels,1),np.argmax(pred_labels,1))

    return recall, precision, F_one_score, acc, avg_acc, ConfMat



def evaluation(pred_node_labels, gt_node_labels):

    #todo: to check whether the evaluation is right

    np_pred_node_labels=pred_node_labels.data.cpu().numpy()
    np_gt_node_labels=gt_node_labels.data.cpu().numpy()

    error_count=0
    total_nodes=0

    for batch_idx in range(np_pred_node_labels.shape[0]):

        total_nodes+=np_pred_node_labels.shape[1]

        #assert len(np_pred_node_labels[batch_idx,:,:].shape)==2

        pred_inds=np.argmax(np_pred_node_labels[batch_idx,:,:],1)
        gt_inds=np_gt_node_labels[batch_idx,:]

        error_count+=np.sum(pred_inds!=gt_inds)

    return error_count/float(total_nodes)



def visdom_viz(vis,curve_vec, win, ylabel, title, color):

    try:
        plt.plot(curve_vec,color=color)
        plt.ylabel(ylabel)
        plt.title(title)

        vis.matplot(plt, win=win)

    except BaseException as err:

        print('Error message: ', err)



def get_metric_from_confmat(confmat, mode):

    # mode ['s' , 'b']

    #N= 7 if mode=='s' else 6 # class number, including NA

    if mode=='s':
        N=7
    elif mode=='b':
        N=6
    elif mode=='atomic':
        N=6
    elif mode=='4class':
        N=4

    recall=np.zeros(N)
    precision=np.zeros(N)
    F_score=np.zeros(N)
    Acc=np.zeros(N)

    correct_cnt=0.
    total_cnt=0.

    for i in range(N):

        recall[i]=confmat[i,i]/(np.sum(confmat[i,:])+1e-7)

        precision[i]=confmat[i,i]/(np.sum(confmat[:,i])+1e-7)

        F_score[i]=2*precision[i]*recall[i]/(precision[i]+recall[i]+1e-7)

        correct_cnt+=confmat[i,i]

        total_cnt+=np.sum(confmat[i,:])

    acc=correct_cnt/total_cnt

    if mode=='s':

        #small_class={'NA': 0, 'single': 1, 'mutual': 2, 'avert': 3, 'refer': 4, 'follow': 5, 'share': 6}

        print('===> Confusion Matrix for Small Label: \n {}'.format(confmat.astype(int)))

        print('===> Precision: \n [NA]: {} % \n [single]: {} % \n [mutual]: {} % \n [avert]: {} % \n [refer]: {} % \n [follow]: {} % \n [share]: {} %'
          .format(precision[0]*100, precision[1]*100, precision[2]*100, precision[3]*100, precision[4]*100, precision[5]*100, precision[6]*100))

        print('===> Recall: \n [NA]: {} % \n [single]: {} % \n [mutual]: {} % \n [avert]: {} % \n [refer]: {} % \n [follow]: {} % \n [share]: {} %'
          .format(recall[0]*100, recall[1]*100, recall[2]*100, recall[3]*100, recall[4]*100, recall[5]*100, recall[6]*100))

        print('===> F score: \n [NA]: {}  \n [single]: {}  \n [mutual]: {}  \n [avert]: {}  \n [refer]: {}  \n [follow]: {}  \n [share]: {} '
          .format(F_score[0], F_score[1], F_score[2], F_score[3], F_score[4], F_score[5], F_score[6]))

        print('===> Accuracy: {} %'.format(acc*100))

    elif mode=='b':

        # big_class = {'NA': 0, 'SingleGaze': 1, 'GazeFollow': 2, 'AvertGaze': 3, 'MutualGaze': 4, 'JointAtt': 5}

        print('===> Confusion Matrix for Big Label: \n {}'.format(confmat.astype(int)))

        print('===> Precision: \n [NA]: {} % \n [SingleGaze]: {} % \n [GazeFollow]: {} % \n [AvertGaze]: {} % \n [MutualGaze]: {} % \n [JointAtt]: {} %'
          .format(precision[0]*100, precision[1]*100, precision[2]*100, precision[3]*100, precision[4]*100, precision[5]*100))

        print('===> Recall: \n [NA]: {} % \n [SingleGaze]: {} % \n [GazeFollow]: {} % \n [AvertGaze]: {} % \n [MutualGaze]: {} % \n [JointAtt]: {} %'
          .format(recall[0]*100, recall[1]*100, recall[2]*100, recall[3]*100, recall[4]*100, recall[5]*100))

        print('===> F score: \n [NA]: {}  \n [SingleGaze]: {}  \n [GazeFollow]: {}  \n [AvertGaze]: {}  \n [MutualGaze]: {}  \n [JointAtt]: {} '
          .format(F_score[0], F_score[1], F_score[2], F_score[3], F_score[4], F_score[5]))

        print('===> Accuracy: {} %'.format(acc*100))

    elif mode=='atomic':

        print('===> Confusion Matrix for Atomic Label: \n {}'.format(confmat.astype(int)))

        print('===> Precision: \n  [single]: {} % \n [mutual]: {} % \n [avert]: {} % \n [refer]: {} % \n [follow]: {} % \n [share]: {} %'
          .format(precision[0]*100, precision[1]*100, precision[2]*100, precision[3]*100, precision[4]*100, precision[5]*100))

        print('===> Recall: \n  [single]: {} % \n [mutual]: {} % \n [avert]: {} % \n [refer]: {} % \n [follow]: {} % \n [share]: {} %'
          .format(recall[0]*100, recall[1]*100, recall[2]*100, recall[3]*100, recall[4]*100, recall[5]*100))

        print('===> F score: \n  [single]: {}  \n [mutual]: {}  \n [avert]: {}  \n [refer]: {}  \n [follow]: {} % \n [share]: {} '
          .format(F_score[0], F_score[1], F_score[2], F_score[3], F_score[4], F_score[5]))

        print('===> Accuracy: {} %'.format(acc*100))

    elif mode=='4class':

        print('===> Confusion Matrix for Atomic Label (4 class): \n {}'.format(confmat.astype(int)))

        print('===> Precision: \n  [single]: {} % \n [mutual]: {} % \n [transient]: {} % \n [share]: {} % \n'
          .format(precision[0]*100, precision[1]*100, precision[2]*100, precision[3]*100))

        print('===> Recall: \n  [single]: {} % \n [mutual]: {} % \n [transient]: {} % \n [share]: {} % \n'
          .format(recall[0]*100, recall[1]*100, recall[2]*100, recall[3]*100))

        print('===> F score: \n  [single]: {}  \n [mutual]: {}  \n [transient]: {}  \n [share]: {} \n '
          .format(F_score[0], F_score[1], F_score[2], F_score[3]))

        print('===> Accuracy: {} %'.format(acc*100))



def visualize_dataset():

    for v in range(1,303):

        vid=str(v)

      # for k in range(1,2):
      #   sq_id=str(k)

        #seq = np.load(op.join('/home/lfan/Dropbox/Projects/ICCV19/RunComm/data/seqs_10/vid_{}_seq_{}.npy'.format(vid, sq_id)))
        #video=np.load(op.join('/home/lfan/Dropbox/Projects/ICCV19/RunComm/data/ant_processed_complex/vid_{}_ant_all.npy'.format(vid)))

        video = np.load(op.join('/home/lfan/Dropbox/RunComm/data/all/ant_processed/vid_{}_ant_all.npy'.format(vid)))

        if not os.path.exists('/home/lfan/Dropbox/Projects/ICCV19/RunComm/data/viz/' + vid):
            os.mkdir('/home/lfan/Dropbox/Projects/ICCV19/RunComm/data/viz/' + vid)

        for isq in range(len(video)):
            frame = video[isq]
            fid = frame['ant'][0]['frame_ind']

            img = np.load(op.join('/home/lfan/Dropbox/Projects/ICCV19/RunComm/data/img_np/{}/{}.npy'.format(vid, str(int(fid) + 1).zfill(5))))

            im = Image.fromarray(img)
            draw = ImageDraw.Draw(im)

            for i in range(len(frame['ant'])):
                pos = frame['ant'][i]['pos']
                BL = frame['ant'][i]['BigAtt']
                SL = frame['ant'][i]['SmallAtt']

                left1 = int(pos[0])
                top1 = int(pos[1])
                right1 = int(pos[2])
                bottom1 = int(pos[3])
                draw.line([(left1, top1), (left1, bottom1), (right1, bottom1), (right1, top1), (left1, top1)], width=4,
                          fill='red')

                draw.text((left1, top1), BL, fill=(255, 255, 255, 255))
                draw.text((left1, top1 + 10), SL, fill=(255, 255, 255, 255))

            attmat = frame['attmat']

            for i in range(attmat.shape[0]):
                for j in range(attmat.shape[1]):

                    if attmat[i, j] == 1:
                        pos1 = frame['ant'][i]['pos']
                        pos2 = frame['ant'][j]['pos']

                        center1_x = (int(pos1[0]) + int(pos1[2])) / 2
                        center1_y = (int(pos1[1]) + int(pos1[3])) / 2

                        center2_x = (int(pos2[0]) + int(pos2[2])) / 2
                        center2_y = (int(pos2[1]) + int(pos2[3])) / 2

                        mid_x = (center1_x + center2_x) / 2
                        mid_y = (center1_y + center2_y) / 2

                        draw.line([(center1_x, center1_y), (mid_x, mid_y)], width=5, fill='green')

            im.save('/home/lfan/Dropbox/Projects/ICCV19/RunComm/data/viz/' + vid + '/' + str(isq) + '.jpg')


def main():

    #get_saved_test_res(os.path.join('/home/lfan/Dropbox/Projects/ICCV19/RunComm/tmp/test_results/','GNN_LSTM_lowdim','raw_test_results.pkl'))

    visualize_dataset()

    pass


if __name__=='__main__':
    main()