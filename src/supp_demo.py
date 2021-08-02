import os
import argparse
import numpy as np
import torch
import torch.autograd
import os.path as op
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
from PIL import Image,  ImageFont
import PIL.ImageDraw as ImageDraw
import random
import cv2
import matplotlib.patches as patches


atomic_label = {'single': 0, 'mutual': 1, 'avert': 2, 'refer': 3, 'follow': 4, 'share': 5}
event_label = {'SingleGaze': 0, 'GazeFollow': 1, 'AvertGaze': 2, 'MutualGaze': 3, 'JointAtt': 4}


atomic={0:'single',1:'mutual',2:'avert',3:'refer',4:'follow',5:'share'}
event={0:'Non-communicative',1:'Gaze Following', 2:'Gaze Aversion', 3:'Mutual Gaze', 4:'Joint Attention'}

#----------------------------------------------------------------------------------
# ground truth backup
# Demo_atomic={
#     '1':[1,1,1,1,3,4,5,0,0,0],
#     '5':[0,0,0,0,0,0,0,4,4,5,5,5],
#     '13':[1,1,1,1,3,3,0,4,4,5,5,5,5],
#     '14':[1,1,3,3,4,5,5,5,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
#     '17':[1,1,1,3,3,0,0,4,4,5,5,0,0,5,5,5],
#     '18':[1,1,1,1,1,3,3,3,0,0,0,0,0,4,4],
#     '145':[0,0,4,4,4,5,5],
#     '19':[0,0,0,0,0, 0,0,0,0,1, 1,1,1,1,1, 1,1,1,3,3, 3,4,4,5,5, 5,5,5,5,5, 0,0,1,1,1, 3,4,4,4,5, 5,5,5,0,0, 0,0],
#     '23':[1,3,0,4,4,5,5,5,5,5,1,1,1,1,1,1,5,5,5,5,5,5],
#     '27':[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1],
#     '44':[1,1,3,0,4,0,1,1,3,3,1,1,3,3,1,1,4,4,5],
#     '55':[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,5,0,1, 1,1,1,1,1],
#     '58':[1,1,1,1,3,3,3,4,4,5,5,5],
#     '64':[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1],
#     '70':[1,1,1,1,1, 1,2,2,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0],
#     '72':[0,0,0,0,1,1,2,2,0,0,0],
#     '74':[0,0,0,2,2,2,0,0,0,0],
#     '80':[1,1,1,1,1, 1,1,1,1,1, 1],
#     '81':[1,1,1,1,1, 2,2,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0],
#     '82':[1,1,2,2,2,0,0,0],
#     '87':[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0],
#     '93':[1,1,1,1,1,1],
#     '94':[1,1,1,1,1, 3,3,1,1,1, 1,1,1,1,1, 1,1,1,1,0, 0,0,0,0,0,  0,0,0,0,5, 5,5,0,0,0, 0,0,0,0,1, 1,1],
#     '120':[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
#     '126':[0,0,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
#     '196':[0,0,0,0,0, 0,0,0,0,0],
#     '198':[0,0,0,0,4,5,5,5],
#     '202':[4,4,5,5,5,5,5],
#     '242':[0,0,0,0,4,4,5],
# }
#
#
# Demo_event={
#     '1':[4],
#     '5':[0,0,0,0,1,1,1,1,1,1,1,1],
#     '13':[4],
#     '14':[4],
#     '17':[4],
#     '18':[4],
#     '145':[1],
#     '19':[0,0,0,0,0, 0,0,0,0,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4],
#     '23':[4],
#     '27':[3],
#     '44':[4],
#     '55':[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 3,3,3,3,3],
#     '58':[4],
#     '64':[3],
#     '70':[2],
#     '72':[2],
#     '74':[2],
#     '80':[3],
#     '81':[2,2,2,2,2, 2,2,2,2,2, 2,0,0,0,0, 0,0,0,0,0, 0],
#     '82':[2],
#     '87':[0],
#     '93':[3],
#     '94':[4],
#     '120':[0,0,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3],
#     '126':[3],
#     '196':[0],
#     '198':[1],
#     '202':[1],
#     '242':[1],
# }

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

Demo_atomic={
    '1':[1,1,1,1,3,4,5,0,0,0],
    '5':[0,0,0,0,0,0,0,4,4,5,5,5],
    '13':[1,1,1,1,3,3,0,4,4,5,5,5,5],
    '14':[1,1,3,3,4,5,5,5,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
    '17':[1,1,1,3,3,0,0,4,4,5,5,0,0,5,5,5],
    '18':[1,1,1,1,1,3,3,3,0,0,0,0,0,4,4],
    '145':[0,0,4,4,4,5,5],
    '19':[0,0,0,0,0, 0,0,0,0,1, 1,1,1,1,1, 1,1,1,3,3, 3,4,4,5,5, 5,5,5,5,5, 0,0,1,1,1, 3,4,4,4,5, 5,5,5,0,0, 0,0],
    '23':[1,3,0,4,4,5,5,5,5,5,1,1,1,1,1,1,5,5,5,5,5,5],
    '27':[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1],
    '44':[1,1,3,0,4,0,1,1,3,3,1,1,3,3,1,1,4,4,5],
    '55':[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,5,0,1, 1,1,1,1,1],
    '58':[1,1,1,1,3,3,3,4,4,5,5,5],
    '64':[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1],
    '70':[1,1,1,1,1, 1,2,2,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0],
    '72':[0,0,0,0,1,1,2,2,0,0,0],
    '74':[0,0,0,2,2,2,0,0,0,0],
    '80':[1,1,1,1,1, 1,1,1,1,1, 1],
    '81':[1,1,1,1,1, 2,2,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0],
    '82':[1,1,2,2,2,0,0,0],
    '87':[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0],
    '93':[1,1,1,1,1,1],
    '94':[1,1,1,1,1, 3,3,1,1,1, 1,1,1,1,1, 1,1,1,1,0, 0,0,0,0,0,  0,0,0,0,5, 5,5,0,0,0, 0,0,0,0,1, 1,1],
    '120':[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    '126':[0,0,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
    '196':[0,0,0,0,0, 0,0,0,0,0],
    '198':[0,0,0,0,4,5,5,5],
    '202':[4,4,5,5,5,5,5],
    '242':[0,0,0,0,4,4,5],
}


Demo_event={
    '1':[4],
    '5':[0,0,0,0,1,1,1,1,1,1,1,1],
    '13':[4],
    '14':[4],
    '17':[4],
    '18':[4],
    '145':[1],
    '19':[0,0,0,0,0, 0,0,0,0,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,4],
    '23':[4],
    '27':[3],
    '44':[4],
    '55':[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 3,3,3,3,3],
    '58':[4],
    '64':[3],
    '70':[2],
    '72':[2],
    '74':[2],
    '80':[3],
    '81':[2,2,2,2,2, 2,2,2,2,2, 2,0,0,0,0, 0,0,0,0,0, 0],
    '82':[2],
    '87':[0],
    '93':[3],
    '94':[4],
    '120':[0,0,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3],
    '126':[3],
    '196':[0],
    '198':[1],
    '202':[1],
    '242':[1],
}

#################################################################################################
#################################################################################################

Test_atomic={
    '1': [1, 1, 1, 1, 3, 1, 5, 5, 0, 4],
    '5': [0, 2, 0, 4, 0, 0, 0, 4, 5, 5, 5, 5],
    '13': [1, 1, 1, 1, 3, 3, 0, 3, 4, 5, 5, 0, 5],
    '14': [1, 1, 3, 3, 5, 5, 5, 5, 0, 0, 2, 0, 3, 0, 0, 0, 2, 0, 0, 1, 1, 1, 4, 1],
    '17': [1, 1, 1, 3, 3, 0, 0, 4, 5, 5, 5, 0, 4, 5, 5, 5],
    '18': [1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 4, 0, 0, 4, 0],
    '145': [0, 0, 4, 5, 0, 5, 5],
    '23': [1, 3, 0, 4, 4, 5, 2, 5, 5, 5, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
    '27': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    '44': [1, 1, 3, 0, 4, 0, 1, 1, 3, 3, 1, 1, 3, 3, 1, 1, 4, 4, 5],
    '58': [1, 1, 1, 1, 3, 3, 5, 3, 4, 5, 5, 5],
    '64': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    '70': [1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
    '72': [0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0],
    '74': [0, 0, 0, 2, 2, 1, 0, 0, 0, 0],
    '80': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    '82': [1, 1, 2, 2, 0, 0, 0, 0],
    '87': [0, 4, 0, 0, 0, 0, 3, 3, 0, 4, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0],
    '93': [1, 1, 1, 1, 1, 1],
    '94': [1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1],
    '120': [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    '126': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    '196': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '198': [0, 0, 0, 0, 4, 5, 5, 5],
    '202': [4, 5, 5, 5, 5, 3, 5],
    '242': [0, 3, 0, 0, 4, 0, 0],
}


Test_event={
    '1': [4],
    '5': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    '13': [4],
    '14': [4],
    '17': [4],
    '18': [4],
    '145': [2],
    '23': [4],
    '27': [3],
    '44': [4],
    '58': [4],
    '64': [3],
    '70': [2],
    '72': [2],
    '74': [2],
    '80': [3],
    '82': [2],
    '87': [1],
    '93': [3],
    '94': [4],
    '120': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3],
    '126': [3],
    '196': [0],
    '198': [1],
    '202': [1],
    '242': [0],
}

def draw_demo():

    data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'

    vid_list_normal=['19','55','81']
    vid_list_slow=['94','27','70','5','87','58','23','44','72','74','145','64','1','202','13','14','242','196','17','80','126','198','82','93','18','120']

    vid_list_all=[]
    vid_list_all.extend(vid_list_normal)
    vid_list_all.extend(vid_list_slow)

    for vid_ind in range(len(vid_list_all)):

        vid=vid_list_all[vid_ind]

        vid='242'

        print('the {} video, vid is {}'.format(vid_ind, vid))

        if not os.path.exists(op.join(data_root,'draw_demo_res')):
            os.mkdir(op.join(data_root,'draw_demo_res'))

        if not os.path.exists(op.join(data_root,'draw_demo_res',vid)):
            os.mkdir(op.join(data_root, 'draw_demo_res', vid))

        save_path = op.join(data_root, 'draw_demo_res', vid)

        video=[op.join(data_root, 'supp_demo', vid, f) for f in sorted(os.listdir(op.join(data_root, 'supp_demo', vid)))]
        annot = np.load(op.join(data_root, 'ant_processed', 'vid_{}_ant_all.npy'.format(vid)))

        for f_j in range(len(video)):
            fid=video[f_j].split('/')[-1][2:5]
            f_ind=int(fid)-1

            img = np.array(Image.open(op.join(data_root,'img', vid,  '{}.png'.format(fid.zfill(5)))), dtype=np.uint8)

            fig, ax = plt.subplots(1)
            ax.imshow(img)
            print('vid {} fid {}'.format(vid, f_ind))

            ant=annot[f_ind]['ant']
            attmat=annot[f_ind]['attmat']

            num_p=0
            num_o=0

            for ind in range(len(ant)):
                if ant[ind]['label'].startswith('Person'):
                    num_p+=1
                elif ant[ind]['label'].startswith('Object'):
                    num_o+=1

            SL=atomic[Demo_atomic[vid][f_j//5]]

            if len(Demo_event[vid])==1:
                BL=event[Demo_event[vid][0]]
            else:
                BL = event[Demo_event[vid][f_j//5]]

            for p_ind in range(num_p):

                pos = ant[p_ind]['pos']

                left1 = int(pos[0])
                top1 = int(pos[1])
                right1 = int(pos[2])
                bottom1 = int(pos[3])

                #----------
                # draw human bbx
                rect = patches.Rectangle((left1, top1), (right1 - left1), (bottom1 - top1), linewidth=1, edgecolor='r',facecolor='none')
                ax.add_patch(rect)

                #----------
                # atomic label

                if vid in vid_list_normal:
                    # ground truth annotation

                    if vid != '55':

                        if p_ind < 2:
                            plt.text(left1, bottom1 + 25, SL,
                                     bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=7)
                        else:
                            plt.text(left1, bottom1 + 25, 'single',
                                     bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=7)
                    else:

                        if num_p==3 and (p_ind == 0 or p_ind == 2):
                            plt.text(left1, bottom1 + 25, SL,
                                     bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=7)
                        elif num_p==3 and p_ind == 1:
                            plt.text(left1, bottom1 + 25, 'single',
                                     bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=7)
                        else:
                            plt.text(left1, bottom1 + 25, SL,
                                     bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=7)


                elif vid in vid_list_slow:
                    # test result

                    if p_ind < 2:

                        if atomic[Test_atomic[vid][f_j // 5]] == SL:
                            plt.text(left1, bottom1 + 25, SL,
                                     bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=7)
                        else:
                            plt.text(left1, bottom1 + 25, SL,
                                     bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=7)
                            plt.text(left1, bottom1 + 55, atomic[Test_atomic[vid][f_j // 5]],
                                     bbox=dict(facecolor='red', edgecolor='none', alpha=0.7, pad=1.0), size=7)
                    else:
                        plt.text(left1, bottom1 + 25, 'single',
                                 bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=7)


            for o_ind in range(num_o):
                pos = ant[num_p+o_ind]['pos']

                left1 = int(pos[0])
                top1 = int(pos[1])
                right1 = int(pos[2])
                bottom1 = int(pos[3])

                #----------
                # draw object bbx
                rect = patches.Rectangle((left1, top1), (right1-left1), (bottom1-top1), linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

            #-------------
            # event label

            if vid in vid_list_normal:
                # ground truth annotation

                plt.text(40, 350, BL, bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=10)
                if vid == '44' and num_p == 3:
                    plt.text(40, 300, 'Non-communicative',
                             bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=10)
                if vid == '55' and num_p == 3:
                    plt.text(40, 300, 'Non-communicative',
                             bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=10)

            elif vid in vid_list_slow:
                    # test result

                if len(Demo_event[vid])==1:
                    test_event=event[Test_event[vid][0]]
                else:
                    test_event = event[Test_event[vid][f_j//5]]

                if test_event==BL:
                    plt.text(40, 350, BL, bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=10)
                else:
                    plt.text(40, 350, BL, bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=10)
                    plt.text(200, 300, test_event, bbox=dict(facecolor='red', edgecolor='none', alpha=0.7, pad=1.0), size=10)


                if vid == '44' and num_p == 3:
                    plt.text(40, 300, 'Non-communicative',
                             bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=10)
                if vid == '55' and num_p == 3:
                    plt.text(40, 300, 'Non-communicative',
                             bbox=dict(facecolor='green', edgecolor='none', alpha=0.7, pad=1.0), size=10)

            #---------------------
            # draw attention arrows
            for i in range(attmat.shape[0]):
                for j in range(attmat.shape[1]):

                    if attmat[i, j] == 1:
                        pos1 = ant[i]['pos']
                        pos2 = ant[j]['pos']

                        center1_x = (int(pos1[0]) + int(pos1[2])) / 2
                        center1_y = (int(pos1[1]) + int(pos1[3])) / 2

                        center2_x = (int(pos2[0]) + int(pos2[2])) / 2
                        center2_y = (int(pos2[1]) + int(pos2[3])) / 2

                        mid_x = int((center1_x + center2_x)/2)
                        mid_y = int((center1_y + center2_y)/2)

                        new_cx=int(0.4*mid_x+0.6*center1_x)
                        new_cy=int(0.4*mid_y+0.6*center1_y)

                        plt.annotate('', xy=(mid_x, mid_y), xytext=(new_cx, new_cy), arrowprops=dict(facecolor='white', edgecolor='white',width=1, headwidth=4, headlength=4.5))


            plt.axis('off')
            height, width, channels = img.shape
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(op.join(save_path, str(f_j).zfill(4)+ '.png'), dpi=1000)


def merge2video():
    vid_list_normal = ['19', '55', '81']
    vid_list_slow = ['94', '27', '70', '5', '87', '58', '23', '44', '72', '74', '145', '64', '1', '202', '13', '14',
                     '242', '196', '17', '80', '126', '198', '82', '93', '18', '120']

    data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'

    #-----------------------
    # start!
    cnt=0

    # title
    for iter in range(50):
        name = str(cnt).zfill(4)
        os.system('cp {} {}'.format(op.join(data_root, 'demo_plus', '1.png'),op.join(data_root, 'demo', name + '.png')))
        cnt+=1

    # atomic intro
    for iter in range(80):
        name = str(cnt).zfill(4)
        os.system('cp {} {}'.format(op.join(data_root, 'demo_plus', '2.png'),op.join(data_root, 'demo', name + '.png')))
        cnt+=1

    # event intro
    for iter in range(80):
        name = str(cnt).zfill(4)
        os.system('cp {} {}'.format(op.join(data_root, 'demo_plus', '3.png'),op.join(data_root, 'demo', name + '.png')))
        cnt+=1

    # dataset annot example front page
    for iter in range(50):
        name = str(cnt).zfill(4)
        os.system('cp {} {}'.format(op.join(data_root, 'demo_plus', '4.png'),op.join(data_root, 'demo', name + '.png')))
        cnt+=1

    # dataset annot example
    for v in range(3):
        vid=vid_list_normal[v]
        img_path=op.join(data_root,'draw_demo_res', vid)
        imgs=[f for f in sorted(os.listdir(img_path))]
        for i in range(len(imgs)):
                name = str(cnt).zfill(4)
                os.system('cp {} {}'.format(op.join(data_root, 'draw_demo_res',vid, imgs[i]), op.join(data_root, 'demo', name + '.png')))
                cnt+=1


    # test result example front page
    for iter in range(80):
        name = str(cnt).zfill(4)
        os.system('cp {} {}'.format(op.join(data_root, 'demo_plus', '5.png'),op.join(data_root, 'demo', name + '.png')))
        cnt+=1


    # test result example
    for v in range(len(vid_list_slow)):
        vid=vid_list_slow[v]
        img_path=op.join(data_root,'draw_demo_res', vid)
        imgs=[f for f in sorted(os.listdir(img_path))]
        for i in range(len(imgs)):

             for j in range(3):
                name = str(cnt).zfill(4)
                os.system('cp {} {}'.format(op.join(data_root, 'draw_demo_res',vid, imgs[i]), op.join(data_root, 'demo', name + '.png')))
                cnt+=1

    # thanks
    for iter in range(50):
        name = str(cnt).zfill(4)
        os.system('cp {} {}'.format(op.join(data_root, 'demo_plus', '6.png'),op.join(data_root, 'demo', name + '.png')))
        cnt+=1




if __name__ == '__main__':


    data_root = '/home/lfan/Dropbox/Projects/ICCV19/DATA/'

    #draw_demo()

    # resize 72 74

    # vid='74'
    #
    # imgs=[f for f in sorted(os.listdir(op.join(data_root, 'draw_demo_res', vid)))]
    #
    #
    # for i in range(len(imgs)):
    #
    #     img=imgs[i]
    #
    #     os.system('convert -resize 640x360! /home/lfan/Dropbox/Projects/ICCV19/DATA/draw_demo_res/{}/{} /home/lfan/Dropbox/Projects/ICCV19/DATA/draw_demo_res/{}/{}'.format(vid, img, vid, img))
    #

    merge2video()






