import os
import numpy as np
import torch.utils.data
import scipy.misc
from . import metadata
import cv2
from torchvision import transforms
from PIL import Image
import random
from dataset import utils
import  matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.ImageDraw as ImageDraw


def random_crop(frame):

    new_frame=[]

    min_x=np.inf
    max_x=0
    min_y=np.inf
    max_y=0

    for i in range(len(frame)):

        pos = frame[i]['pos']
        x1 = int(pos[0])
        y1 = int(pos[1])
        x2 = int(pos[2])
        y2 = int(pos[3])

        if x1 < min_x:
            min_x = x1
        if y1 < min_y:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2

    min_x= int(random.uniform(0, min_x))
    max_x = int(random.uniform(max_x, 639))
    min_y = int(random.uniform(0, min_y))
    max_y = int(random.uniform(max_y, 359))

    for i in range(len(frame)):

        node=frame[i]

        newnode={}
        newnode['vid'] = node['vid']
        newnode['frame_ind'] = node['frame_ind']
        newnode['label'] = node['label']
        newnode['BigAtt'] = node['BigAtt']
        newnode['SmallAtt'] = node['SmallAtt']
        newnode['focus'] = node['focus']
        pos = node['pos']
        x1 = int(pos[0])
        y1 = int(pos[1])
        x2 = int(pos[2])
        y2 = int(pos[3])

        newnode['pos']=[x1-min_x, y1-min_y, x2-min_x, y2-min_y]

        new_frame.append(newnode)

    crop_range=[min_x, min_y, max_x, max_y]

    return new_frame, crop_range


def h_flip(frame):

    H=360.
    W=640.

    new_frame=[]

    for i in range(len(frame)):

        node=frame[i]

        newnode={}
        newnode['vid'] = node['vid']
        newnode['frame_ind'] = node['frame_ind']
        newnode['label'] = node['label']
        newnode['BigAtt'] = node['BigAtt']
        newnode['SmallAtt'] = node['SmallAtt']
        newnode['focus'] = node['focus']
        pos = node['pos']
        x1 = int(pos[0])
        y1 = int(pos[1])
        x2 = int(pos[2])
        y2 = int(pos[3])

        newnode['pos'] = [(W - 1 - x2), y1, (W - 1 - x1), y2]

        new_frame.append(newnode)

    return new_frame

class mydataset_test_seq(torch.utils.data.Dataset):

    def __init__(self, list):

        self.list = list
        #self.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'
        self.data_root='/media/ramdisk/'
        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.atomic_label = {'single': 0, 'mutual': 1, 'avert': 2, 'refer': 3, 'follow': 4, 'share': 5}
        self.event_label={'SingleGaze':0,'GazeFollow':1,'AvertGaze':2,'MutualGaze':3,'JointAtt':4}
        #'SingleGaze':1, 'GazeFollow':2, 'AvertGaze':3, 'MutualGaze':4, 'JointAtt':5
        #self.atomic_label_4={'single': 0, 'mutual': 1, 'avert': 2, 'refer': 2, 'follow': 2, 'share': 3}
        # self.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
        self.round_cnt = None
        self.h_flip_prob = 0.5
        self.crop_prob = 0.5
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        # self.seq_size = 10

    def __getitem__(self, index):

        rec = self.list[index]

        # [vid, nid1, nid2, s, e, gt_atomic, event, j]

        vid, nid1, nid2, start_fid, end_fid, gt_atomic, ev, sq_ind, pos_in_sq=rec
        event_label=self.event_label[ev]

        head_patch_sq = np.zeros((self.seq_size, 4, 3, 224, 224))
        # face_kp_sq = np.zeros((self.seq_size, 2, 68,3))
        pos_sq = np.zeros((self.seq_size, self.node_num, 6))
        attmat_sq = np.zeros((self.seq_size, self.node_num, self.node_num))
        ID_rec=np.zeros((8))
        #vid, nid1, nid2, start_fid, end_fid, mode = rec
        #atomic_label = self.atomic_label[mode]
        if gt_atomic!='NA':
            atomic_label=self.atomic_label[gt_atomic]
        else:
            atomic_label=-1

        video = np.load(os.path.join(self.data_root, 'ant_processed', 'vid_{}_ant_all.npy'.format(vid)))

        ID_rec[0]=int(vid)
        ID_rec[1]=int(nid1)
        ID_rec[2]=int(nid2)
        ID_rec[3]=int(start_fid)
        ID_rec[4]=int(end_fid)
        ID_rec[5]=int(event_label)
        ID_rec[6]=int(sq_ind)
        ID_rec[7]=int(pos_in_sq)


        for sq_id in range(self.seq_size):

            fid = start_fid + sq_id

            frame = video[fid]['ant']
            attmat = video[fid]['attmat']

            # print('vid {} fid {} mode {} start_id {} total node num {} nid1 {} nid2 {}'.format(vid, fid, atomic_label, start_fid,len(frame), nid1, nid2))

            img_path = os.path.join(self.data_root, 'img_np', vid,
                                    '{}.npy'.format(str(int(frame[0]['frame_ind']) + 1).zfill(5)))
            img_feature = np.load(img_path)

            # human
            for node_i in [0, 1]:
                nid = [nid1, nid2][node_i]

                pos = frame[nid]['pos']
                head = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],
                                  (224, 224))
                head = self.transforms(head).numpy()

                for c in [0, 1, 2]:
                    head[c, :, :] = (head[c, :, :] - 0.5) / 0.5

                pos_vec = np.array(
                    [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                     (float(pos[0]) + float(pos[2])) / (2 * 640.), (float(pos[1]) + float(pos[3])) / (2 * 360.)])

                head_patch_sq[sq_id, node_i, ...] = head
                pos_sq[sq_id, node_i, :] = pos_vec

            # object
            tar_ind1 = np.argwhere(attmat[nid1, :] == 1).flatten()
            tar_ind2 = np.argwhere(attmat[nid2, :] == 1).flatten()

            ind_list = []
            ind_list.append(nid1)
            ind_list.append(nid2)  # [nid1,nid2]

            if len(tar_ind1) > 0 and tar_ind1[0] != nid1 and tar_ind1[0] != nid2:
                ind_list.append(tar_ind1[0])

            if len(tar_ind2) > 0 and tar_ind2[0] != nid1 and tar_ind2[0] != nid2:
                ind_list.append(tar_ind2[0])

            if len(ind_list) == 2 or len(ind_list) == 3:
                # pick object
                for tmp_i in range(len(frame)):
                    if frame[tmp_i]['label'].startswith('Object') and tmp_i not in ind_list:
                        ind_list.append(tmp_i)

            if len(ind_list) > 4:
                ind_list = ind_list[:4]

            if len(ind_list) > 2:
                for obj_i in range(2, len(ind_list)):
                    pos = frame[ind_list[obj_i]]['pos']
                    pos_vec = np.array(
                        [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                         (float(pos[0]) + float(pos[2])) / (2 * 640.), (float(pos[1]) + float(pos[3])) / (2 * 360.)])
                    pos_sq[sq_id, obj_i, :] = pos_vec

                    object = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],
                                        (224, 224))
                    object = self.transforms(object).numpy()

                    for c in [0, 1, 2]:
                        object[c, :, :] = (object[c, :, :] - 0.5) / 0.5

                    head_patch_sq[sq_id, obj_i, ...] = object

            # attmat
            attmat_sq[sq_id, :2, :len(ind_list)] = attmat[ind_list[:2], :][:, ind_list]

            if len(ind_list) < 4:
                # sample object
                # find forbidden zone
                minx = 1000
                miny = 1000
                maxx = -1
                maxy = -1

                for ni in range(len(ind_list)):
                    minx = min(minx, float(frame[ni]['pos'][0]))
                    miny = min(miny, float(frame[ni]['pos'][1]))
                    maxx = max(maxx, float(frame[ni]['pos'][2]))
                    maxy = max(maxy, float(frame[ni]['pos'][3]))

                num = 4 - len(ind_list)
                for obj_i in range(num):

                    if np.random.uniform(0, 1, 1) < (minx / (minx + 640 - maxx)):
                        obj_x = np.random.uniform(0, minx, 1)
                    else:
                        obj_x = np.random.uniform(maxx, 640, 1)

                    if np.random.uniform(0, 1, 1) < (miny / (miny + 360 - maxy)):
                        obj_y = np.random.uniform(0, miny, 1)
                    else:
                        obj_y = np.random.uniform(maxy, 360, 1)

                    # pos=[obj_x-3, obj_y-3, obj_x+3, obj_y+3]
                    # pos = [obj_x[0] - 3, obj_y[0] - 3, obj_x[0] + 3, obj_y[0] + 3]
                    pos = [max(0, obj_x[0] - 3), max(0, obj_y[0] - 3), min(obj_x[0] + 3, 639), min(obj_y[0] + 3, 359)]

                    pos_vec = np.array(
                        [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                         (float(pos[0]) + float(pos[2])) / (2 * 640.), (float(pos[1]) + float(pos[3])) / (2 * 360.)])

                    pos_sq[sq_id, len(ind_list) + obj_i, :] = pos_vec

                    object = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],
                                        (224, 224))
                    object = self.transforms(object).numpy()

                    for c in [0, 1, 2]:
                        object[c, :, :] = (object[c, :, :] - 0.5) / 0.5

                    head_patch_sq[sq_id, len(ind_list) + obj_i, ...] = object

        # return head_patch_sq, pos_sq, attmat_sq, atomic_label,face_kp_sq, ID_rec
        return head_patch_sq, pos_sq, attmat_sq, atomic_label, ID_rec #event_label, pos_in_sq

    def __len__(self):

        return len(self.list)


class mydataset_atomic(torch.utils.data.Dataset):

    def __init__(self, dict, is_train):

        self.dict = dict
        self.is_train = is_train
        #self.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'
        self.data_root='/media/ramdisk/'
        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.atomic_label = {'single': 0, 'mutual': 1, 'avert': 2, 'refer': 3, 'follow': 4, 'share': 5}
        #self.atomic_label_4={'single': 0, 'mutual': 1, 'avert': 2, 'refer': 2, 'follow': 2, 'share': 3}
        # self.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
        self.round_cnt = None
        self.h_flip_prob = 0.5
        self.crop_prob = 0.5
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        # self.seq_size = 10

    def __getitem__(self, index):

        if self.is_train:

            if index % 6 == 0:
                mode = 'single'
            elif index % 6 == 1:
                mode = 'mutual'
            elif index % 6 == 2:
                mode = 'avert'
            elif index % 6 == 3:
                mode = 'refer'
            elif index % 6 == 4:
                mode = 'follow'
            elif index % 6 == 5:
                mode = 'share'

            rec = self.dict[mode][index/ 6 - self.round_cnt[mode] * len(self.dict[mode])]

            head_patch_sq = np.zeros((self.seq_size, 4, 3, 224, 224))
            #face_kp_sq=np.zeros((self.seq_size, 2, 68, 3))
            pos_sq = np.zeros((self.seq_size, self.node_num, 6))
            atomic_label = self.atomic_label[mode]
            attmat_sq = np.zeros((self.seq_size, self.node_num, self.node_num))
            ID_rec=np.zeros((5))

            vid, nid1, nid2, start_fid, end_fid, _ = rec
            video = np.load(os.path.join(self.data_root, 'ant_processed', 'vid_{}_ant_all.npy'.format(vid)))

            ID_rec[0]=int(vid)
            ID_rec[1]=int(nid1)
            ID_rec[2]=int(nid2)
            ID_rec[3]=int(start_fid)
            ID_rec[4]=int(end_fid)

            # #------------------------------------------------
            # # data augmentation
            # if random.uniform(0, 1) < self.h_flip_prob:
            #     h_flip_flag=True
            # else:
            #     h_flip_flag=False
            #     # img_feature = np.fliplr(img_feature)
            #     # frame = h_flip(frame)
            # if random.uniform(0, 1) < self.crop_prob:
            #     crop_flag=True
            # else:
            #     crop_flag=False
                # frame, crop_range = random_crop(frame)
                # img_feature = img_feature[crop_range[1]:crop_range[3], crop_range[0]:crop_range[2], :]
            #--------------------------------------

            for sq_id in range(self.seq_size):

                fid=start_fid+sq_id

                frame = video[fid]['ant']
                attmat = video[fid]['attmat']

                #print('vid {} fid {} mode {} start_id {} total node num {} nid1 {} nid2 {}'.format(vid, fid, atomic_label, start_fid,len(frame), nid1, nid2))

                img_path = os.path.join(self.data_root, 'img_np', vid, '{}.npy'.format(str(int(frame[0]['frame_ind']) + 1).zfill(5)))
                img_feature = np.load(img_path)

                # if h_flip_flag:
                #     img_feature = np.fliplr(img_feature)
                #     frame = h_flip(frame)
                #
                # if crop_flag:
                #
                #     frame, crop_range = random_crop(frame)
                #     img_feature = img_feature[crop_range[1]:crop_range[3], crop_range[0]:crop_range[2], :]

                 # human
                for node_i in [0,1]:
                    nid=[nid1, nid2][node_i]

                    pos = frame[nid]['pos']
                    head = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],(224, 224))
                    head = self.transforms(head).numpy()

                    for c in [0, 1, 2]:
                        head[c, :, :] = (head[c, :, :] - 0.5) / 0.5

                    pos_vec = np.array(
                        [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                         (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])

                    head_patch_sq[sq_id, node_i, ...] = head
                    hid=nid+1
                    # try:
                    #     face_kp=np.load(os.path.join(self.data_root, 'head_features','vid_{}_fid_{}_hid_{}.npy'.format(vid, fid, hid)))
                    #     face_kp_sq[sq_id,node_i,...]=face_kp
                    # except:
                    #     pass

                    pos_sq[sq_id, node_i, :] = pos_vec

                # object
                tar_ind1=np.argwhere(attmat[nid1,:]==1).flatten()
                tar_ind2=np.argwhere(attmat[nid2,:]==1).flatten()

                ind_list=[]
                ind_list.append(nid1)
                ind_list.append(nid2) #[nid1,nid2]

                if len(tar_ind1)>0 and tar_ind1[0]!=nid1 and tar_ind1[0]!=nid2:
                    ind_list.append(tar_ind1[0])

                if len(tar_ind2)>0 and tar_ind2[0]!=nid1 and tar_ind2[0]!=nid2:
                    ind_list.append(tar_ind2[0])


                if len(ind_list)==2 or len(ind_list)==3:
                    # pick object
                    for tmp_i in range(len(frame)):
                        if frame[tmp_i]['label'].startswith('Object') and tmp_i not in ind_list:
                            ind_list.append(tmp_i)

                if len(ind_list)>4:
                    ind_list=ind_list[:4]

                if len(ind_list)>2:
                    for obj_i in range(2, len(ind_list)):
                        pos=frame[ind_list[obj_i]]['pos']

                        #print(pos)

                        pos_vec = np.array(
                            [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                             (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])
                        pos_sq[sq_id, obj_i, :] = pos_vec

                        object = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],
                                      (224, 224))
                        object = self.transforms(object).numpy()

                        for c in [0, 1, 2]:
                            object[c, :, :] = (object[c, :, :] - 0.5) / 0.5

                        head_patch_sq[sq_id, obj_i, ...] = object

                # attmat
                attmat_sq[sq_id,:2, :len(ind_list)]=attmat[ind_list[:2],:][:,ind_list]

                if len(ind_list)<4:
                    # sample object
                    # find forbidden zone
                    minx=1000
                    miny=1000
                    maxx=-1
                    maxy=-1

                    for ni in range(len(ind_list)):
                        minx=min(minx, float(frame[ni]['pos'][0]))
                        miny=min(miny, float(frame[ni]['pos'][1]))
                        maxx=max(maxx, float(frame[ni]['pos'][2]))
                        maxy=max(maxy, float(frame[ni]['pos'][3]))

                    num=4-len(ind_list)
                    for obj_i in range(num):

                        if np.random.uniform(0,1,1)<(minx/(minx+640-maxx)):
                            obj_x=np.random.uniform(0,minx,1)
                        else:
                            obj_x = np.random.uniform(maxx, 640, 1)

                        if np.random.uniform(0, 1, 1) < (miny / (miny + 360 - maxy)):
                            obj_y = np.random.uniform(0, miny, 1)
                        else:
                            obj_y = np.random.uniform(maxy, 360, 1)

                        pos=[max(0,obj_x[0]-3), max(0,obj_y[0]-3), min(obj_x[0]+3,639), min(obj_y[0]+3,359)]

                        pos_vec = np.array(
                            [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                             (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])

                        pos_sq[sq_id, len(ind_list)+obj_i, :] = pos_vec


                        # print(pos)
                        # print(img_feature.shape)
                        # print(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :])

                        object = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],(224, 224))
                        object = self.transforms(object).numpy()

                        for c in [0, 1, 2]:
                            object[c, :, :] = (object[c, :, :] - 0.5) / 0.5

                        head_patch_sq[sq_id, len(ind_list)+obj_i, ...] = object


            if (index / 6 - self.round_cnt[mode] * len(self.dict[mode])) == (len(self.dict[mode]) - 1):
                # if (index / 6 - self.round_cnt[event] * len(self.seq[event])) == (len(self.seq[event]) - 1):
                self.round_cnt[mode] += 1

            #return  head_patch_sq, pos_sq, attmat_sq, atomic_label,face_kp_sq, ID_rec
            return head_patch_sq, pos_sq, attmat_sq, atomic_label #, ID_rec

        else:

            rec = self.dict[index]

            head_patch_sq = np.zeros((self.seq_size, 4, 3, 224, 224))
            #face_kp_sq = np.zeros((self.seq_size, 2, 68,3))
            pos_sq = np.zeros((self.seq_size, self.node_num, 6))
            attmat_sq = np.zeros((self.seq_size, self.node_num, self.node_num))
            #ID_rec=np.zeros((5))


            vid, nid1, nid2, start_fid, end_fid, mode= rec
            atomic_label =self.atomic_label[mode]

            video = np.load(os.path.join(self.data_root, 'ant_processed', 'vid_{}_ant_all.npy'.format(vid)))

            # ID_rec[0]=int(vid)
            # ID_rec[1]=int(nid1)
            # ID_rec[2]=int(nid2)
            # ID_rec[3]=int(start_fid)
            # ID_rec[4]=int(end_fid)
            #------------------------------------------------
            # # data augmentation
            # if random.uniform(0, 1) < self.h_flip_prob:
            #     h_flip_flag=True
            # else:
            #     h_flip_flag=False
            #     # img_feature = np.fliplr(img_feature)
            #     # frame = h_flip(frame)
            # if random.uniform(0, 1) < self.crop_prob:
            #     crop_flag=True
            # else:
            #     crop_flag=False


            for sq_id in range(self.seq_size):

                fid=start_fid+sq_id

                frame = video[fid]['ant']
                attmat = video[fid]['attmat']

                #print('vid {} fid {} mode {} start_id {} total node num {} nid1 {} nid2 {}'.format(vid, fid, atomic_label, start_fid,len(frame), nid1, nid2))

                img_path = os.path.join(self.data_root, 'img_np', vid, '{}.npy'.format(str(int(frame[0]['frame_ind']) + 1).zfill(5)))
                img_feature = np.load(img_path)

                # if h_flip_flag:
                #     img_feature = np.fliplr(img_feature)
                #     frame = h_flip(frame)
                #
                # if crop_flag:
                #
                #     frame, crop_range = random_crop(frame)
                #     img_feature = img_feature[crop_range[1]:crop_range[3], crop_range[0]:crop_range[2], :]

                # if random.uniform(0, 1) < self.h_flip_prob:
                #     ##if True:
                #
                #     # plt.imshow(img_feature)
                #     img_feature = np.fliplr(img_feature)
                #     frame = h_flip(frame)

                # if random.uniform(0, 1) < self.crop_prob:
                #     ## if True:
                #
                #     # plt.imshow(img_feature)
                #
                #     frame, crop_range = random_crop(frame)
                #     img_feature = img_feature[crop_range[1]:crop_range[3], crop_range[0]:crop_range[2], :]

                 # human
                for node_i in [0,1]:
                    nid=[nid1, nid2][node_i]

                    pos = frame[nid]['pos']
                    head = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],(224, 224))
                    head = self.transforms(head).numpy()

                    for c in [0, 1, 2]:
                        head[c, :, :] = (head[c, :, :] - 0.5) / 0.5

                    pos_vec = np.array(
                        [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                         (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])

                    head_patch_sq[sq_id, node_i, ...] = head
                    hid=nid+1
                    # try:
                    #     face_kp=np.load(os.path.join(self.data_root, 'head_features','vid_{}_fid_{}_hid_{}.npy'.format(vid, fid, hid)))
                    #     face_kp_sq[sq_id,node_i,...]=face_kp
                    # except:
                    #     pass
                    pos_sq[sq_id, node_i, :] = pos_vec


                # object
                tar_ind1=np.argwhere(attmat[nid1,:]==1).flatten()
                tar_ind2=np.argwhere(attmat[nid2,:]==1).flatten()

                ind_list=[]
                ind_list.append(nid1)
                ind_list.append(nid2) #[nid1,nid2]

                if len(tar_ind1)>0 and tar_ind1[0]!=nid1 and tar_ind1[0]!=nid2:
                    ind_list.append(tar_ind1[0])

                if len(tar_ind2)>0 and tar_ind2[0]!=nid1 and tar_ind2[0]!=nid2:
                    ind_list.append(tar_ind2[0])


                if len(ind_list)==2 or len(ind_list)==3:
                    # pick object
                    for tmp_i in range(len(frame)):
                        if frame[tmp_i]['label'].startswith('Object') and tmp_i not in ind_list:
                            ind_list.append(tmp_i)

                if len(ind_list)>4:
                    ind_list=ind_list[:4]

                if len(ind_list)>2:
                    for obj_i in range(2, len(ind_list)):
                        pos=frame[ind_list[obj_i]]['pos']
                        pos_vec = np.array(
                            [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                             (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])
                        pos_sq[sq_id, obj_i, :] = pos_vec

                        object = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],(224, 224))
                        object = self.transforms(object).numpy()

                        for c in [0, 1, 2]:
                            object[c, :, :] = (object[c, :, :] - 0.5) / 0.5

                        head_patch_sq[sq_id, obj_i, ...] = object

                # attmat
                attmat_sq[sq_id,:2, :len(ind_list)]=attmat[ind_list[:2],:][:,ind_list]

                if len(ind_list)<4:
                    # sample object
                    # find forbidden zone
                    minx=1000
                    miny=1000
                    maxx=-1
                    maxy=-1

                    for ni in range(len(ind_list)):
                        minx=min(minx, float(frame[ni]['pos'][0]))
                        miny=min(miny, float(frame[ni]['pos'][1]))
                        maxx=max(maxx, float(frame[ni]['pos'][2]))
                        maxy=max(maxy, float(frame[ni]['pos'][3]))

                    num=4-len(ind_list)
                    for obj_i in range(num):

                        if np.random.uniform(0,1,1)<(minx/(minx+640-maxx)):
                            obj_x=np.random.uniform(0,minx,1)
                        else:
                            obj_x = np.random.uniform(maxx, 640, 1)

                        if np.random.uniform(0, 1, 1) < (miny / (miny + 360 - maxy)):
                            obj_y = np.random.uniform(0, miny, 1)
                        else:
                            obj_y = np.random.uniform(maxy, 360, 1)

                        #pos=[obj_x-3, obj_y-3, obj_x+3, obj_y+3]
                        #pos = [obj_x[0] - 3, obj_y[0] - 3, obj_x[0] + 3, obj_y[0] + 3]
                        pos = [max(0, obj_x[0] - 3), max(0, obj_y[0] - 3), min(obj_x[0] + 3, 639),min(obj_y[0] + 3, 359)]

                        pos_vec = np.array([float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                             (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])

                        pos_sq[sq_id, len(ind_list)+obj_i, :] = pos_vec

                        object = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],(224, 224))
                        object = self.transforms(object).numpy()

                        for c in [0, 1, 2]:
                            object[c, :, :] = (object[c, :, :] - 0.5) / 0.5

                        head_patch_sq[sq_id, len(ind_list)+obj_i, ...] = object

            #return head_patch_sq, pos_sq, attmat_sq, atomic_label,face_kp_sq, ID_rec
            return head_patch_sq, pos_sq, attmat_sq, atomic_label


    def __len__(self):
        if self.is_train:
            max_len = 0
            for event in ['single', 'mutual', 'avert', 'refer', 'follow', 'share']:
                max_len = max(len(self.dict[event]), max_len)

            return max_len * 6

        else:
            return len(self.dict)


class mydataset_pos_label_lstm(torch.utils.data.Dataset):

      def __init__(self, seq, is_train):

          self.seq=seq
          self.is_train=is_train
          self.paths = utils.Paths()
          self.transforms = transforms.Compose([transforms.ToTensor()])

          #self.SL = {'NA': 0, 'single': 1, 'mutual': 2, 'avert': 3, 'refer': 4, 'follow': 5, 'share': 6}
          self.BL = {'NA': 0, 'SingleGaze': 1, 'GazeFollow': 2, 'AvertGaze': 3, 'MutualGaze': 4, 'JointAtt': 5}

          # self.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
          self.round_cnt = None
          self.h_flip_prob = 0.5
          self.crop_prob = 0.5
          self.seq_size = 20
          #self.seq_size = 10


      def __getitem__(self, index):

          if self.is_train:

              # if index % 6 == 0:
              #     event = 'single'
              # elif index % 6 == 1:
              #     event = 'mutual'
              # elif index % 6 == 2:
              #     event = 'avert'
              # elif index % 6 == 3:
              #     event = 'refer'
              # elif index % 6 == 4:
              #     event = 'follow'
              # elif index % 6 == 5:
              #     event = 'share'

              if index % 5 == 0:
                  event = 'SingleGaze'
              elif index % 5 == 1:
                  event = 'GazeFollow'
              elif index % 5 == 2:
                  event = 'AvertGaze'
              elif index % 5 == 3:
                  event = 'MutualGaze'
              elif index % 5 == 4:
                  event = 'JointAtt'

              #video = self.seq[event][index / 6 - self.round_cnt[event] * len(self.seq[event])]
              video = self.seq[event][index/5-self.round_cnt[event]*len(self.seq[event])]

              patches_sq = np.zeros((self.seq_size, 6, 3, 224, 224))
              poses_sq = np.zeros((self.seq_size, 6, 6))
              s_labels_sq = np.zeros((self.seq_size,6))
              att_mat_sq = np.zeros((self.seq_size,6,6))
              node_num_sq=np.zeros((self.seq_size))

              for sq_id in range(self.seq_size):
                  frame = video[sq_id]['ant']
                  attmat = video[sq_id]['attmat']
                  node_num = len(frame)

                  vid = frame[0]['vid']
                  fid = frame[0]['frame_ind']

                  img_path = os.path.join('/media/ramdisk/', 'img_np', vid,'{}.npy'.format(str(int(fid) + 1).zfill(5)))
                  img_feature = np.load(img_path)

                  if random.uniform(0, 1) < self.h_flip_prob:
                      ##if True:

                      # plt.imshow(img_feature)
                      img_feature = np.fliplr(img_feature)
                      frame = h_flip(frame)

                  if random.uniform(0, 1) < self.crop_prob:
                      ## if True:

                      # plt.imshow(img_feature)

                      frame, crop_range = random_crop(frame)
                      img_feature = img_feature[crop_range[1]:crop_range[3], crop_range[0]:crop_range[2], :]

                  for i in range(node_num):
                      pos = frame[i]['pos']
                      #s_label = self.SL[frame[i]['SmallAtt']]
                      s_label = self.BL[frame[i]['BigAtt']]

                      head_patch = cv2.resize(
                          img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],
                          (224, 224))

                      # todo: check div 255--yes!
                      head_patch = self.transforms(head_patch).numpy()

                      for c in [0, 1, 2]:
                          head_patch[c, :, :] = (head_patch[c, :, :] - 0.5) / 0.5

                      pos = np.array(
                          [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                           (float(pos[0]) + float(pos[2])) / (2 * 640.),
                           (float(pos[1]) + float(pos[3])) / (2 * 360.)])

                      patches_sq[sq_id, i, ...] = head_patch
                      poses_sq[sq_id, i, :] = pos
                      s_labels_sq[sq_id, i] = s_label
                  att_mat_sq[sq_id,:node_num,:node_num]=attmat
                  node_num_sq[sq_id]=node_num

              if (index / 5 - self.round_cnt[event] * len(self.seq[event])) == (len(self.seq[event]) - 1):
              #if (index / 6 - self.round_cnt[event] * len(self.seq[event])) == (len(self.seq[event]) - 1):

                  self.round_cnt[event] += 1

              return patches_sq, poses_sq, s_labels_sq, node_num_sq, att_mat_sq

          else:

              video = self.seq[index]

              patches_sq = np.zeros((self.seq_size, 6, 3, 224, 224))
              poses_sq = np.zeros((self.seq_size, 6, 6))
              s_labels_sq = np.zeros((self.seq_size,6))
              att_mat_sq = np.zeros((self.seq_size,6,6))
              node_num_sq=np.zeros((self.seq_size))

              for sq_id in range(self.seq_size):

                  frame = video[sq_id]['ant']
                  attmat = video[sq_id]['attmat']
                  node_num = len(frame)

                  vid = frame[0]['vid']
                  fid = frame[0]['frame_ind']

                  img_path = os.path.join('/media/ramdisk/', 'img_np', vid,
                                          '{}.npy'.format(str(int(fid) + 1).zfill(5)))
                  img_feature = np.load(img_path)

                  for i in range(node_num):
                      pos = frame[i]['pos']
                      #s_label = self.SL[frame[i]['SmallAtt']]
                      s_label = self.BL[frame[i]['BigAtt']]
                      head_patch = cv2.resize(
                          img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],
                          (224, 224))

                      # todo: check div 255
                      head_patch = self.transforms(head_patch).numpy()

                      for c in [0, 1, 2]:
                          head_patch[c, :, :] = (head_patch[c, :, :] - 0.5) / 0.5

                      pos = np.array(
                          [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                           (float(pos[0]) + float(pos[2])) / (2 * 640.),
                           (float(pos[1]) + float(pos[3])) / (2 * 360.)])

                      patches_sq[sq_id,i, ...] = head_patch
                      poses_sq[sq_id,i, :] = pos
                      s_labels_sq[sq_id,i] = s_label
                  att_mat_sq[sq_id, :node_num, :node_num] = attmat
                  node_num_sq[sq_id] = node_num

              return patches_sq, poses_sq, s_labels_sq, node_num_sq, att_mat_sq

      def __len__(self):

          if self.is_train:
              max_len = 0
              #for event in ['single', 'mutual', 'avert', 'refer', 'follow', 'share']:
              for event in ['SingleGaze', 'GazeFollow', 'AvertGaze', 'MutualGaze', 'JointAtt']:
                  max_len = max(len(self.seq[event]), max_len)

              #return max_len * 6
              return max_len * 5

          else:

              return len(self.seq)



class mydataset_pos_label_blc(torch.utils.data.Dataset):
    # generating class balanced data

    def __init__(self, seq, is_train):

        self.seq = seq
        self.is_train=is_train
        self.paths = utils.Paths()
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.SL = {'NA': 0, 'single': 1, 'mutual': 2, 'avert': 3, 'refer': 4, 'follow': 5, 'share': 6}
        self.BL = {'NA':0,'SingleGaze':1,'GazeFollow':2,'AvertGaze':3,'MutualGaze':4,'JointAtt':5}
        #self.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
        self.round_cnt=None
        self.h_flip_prob=0.5
        self.crop_prob=0.5

        #print('init round cnt! ', self.round_cnt)

    def __getitem__(self, index):

        #print(index)

        if self.is_train:

            if index % 6 == 0:
                event = 'single'
            elif index % 6 == 1:
                event = 'mutual'
            elif index % 6 == 2:
                event = 'avert'
            elif index % 6 == 3:
                event = 'refer'
            elif index % 6 == 4:
                event = 'follow'
            elif index % 6 == 5:
                event='share'

            frame=self.seq[event][index/6-self.round_cnt[event]*len(self.seq[event])]['ant']
            attmat=self.seq[event][index/6-self.round_cnt[event]*len(self.seq[event])]['attmat']
            node_num = len(frame)

            patches = np.zeros((node_num, 3, 224, 224))
            poses = np.zeros((node_num, 6))
            s_labels = np.zeros(node_num)

            vid = frame[0]['vid']
            fid = frame[0]['frame_ind']

            img_path = os.path.join(self.paths.data_root, 'img_np', vid, '{}.npy'.format(str(int(fid) + 1).zfill(5)))
            img_feature = np.load(img_path)

            # data augmentation--horizontal flip

            if random.uniform(0,1)<self.h_flip_prob:
            ##if True:

                #plt.imshow(img_feature)
                img_feature=np.fliplr(img_feature)
                frame=h_flip(frame)

                #im = Image.fromarray(img_feature)
                #draw = ImageDraw.Draw(im)
                #
                # for i in range(len(frame)):
                #
                #     pos=frame[i]['pos']
                #
                #     left1 = int(pos[0])
                #     top1 = int(pos[1])
                #     right1 = int(pos[2])
                #     bottom1 = int(pos[3])
                #     draw.line([(left1, top1), (left1, bottom1), (right1, bottom1), (right1, top1), (left1, top1)], width=4,
                #           fill='red')
                #
                # im.show()

            if random.uniform(0,1)<self.crop_prob:

            ## if True:

                #plt.imshow(img_feature)

                frame, crop_range=random_crop(frame)
                img_feature = img_feature[crop_range[1]:crop_range[3], crop_range[0]:crop_range[2], :]

                #im = Image.fromarray(img_feature)
                #draw = ImageDraw.Draw(im)

                # for i in range(len(frame)):
                #
                #     pos=frame[i]['pos']
                #
                #     left1 = int(pos[0])
                #     top1 = int(pos[1])
                #     right1 = int(pos[2])
                #     bottom1 = int(pos[3])
                #     draw.line([(left1, top1), (left1, bottom1), (right1, bottom1), (right1, top1), (left1, top1)], width=4,
                #           fill='red')
                #
                # im.show()


            for i in range(node_num):
                pos = frame[i]['pos']
                #s_label = self.SL[frame[i]['SmallAtt']]
                #todo: use big label here!
                s_label = self.BL[frame[i]['BigAtt']]

                head_patch = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],
                                        (224, 224))

                # todo: check div 255--yes!
                head_patch = self.transforms(head_patch).numpy()

                for c in [0, 1, 2]:
                    head_patch[c, :, :] = (head_patch[c, :, :] - 0.5) / 0.5

                pos = np.array([float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                                (float(pos[0]) + float(pos[2])) / (2 * 640.),
                                (float(pos[1]) + float(pos[3])) / (2 * 360.)])

                patches[i, ...] = head_patch
                poses[i, :] = pos
                s_labels[i] = s_label

            if (index / 6 - self.round_cnt[event] * len(self.seq[event])) == (len(self.seq[event]) - 1):

                self.round_cnt[event] += 1

            return patches, poses, s_labels, node_num, attmat


        else:

            frame = self.seq[index]['ant']
            attmat= self.seq[index]['attmat']
            node_num = len(frame)

            patches = np.zeros((node_num, 3, 224, 224))
            poses = np.zeros((node_num, 6))
            s_labels = np.zeros(node_num)

            vid = frame[0]['vid']
            fid = frame[0]['frame_ind']

            img_path = os.path.join(self.paths.data_root, 'img_np', vid, '{}.npy'.format(str(int(fid) + 1).zfill(5)))
            img_feature = np.load(img_path)

            for i in range(node_num):
                pos = frame[i]['pos']
                #s_label = self.SL[frame[i]['SmallAtt']]
                s_label = self.BL[frame[i]['BigAtt']]
                head_patch = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :],
                                        (224, 224))

                # todo: check div 255
                head_patch = self.transforms(head_patch).numpy()

                for c in [0, 1, 2]:
                    head_patch[c, :, :] = (head_patch[c, :, :] - 0.5) / 0.5

                pos = np.array([float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                                (float(pos[0]) + float(pos[2])) / (2 * 640.),
                                (float(pos[1]) + float(pos[3])) / (2 * 360.)])

                patches[i, ...] = head_patch
                poses[i, :] = pos
                s_labels[i] = s_label

            return patches, poses, s_labels, node_num, attmat

    def __len__(self):

        if self.is_train:
            max_len = 0
            for event in ['single', 'mutual', 'avert', 'refer', 'follow', 'share']:
                max_len = max(len(self.seq[event]), max_len)

            return max_len*6

        else:

            return len(self.seq)

class mydataset_pos_label(torch.utils.data.Dataset):
    def __init__(self,seq):

        self.seq=seq
        self.paths = utils.Paths()
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.SL={'NA':0,'single':1,'mutual':2,'avert':3,'refer':4,'follow':5,'share':6}

    def __getitem__(self, index):

        frame = self.seq[index]
        node_num=len(frame)

        patches=np.zeros((node_num,3,224,224))
        poses=np.zeros((node_num,6))
        s_labels=np.zeros(node_num)


        vid = frame[0]['vid']
        fid = frame[0]['frame_ind']

        img_path = os.path.join(self.paths.data_root, 'img_np', vid, '{}.npy'.format(str(int(fid) + 1).zfill(5)))
        img_feature = np.load(img_path)

        for i in range(node_num):
            pos=frame[i]['pos']
            s_label=self.SL[frame[i]['SmallAtt']]
            head_patch = cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :], (224, 224))

            # todo: check div 255
            head_patch = self.transforms(head_patch).numpy()

            for c in [0, 1, 2]:
                head_patch[c, :, :] = (head_patch[c, :, :] - 0.5) / 0.5


            pos=np.array([float(pos[0])/640.,  float(pos[1])/360.,  float(pos[2])/640.,  float(pos[3])/360.,
                          (float(pos[0])+float(pos[2]))/(2*640.),   (float(pos[1])+float(pos[3]))/(2*360.)])

            patches[i,...]=head_patch
            poses[i,:]=pos
            s_labels[i]=s_label

        return patches, poses, s_labels, node_num


    def __len__(self):

        return len(self.seq)


class mydataset_attmat(torch.utils.data.Dataset):

    def __init__(self,seq):

        self.seq=seq
        self.paths=utils.Paths()
        self.transforms = transforms.Compose([transforms.ToTensor()])
        # self.mean=[0.5, 0.5, 0.5]
        # self.std=[0.5, 0.5, 0.5]
        # self.H=360.
        # self.W=640.

    def __getitem__(self, index):

        rec = self.seq[index]

        vid=rec['vid']
        fid=rec['fid']
        h_pos=rec['h_pos']
        t_pos=rec['t_pos']
        att_gt=rec['att_gt']

        img_path =  os.path.join(self.paths.data_root, 'img_np', vid, '{}.npy'.format(str(int(fid)+1).zfill(5)))
        img_feature=np.load(img_path)

        head_patch=cv2.resize(img_feature[int(h_pos[1]):(int(h_pos[3]) + 1), int(h_pos[0]):(int(h_pos[2]) + 1), :], (224, 224))

        head_patch=self.transforms(head_patch).numpy()

        for c in [0,1,2]:
            head_patch[c,:,:]=(head_patch[c,:,:]-0.5)/0.5

        pos=np.array([float(h_pos[0])/640.,  float(h_pos[1])/360.,  float(h_pos[2])/640.,  float(h_pos[3])/360.,
                          (float(h_pos[0])+float(h_pos[2]))/(2*640.),   (float(h_pos[1])+float(h_pos[3]))/(2*360.),
                          float(t_pos[0])/640., float(t_pos[1])/360., float(t_pos[2])/640., float(t_pos[3])/360.,
                          (float(t_pos[0]) + float(t_pos[2])) / (2 * 640.), (float(t_pos[1]) + float(t_pos[3])) / (2 * 360.)])


        return head_patch, pos, att_gt


    def __len__(self):

        return len(self.seq)



class mydataset_headpose(torch.utils.data.Dataset):

    def __init__(self,seq):

        self.seq=seq
        self.paths=utils.Paths()
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.mean=[0.5, 0.5, 0.5]
        self.std=[0.5, 0.5, 0.5]


    def __getitem__(self, index):

        #index=91
        rec = self.seq[index]

        vid=rec['vid']
        frame_id=rec['frame_ind']

        img_path =  os.path.join(self.paths.data_root, 'images', vid, '{}.png'.format(str(int(frame_id)+1).zfill(5)))
        #img_feature = Image.open(img_path)
        img_feature=scipy.misc.imread(img_path, mode='RGB')

        pos=rec['pos']
        print(img_feature.shape)
        print(index, vid, frame_id, pos)

        head_patch=cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :], (224, 224))
        #head_patch=img_feature.crop((int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3]))).resize((224,224))
        #head_patch.show()

        head_patch=self.transforms(head_patch).numpy()

        for c in [0,1,2]:
            head_patch[c,:,:]=(head_patch[c,:,:]-0.5)/0.5


        #todo: rpy range to [-1, 1]
        roll=rec['r_p_y'][0]/25.
        pitch=rec['r_p_y'][1]/45.
        yaw=rec['r_p_y'][2]/100.

        rpy=[roll, pitch, yaw]

        return head_patch, rpy


    def __len__(self):

        return len(self.seq)



class mydataset_resnet_fc(torch.utils.data.Dataset):

    def __init__(self, ant_file_list):

        self.ant_list=list()
        self.paths=utils.Paths()

        for ind in range(len(ant_file_list)):
            self.ant_list.extend(np.load(ant_file_list[ind]))

        #todo: use partial data
        random.shuffle(self.ant_list)
        self.ant_list=self.ant_list[:len(self.ant_list)//3]

        # self.transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=metadata.train_mean_value,
        #                      std=metadata.train_std_value)])

        # self.transform=transforms.Compose([transforms.ToTensor()])
        # todo: data augmentation
        self.transforms = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index):

        rec=self.ant_list[index]
        vid=rec[0]['vid']
        frame_id=rec[0]['frame_ind']

        #img_feature = scipy.misc.imread(os.path.join(self.paths.data_root, 'all', 'img', vid, '{}.png'
                                                               #.format(str(int(frame_id) + 1).zfill(5))),mode='RGB')
        img_feature=Image.open(os.path.join(self.paths.data_root, 'all', 'img', vid, '{}.png'.format(str(int(frame_id) + 1).zfill(5))))

        #scipy.misc.imshow(img_feature)

        node_feature=list()
        gt_label=list()


        for node_id in range(len(rec)):
            node=rec[node_id]
            pos=node['pos']

            #loc_map=np.zeros((img_feature.shape[0],img_feature.shape[1],1))
            #loc_map[int(pos[1]):(int(pos[3])+1), int(pos[0]):(int(pos[2])+1),:]=1
            #node_patch=cv2.resize(img_feature[int(pos[1]):(int(pos[3]) + 1), int(pos[0]):(int(pos[2]) + 1), :], (224,224))
            #print(pos)
            node_patch = img_feature.crop((int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3]))).resize((224,224), resample=Image.BICUBIC)
            #node_patch.show()
            #scipy.misc.imshow(node_patch)

            node_feature.append(self.transforms(node_patch).numpy())

            # todo: img should be correctly normalized

            gt_label.append(metadata.big_class[node['BigAtt']])


        return node_feature, gt_label


    def __len__(self):

        return len(self.ant_list)



def main():

    tmp=np.array([1,2,3])
    print(tmp.argwhere(2))

    pass

if __name__=='__main__':

    main()



