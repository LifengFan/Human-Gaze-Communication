import sys
sys.path.append("/home/lfan/Dropbox/Projects/ICCV19/RunComm/src/")
import dataset
import dataset.mydataset
import torch
import torch.utils.data
import os
from os import listdir
from dataset import metadata
import random
import collate_fn
import numpy as np
import argparse
import scipy.misc
import cv2
import os.path as op
import pickle

def get_data_demo(args):

    #args.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'
    args.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'
    f = open(op.join(args.data_root, 'event', "clean_event_seq.pkl"), "rb")
    atomic_dict = pickle.load(f)

    test_list=[]

    for m in atomic_dict.keys():
        rec_all=atomic_dict[m]
        for i in range(len(rec_all)):
            rec=rec_all[i]

            ID=rec[0]
            gt_sq=rec[1]
            sq_len=len(gt_sq)

            vid, nid1, nid2, start_id, end_id, event=ID

            for j in range(sq_len):

                s=start_id+j*5
                e=s+4

                gt_atomic=gt_sq[j]

                test_list.append([vid, nid1,nid2,s, e, gt_atomic, event, i, j])


    test_set=dataset.mydataset.mydataset_test_seq(test_list)

    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_test_atomic_in_event, batch_size=args.batch_size,  shuffle=False)

    print('Datset sizes: {} testing.'.format(len(test_loader)))

    return test_set, test_loader



def get_data_test_atomic_in_event(args):

    #args.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'
    args.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'

    f = open(op.join(args.data_root, 'event', "clean_event_seq.pkl"), "rb")
    atomic_dict = pickle.load(f)

    test_list=[]

    for m in atomic_dict.keys():
        rec_all=atomic_dict[m]
        for i in range(len(rec_all)):
            rec=rec_all[i]

            ID=rec[0]
            gt_sq=rec[1]
            sq_len=len(gt_sq)

            vid, nid1, nid2, start_id, end_id, event=ID

            for j in range(sq_len):

                s=start_id+j*5
                e=s+4

                gt_atomic=gt_sq[j]

                test_list.append([vid, nid1,nid2,s, e, gt_atomic, event, i, j])


    test_set=dataset.mydataset.mydataset_test_seq(test_list)

    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_test_atomic_in_event, batch_size=args.batch_size,  shuffle=False)

    print('Datset sizes: {} testing.'.format(len(test_loader)))

    return test_set, test_loader


def get_data_atomic(args):

    #args.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'
    args.data_root='/media/ramdisk/'

    f = open(op.join(args.data_root, 'atomic', "atomic_sample.pkl"), "rb")
    atomic_dict = pickle.load(f)

    # for mode in atomic_dict.keys():
    #     random.shuffle(atomic_dict[mode])

    train_dict={'NA': list(), 'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}
    val_dict = {'NA': list(), 'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}
    test_dict = {'NA': list(), 'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}

    val_seq=[]
    test_seq=[]

    for mode in atomic_dict.keys():

        L=len(atomic_dict[mode])
        train_dict[mode].extend(atomic_dict[mode][: (L//2)])
        val_dict[mode].extend(atomic_dict[mode][(L // 2):(L // 2 + L // 10)])
        test_dict[mode].extend(atomic_dict[mode][(L // 2 + L // 10):])

        random.shuffle(train_dict[mode])
        random.shuffle(val_dict[mode])
        #random.shuffle(test_dict[mode])

    for mode in atomic_dict.keys():
        val_seq.extend(val_dict[mode])
        test_seq.extend(test_dict[mode])

    random.shuffle(val_seq)
    #random.shuffle(test_seq)

    train_set=dataset.mydataset.mydataset_atomic(train_dict,is_train=True)
    val_set=dataset.mydataset.mydataset_atomic(val_dict, is_train=True)
    test_set=dataset.mydataset.mydataset_atomic(test_dict, is_train=True)

    train_loader=torch.utils.data.DataLoader(train_set,collate_fn=collate_fn.collate_fn_atomic, batch_size=args.batch_size, shuffle=False)
    val_loader=torch.utils.data.DataLoader(val_set,collate_fn=collate_fn.collate_fn_atomic, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_atomic, batch_size=args.batch_size,  shuffle=False)

    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader),len(val_loader),len(test_loader)))

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def get_data_pos_bl_lstm(args):

    args.data_root = '/media/ramdisk/'

    files = [os.path.join(args.data_root, 'seqs_20', f) for f in sorted(listdir(os.path.join(args.data_root, 'seqs_20'))) if
             os.path.isfile(os.path.join(args.data_root, 'seqs_20', f))]

    # random.seed(0)
    # idx = np.random.permutation(len(files))
    # idx = idx.tolist()

    L=len(files)
    train_ids=files[:(L//2)]
    val_ids=files[(L//2):(L//2+L//4)]
    test_ids=files[(L//2+L//4):]

    train_seq = {'NA': list(), 'SingleGaze': list(), 'GazeFollow': list(), 'AvertGaze': list(), 'MutualGaze': list(), 'JointAtt': list()}

    train_vids={'NA': list(), 'SingleGaze': list(), 'GazeFollow': list(), 'AvertGaze': list(), 'MutualGaze': list(), 'JointAtt': list()}


    # same_nm_cnt=0
    # total_nm_cnt=0

    for i in range(len(train_ids)):

        video = np.load(train_ids[i], encoding='latin1')
        vid=video[0]['ant'][0]['vid']
        counter = {'SingleGaze': 0, 'GazeFollow': 0, 'AvertGaze': 0, 'MutualGaze': 0, 'JointAtt': 0}

        if len(video[0]['ant'])<=6:
            # diff=0
            for j in range(len(video)):
                # frame = video[j]['ant']
                frame = video[j]

                # if j==0:
                #     nm1=len(frame['ant'])
                # elif len(frame['ant'])==nm1:
                #       continue
                # elif len(frame['ant'])!=nm1:
                #      diff=1


                for k in range(len(frame['ant'])):
                    lab = frame['ant'][k]['BigAtt']
                    if lab != 'NA':
                        counter[lab] += 1

            # if diff==0:
            #     same_nm_cnt+=1
            #     total_nm_cnt+=1
            # else:
            #     total_nm_cnt+=1

            if counter['GazeFollow'] > 0:
                train_seq['GazeFollow'].append(video)
                train_vids['GazeFollow'].append(vid)

            elif counter['AvertGaze'] > 0:
                train_seq['AvertGaze'].append(video)
                train_vids['AvertGaze'].append(vid)

            elif counter['JointAtt'] > 0:
                train_seq['JointAtt'].append(video)
                train_vids['JointAtt'].append(vid)

            elif counter['MutualGaze'] > 0:
                train_seq['MutualGaze'].append(video)
                train_vids['MutualGaze'].append(vid)

            elif counter['SingleGaze'] > 0:
                train_seq['SingleGaze'].append(video)
                train_vids['SingleGaze'].append(vid)

            else:
                train_seq['NA'].append(video)
                train_vids['NA'].append(vid)

    #print('training same node proportion {}/{}'.format(same_nm_cnt,total_nm_cnt))

    random.shuffle(train_seq['NA'])
    random.shuffle(train_seq['SingleGaze'])
    random.shuffle(train_seq['MutualGaze'])
    random.shuffle(train_seq['JointAtt'])
    random.shuffle(train_seq['AvertGaze'])
    random.shuffle(train_seq['GazeFollow'])

    # -----------------------------------------------
    same_nm_cnt=0
    # total_nm_cnt=0

    val_seq = []
    val_vids=[]
    for i in range(len(val_ids)):
        video = np.load(val_ids[i], encoding='latin1')
        vid = video[0]['ant'][0]['vid']

        if len(video[0]['ant']) <= 6:
            val_seq.append(video)
            val_vids.append(vid)

    #     diff=0
    #     for j in range(len(video)):
    #
    #         frame=video[j]
    #
    #         if j==0:
    #             nm1=len(frame['ant'])
    #         elif len(frame['ant'])==nm1:
    #               continue
    #         elif len(frame['ant'])!=nm1:
    #              diff=1
    #
    #     if diff == 0:
    #         same_nm_cnt += 1
    #         total_nm_cnt += 1
    #     else:
    #         total_nm_cnt += 1
    #
    # print('validation same node proportion {}/{}'.format(same_nm_cnt, total_nm_cnt))



    # -----------------------------------------------
    # same_nm_cnt = 0
    # total_nm_cnt = 0

    test_seq = []
    test_vids=[]
    for i in range(len(test_ids)):
        video = np.load(test_ids[i], encoding='latin1')
        vid = video[0]['ant'][0]['vid']

        if len(video[0]['ant']) <= 6:
            test_seq.append(video)
            test_vids.append(vid)

        # diff = 0
        # for j in range(len(video)):
        #
        #     frame = video[j]

    #         if j == 0:
    #             nm1 = len(frame['ant'])
    #         elif len(frame['ant']) == nm1:
    #             continue
    #         elif len(frame['ant']) != nm1:
    #             diff = 1
    #
    #     if diff == 0:
    #         same_nm_cnt += 1
    #         total_nm_cnt += 1
    #     else:
    #         total_nm_cnt += 1
    #
    # print('validation same node proportion {}/{}'.format(same_nm_cnt, total_nm_cnt))


    train_set=dataset.mydataset.mydataset_pos_label_lstm(train_seq,is_train=True)
    val_set=dataset.mydataset.mydataset_pos_label_lstm(val_seq, is_train=False)
    test_set=dataset.mydataset.mydataset_pos_label_lstm(test_seq, is_train=False)

    train_loader=torch.utils.data.DataLoader(train_set,collate_fn=collate_fn.collate_fn_lstm, batch_size=args.batch_size, shuffle=False)
    val_loader=torch.utils.data.DataLoader(val_set,collate_fn=collate_fn.collate_fn_lstm, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_lstm, batch_size=args.batch_size,  shuffle=False)

    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader),len(val_loader),len(test_loader)))

    return train_set, val_set, test_set, train_loader, val_loader, test_loader



def get_data_pos_label_lstm(args):

    args.data_root = '/media/ramdisk/'

    files = [os.path.join(args.data_root, 'seqs_20', f) for f in sorted(listdir(os.path.join(args.data_root, 'seqs_20'))) if
             os.path.isfile(os.path.join(args.data_root, 'seqs_20', f))]

    # random.seed(0)
    # idx = np.random.permutation(len(files))
    # idx = idx.tolist()

    L=len(files)
    train_ids=files[:(L//2)]
    val_ids=files[(L//2):(L//2+L//4)]
    test_ids=files[(L//2+L//4):]

    train_seq = {'NA': list(), 'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(),
                 'share': list()}

    train_vids={'NA': list(), 'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(),'share': list()}
    # same_nm_cnt=0
    # total_nm_cnt=0

    for i in range(len(train_ids)):

        video = np.load(train_ids[i], encoding='latin1')
        vid=video[0]['ant'][0]['vid']
        counter = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}

        if len(video[0]['ant'])<=6:
            # diff=0
            for j in range(len(video)):
                # frame = video[j]['ant']
                frame = video[j]

                # if j==0:
                #     nm1=len(frame['ant'])
                # elif len(frame['ant'])==nm1:
                #       continue
                # elif len(frame['ant'])!=nm1:
                #      diff=1


                for k in range(len(frame['ant'])):
                    lab = frame['ant'][k]['SmallAtt']
                    if lab != 'NA':
                        counter[lab] += 1

            # if diff==0:
            #     same_nm_cnt+=1
            #     total_nm_cnt+=1
            # else:
            #     total_nm_cnt+=1


            if counter['refer'] > 0:
                train_seq['refer'].append(video)
                train_vids['refer'].append(vid)

            elif counter['follow'] > 0:
                train_seq['follow'].append(video)
                train_vids['follow'].append(vid)

            elif counter['avert'] > 0:
                train_seq['avert'].append(video)
                train_vids['avert'].append(vid)

            elif counter['share'] > 0:
                train_seq['share'].append(video)
                train_vids['share'].append(vid)

            elif counter['mutual'] > 0:
                train_seq['mutual'].append(video)
                train_vids['mutual'].append(vid)

            elif counter['single'] > 0:
                train_seq['single'].append(video)
                train_vids['single'].append(vid)

            else:
                train_seq['NA'].append(video)
                train_vids['NA'].append(vid)

    #print('training same node proportion {}/{}'.format(same_nm_cnt,total_nm_cnt))

    random.shuffle(train_seq['NA'])
    random.shuffle(train_seq['single'])
    random.shuffle(train_seq['mutual'])
    random.shuffle(train_seq['avert'])
    random.shuffle(train_seq['refer'])
    random.shuffle(train_seq['follow'])
    random.shuffle(train_seq['share'])

    # -----------------------------------------------
    same_nm_cnt=0
    # total_nm_cnt=0

    val_seq = []
    val_vids=[]
    for i in range(len(val_ids)):
        video = np.load(val_ids[i], encoding='latin1')
        vid = video[0]['ant'][0]['vid']

        if len(video[0]['ant']) <= 6:
            val_seq.append(video)
            val_vids.append(vid)

    #     diff=0
    #     for j in range(len(video)):
    #
    #         frame=video[j]
    #
    #         if j==0:
    #             nm1=len(frame['ant'])
    #         elif len(frame['ant'])==nm1:
    #               continue
    #         elif len(frame['ant'])!=nm1:
    #              diff=1
    #
    #     if diff == 0:
    #         same_nm_cnt += 1
    #         total_nm_cnt += 1
    #     else:
    #         total_nm_cnt += 1
    #
    # print('validation same node proportion {}/{}'.format(same_nm_cnt, total_nm_cnt))



    # -----------------------------------------------
    # same_nm_cnt = 0
    # total_nm_cnt = 0

    test_seq = []
    test_vids=[]
    for i in range(len(test_ids)):
        video = np.load(test_ids[i], encoding='latin1')
        vid = video[0]['ant'][0]['vid']

        if len(video[0]['ant']) <= 6:
            test_seq.append(video)
            test_vids.append(vid)

        # diff = 0
        # for j in range(len(video)):
        #
        #     frame = video[j]

    #         if j == 0:
    #             nm1 = len(frame['ant'])
    #         elif len(frame['ant']) == nm1:
    #             continue
    #         elif len(frame['ant']) != nm1:
    #             diff = 1
    #
    #     if diff == 0:
    #         same_nm_cnt += 1
    #         total_nm_cnt += 1
    #     else:
    #         total_nm_cnt += 1
    #
    # print('validation same node proportion {}/{}'.format(same_nm_cnt, total_nm_cnt))


    train_set=dataset.mydataset.mydataset_pos_label_lstm(train_seq,is_train=True)
    val_set=dataset.mydataset.mydataset_pos_label_lstm(val_seq, is_train=False)
    test_set=dataset.mydataset.mydataset_pos_label_lstm(test_seq, is_train=False)

    train_loader=torch.utils.data.DataLoader(train_set,collate_fn=collate_fn.collate_fn_lstm, batch_size=args.batch_size, shuffle=False)
    val_loader=torch.utils.data.DataLoader(val_set,collate_fn=collate_fn.collate_fn_lstm, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_lstm, batch_size=args.batch_size,  shuffle=False)

    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader),len(val_loader),len(test_loader)))

    return train_set, val_set, test_set, train_loader, val_loader, test_loader



def get_data_pos_label_blc(args):

    args.data_root='/media/ramdisk/'

    files=[f for f in sorted(listdir(op.join(args.data_root,'ant_processed')))]

    L=len(files)

    train_files=files[:L//2]
    val_files=files[L//2:(L//2+L//4)]
    test_files=files[(L//2+L//4):]

    train_seq={'NA':list(),'single':list(),'mutual': list(),'avert': list(),'refer': list(),'follow': list(), 'share': list()}
    #train_seq = {'NA': list(), 'SingleGaze': list(), 'GazeFollow': list(), 'AvertGaze': list(), 'MutualGaze': list(), 'JointAtt': list()}


    for i in range(len(train_files)):
        video = np.load(op.join(args.data_root, 'ant_processed', train_files[i]), encoding='latin1')

        for j in range(len(video)):
            #frame = video[j]['ant']
            frame = video[j]

            if len(frame['ant']) <= 6:
                counter = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
                #counter={'SingleGaze':0,'GazeFollow':0,'AvertGaze':0,'MutualGaze':0,'JointAtt':0}

                for k in range(len(frame['ant'])):

                     lab=frame['ant'][k]['SmallAtt']
                     #lab = frame['ant'][k]['BigAtt']

                     if lab!='NA':
                        counter[lab]+=1


                if counter['refer']>0:
                    train_seq['refer'].append(frame)

                elif counter['follow']>0:
                    train_seq['follow'].append(frame)

                elif counter['avert']>0:
                    train_seq['avert'].append(frame)

                elif counter['share']>0:
                    train_seq['share'].append(frame)

                elif counter['mutual']>0:
                    train_seq['mutual'].append(frame)

                elif counter['single']>0:
                    train_seq['single'].append(frame)
                else:
                    train_seq['NA'].append(frame)

                # if counter['GazeFollow'] > 0:
                #
                #     train_seq['GazeFollow'].append(frame)
                #
                # elif counter['AvertGaze'] > 0:
                #     train_seq['AvertGaze'].append(frame)
                #
                # elif counter['JointAtt'] > 0:
                #     train_seq['JointAtt'].append(frame)
                #
                # elif counter['MutualGaze'] > 0:
                #     train_seq['MutualGaze'].append(frame)
                #
                # elif counter['SingleGaze'] > 0:
                #     train_seq['SingleGaze'].append(frame)
                # else:
                #     train_seq['NA'].append(frame)



    random.shuffle(train_seq['NA'])
    random.shuffle(train_seq['single'])
    random.shuffle(train_seq['mutual'])
    random.shuffle(train_seq['avert'])
    random.shuffle(train_seq['refer'])
    random.shuffle(train_seq['follow'])
    random.shuffle(train_seq['share'])


    # random.shuffle(train_seq['NA'])
    # random.shuffle(train_seq['single'])
    # random.shuffle(train_seq['mutual'])
    # random.shuffle(train_seq['avert'])
    # random.shuffle(train_seq['refer'])
    # random.shuffle(train_seq['follow'])
    # random.shuffle(train_seq['share'])

    # -----------------------------------------------
    val_seq = []
    for i in range(len(val_files)):
        video = np.load(op.join(args.data_root, 'ant_processed_complex', val_files[i]), encoding='latin1')
        for j in range(len(video)):
            #frame = video[j]['ant']
            frame = video[j]

            if len(frame['ant']) <= 6:
                val_seq.append(frame)
    # -----------------------------------------------
    test_seq=[]
    for i in range(len(test_files)):
        video=np.load(op.join(args.data_root,'ant_processed_complex',test_files[i]), encoding='latin1')
        for j in range(len(video)):
            #frame=video[j]['ant']
            frame=video[j]

            if len(frame['ant']) <= 6:
                test_seq.append(frame)

    train_set=dataset.mydataset.mydataset_pos_label_blc(train_seq, is_train=True)
    validate_set=dataset.mydataset.mydataset_pos_label_blc(val_seq, is_train=False)
    test_set=dataset.mydataset.mydataset_pos_label_blc(test_seq, is_train=False)

    train_loader=torch.utils.data.DataLoader(train_set,collate_fn=collate_fn.collate_fn_Nodes2SL, batch_size=args.batch_size, shuffle=False)
    validate_loader=torch.utils.data.DataLoader(validate_set,collate_fn=collate_fn.collate_fn_Nodes2SL, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_Nodes2SL, batch_size=args.batch_size,  shuffle=False)

    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader),len(validate_loader),len(test_loader)))

    return train_set,validate_set, test_set, train_loader, validate_loader, test_loader



def get_data_pos_label(args):

    files=[f for f in sorted(listdir(op.join(args.data_root,'ant_processed_complex')))]

    L=len(files)

    train_files=files[:L//2]
    val_files=files[L//2:(L//2+L//4)]
    test_files=files[(L//2+L//4):]

    # set max_node_num=6
    # -----------------------------------------------
    train_seq=[]
    for i in range(len(train_files)):
        video=np.load(op.join(args.data_root,'ant_processed_complex',train_files[i]), encoding='latin1')

        for j in range(len(video)):
            frame=video[j]['ant']
            if len(frame)<=6:
                train_seq.append(frame)
    # -----------------------------------------------
    val_seq = []
    for i in range(len(val_files)):
        video = np.load(op.join(args.data_root, 'ant_processed_complex', val_files[i]), encoding='latin1')
        for j in range(len(video)):
            frame = video[j]['ant']
            if len(frame) <= 6:
                val_seq.append(frame)
    # -----------------------------------------------
    test_seq=[]
    for i in range(len(test_files)):
        video=np.load(op.join(args.data_root,'ant_processed_complex',test_files[i]), encoding='latin1')
        for j in range(len(video)):
            frame=video[j]['ant']
            if len(frame) <= 6:
                test_seq.append(frame)

    #-------------------------------------------------
    train_set = dataset.mydataset.mydataset_pos_label(train_seq)
    validate_set = dataset.mydataset.mydataset_pos_label(val_seq)
    test_set = dataset.mydataset.mydataset_pos_label(test_seq)

    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn.collate_fn_Nodes2SL,
                                           batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8)
    validate_loader = torch.utils.data.DataLoader(validate_set, collate_fn=collate_fn.collate_fn_Nodes2SL,
                                              batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_Nodes2SL,
                                          batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=8)

    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader), len(validate_loader),
                                                                     len(test_loader)))

    return train_set, validate_set, test_set, train_loader, validate_loader, test_loader


def get_data_attmat(args):

#     files=[f for f in sorted(listdir(op.join(args.data_root,'ant_processed')))]
#
#     L=len(files)
#
#     train_files=files[:L//2]
#     val_files=files[L//2:(L//2+L//4)]
#     test_files=files[(L//2+L//4):]
#
# #-------------------------------
# # train
#     train_seq=[]
#
#     for i in range(len(train_files)):
#
#         video=np.load(op.join(args.data_root,'ant_processed', files[i]))
#
#         for j in range(len(video)):
#             frame=video[j]
#
#             for h in range(len(frame['ant'])):
#                 if frame['ant'][h]['label'].startswith('Person'):
#
#                     for t in range(len(frame['ant'])):
#
#                         if t!=h:
#                             h_pos=frame['ant'][h]['pos']
#                             t_pos=frame['ant'][t]['pos']
#                             vid=frame['ant'][h]['vid']
#                             fid=frame['ant'][h]['frame_ind']
#
#                             att_gt=frame['attmat'][h][t]
#
#                             train_seq.append({ 'vid':vid, 'fid':fid, 'h_pos':h_pos, 't_pos':t_pos, 'att_gt':att_gt})
#
#
#  #---------------------------------
#  # validate
#     val_seq = []
#
#     for i in range(len(val_files)):
#
#         video = np.load(op.join(args.data_root, 'ant_processed', files[i]))
#
#         for j in range(len(video)):
#             frame = video[j]
#
#             for h in range(len(frame['ant'])):
#                 if frame['ant'][h]['label'].startswith('Person'):
#
#                     for t in range(len(frame['ant'])):
#
#                         if t != h:
#                             h_pos = frame['ant'][h]['pos']
#                             t_pos = frame['ant'][t]['pos']
#                             vid = frame['ant'][h]['vid']
#                             fid = frame['ant'][h]['frame_ind']
#
#                             att_gt = frame['attmat'][h][t]
#
#                             val_seq.append({'vid': vid, 'fid': fid, 'h_pos': h_pos, 't_pos': t_pos, 'att_gt': att_gt})
#
# # ---------------------------------
# # test
#
#     test_seq = []
#
#     for i in range(len(test_files)):
#
#         video = np.load(op.join(args.data_root, 'ant_processed', files[i]))
#
#         for j in range(len(video)):
#             frame = video[j]
#
#             for h in range(len(frame['ant'])):
#                 if frame['ant'][h]['label'].startswith('Person'):
#
#                     for t in range(len(frame['ant'])):
#
#                         if t != h:
#                             h_pos = frame['ant'][h]['pos']
#                             t_pos = frame['ant'][t]['pos']
#                             vid = frame['ant'][h]['vid']
#                             fid = frame['ant'][h]['frame_ind']
#
#                             att_gt = frame['attmat'][h][t]
#
#                             test_seq.append({'vid': vid, 'fid': fid, 'h_pos': h_pos, 't_pos': t_pos, 'att_gt': att_gt})
#
#
#     np.save(os.path.join(args.data_root,'attmat','attmat_train_seq.npy'), train_seq)
#     np.save(os.path.join(args.data_root,'attmat','attmat_val_seq.npy'), val_seq)
#     np.save(os.path.join(args.data_root,'attmat','attmat_test_seq.npy'), test_seq)

     train_seq=np.load(op.join(args.data_root,'attmat','attmat_train_seq.npy'), encoding='latin1', allow_pickle = True)
     val_seq=np.load(op.join(args.data_root,'attmat','attmat_val_seq.npy'), encoding='latin1', allow_pickle = True)
     test_seq=np.load(op.join(args.data_root,'attmat','attmat_test_seq.npy'), encoding='latin1', allow_pickle = True)


     train_set = dataset.mydataset.mydataset_attmat(train_seq)
     validate_set = dataset.mydataset.mydataset_attmat(val_seq)
     test_set = dataset.mydataset.mydataset_attmat(test_seq)

     train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn.collate_fn_attmat,
                                           batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8)
     validate_loader = torch.utils.data.DataLoader(validate_set, collate_fn=collate_fn.collate_fn_attmat,
                                              batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8)
     test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_attmat,
                                          batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=8)

     print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader), len(validate_loader),
                                                                     len(test_loader)))

     return train_set, validate_set, test_set, train_loader, validate_loader, test_loader


def save_img_as_array(args):

    for vid in range(238,302):

        imgs=[f for f in sorted(listdir(os.path.join('/home/lfan/Dropbox/RunComm/data/all/img/', str(vid)))) if f.endswith('.png')]

        for j in range(len(imgs)):

            img=scipy.misc.imread(os.path.join('/home/lfan/Dropbox/RunComm/data/all/img/', str(vid), imgs[j]))
            #print(img.shape)
            img=np.asarray(img)


            if not os.path.isdir( os.path.join('/home/lfan/Dropbox/RunComm/data/all/img_np/', str(vid))):
                os.mkdir(os.path.join('/home/lfan/Dropbox/RunComm/data/all/img_np/', str(vid)))

            np.save(os.path.join('/home/lfan/Dropbox/RunComm/data/all/img_np/', str(vid), imgs[j].rstrip('png')+'npy'),img)


def get_data_headpose(args):

    # files=[os.path.join(args.data_root,'headpose',f) for f in sorted(listdir(os.path.join(args.data_root,'headpose'))) if f.startswith('vid')]
    # L=len(files)
    #
    # train_files=files[:L//2]
    # val_files=files[L//2:(L//2+L//4)]
    # test_files=files[(L//2+L//4):]
    #
    # print('Getting headpose data: {} train files, {} val files, {} test files'.format(len(train_files), len(val_files), len(test_files)))
    #
    # train_seq=[]
    # for i in range(len(train_files)):
    #     tmp=np.load(train_files[i])
    #     for j in range(len(tmp)):
    #         rec=tmp[j]['ant']
    #         for k in range(len(rec)):
    #
    #             if 'r_p_y' in rec[k]:
    #
    #                     train_seq.append(rec[k])
    #
    # val_seq=[]
    # for i in range(len(val_files)):
    #     tmp=np.load(val_files[i])
    #     for j in range(len(tmp)):
    #         rec=tmp[j]['ant']
    #         for k in range(len(rec)):
    #             if 'r_p_y' in rec[k]:
    #
    #                     val_seq.append(rec[k])
    #
    # test_seq=[]
    # for i in range(len(test_files)):
    #     tmp=np.load(test_files[i])
    #     for j in range(len(tmp)):
    #         rec=tmp[j]['ant']
    #         for k in range(len(rec)):
    #             if 'r_p_y' in rec[k]:
    #
    #                     test_seq.append(rec[k])
    #
    # np.save(os.path.join(args.data_root,'headpose','headpose_train_seq.npy'), train_seq)
    # np.save(os.path.join(args.data_root,'headpose','headpose_val_seq.npy'), val_seq)
    # np.save(os.path.join(args.data_root,'headpose','headpose_test_seq.npy'), test_seq)

    train_seq=np.load(os.path.join(args.data_root, 'headpose', 'headpose_train_seq.npy'), encoding='latin1')
    val_seq=np.load(os.path.join(args.data_root, 'headpose', 'headpose_val_seq.npy'), encoding='latin1')
    test_seq=np.load(os.path.join(args.data_root, 'headpose', 'headpose_test_seq.npy'), encoding='latin1')

    train_set=dataset.mydataset.mydataset_headpose(train_seq)
    validate_set=dataset.mydataset.mydataset_headpose(val_seq)
    test_set=dataset.mydataset.mydataset_headpose(test_seq)

    train_loader=torch.utils.data.DataLoader(train_set,collate_fn=collate_fn.collate_fn_headpose, batch_size=args.batch_size, shuffle=False,pin_memory=True, num_workers = 8)
    validate_loader=torch.utils.data.DataLoader(validate_set,collate_fn=collate_fn.collate_fn_headpose, batch_size=args.batch_size, shuffle=False,pin_memory=True, num_workers = 8)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_headpose, batch_size=args.batch_size,  shuffle=False,pin_memory=False, num_workers = 8)

    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader),len(validate_loader),len(test_loader)))

    return train_set,validate_set, test_set, train_loader, validate_loader, test_loader


def get_data_resnet_fc(args):

    train_ant_all_files=[os.path.join(args.data_root, 'train', 'ant_processed',f) for f in sorted(listdir(os.path.join(args.data_root,'train','ant_processed'))) if f.endswith('ant_all.npy')]

    val_ant_all_files = [os.path.join(args.data_root, 'validate', 'ant_processed',f) for f in sorted(listdir(os.path.join(args.data_root, 'validate', 'ant_processed'))) if f.endswith('ant_all.npy')]

    test_ant_all_files = [os.path.join(args.data_root,'test', 'ant_processed', f) for f in sorted(listdir(os.path.join(args.data_root, 'test', 'ant_processed'))) if f.endswith('ant_all.npy')]

    train_set=dataset.mydataset.mydataset_resnet_fc(train_ant_all_files)
    validate_set=dataset.mydataset.mydataset_resnet_fc(val_ant_all_files)
    test_set=dataset.mydataset.mydataset_resnet_fc(test_ant_all_files)

    train_loader=torch.utils.data.DataLoader(train_set,collate_fn=collate_fn.collate_fn_resnet_fc, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers = 8)
    validate_loader=torch.utils.data.DataLoader(validate_set,collate_fn=collate_fn.collate_fn_resnet_fc, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers =8)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_resnet_fc, batch_size=args.batch_size,  shuffle=False,pin_memory=True, num_workers = 8)

    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader),len(validate_loader),len(test_loader)))

    return train_set,validate_set, test_set, train_loader, validate_loader, test_loader





def main():

    parser = argparse.ArgumentParser()
    # path settings
    parser.add_argument('--data-root', default='/home/lfan/Dropbox/Projects/ICCV19/DATA/')
    args=parser.parse_args()

    #tmp=np.load('/home/lfan/Dropbox/Projects/ICCV19/RunComm/data/headpose/headpose_train_seq.npy')
    #get_data_headpose(args)
    #save_img_as_array(args)
    #get_data_attmat(args)
    #get_data_pos_label(args)
    #get_data_pos_label_lstm(args)

    get_data_demo(args)

    pass



if __name__=='__main__':
    main()
