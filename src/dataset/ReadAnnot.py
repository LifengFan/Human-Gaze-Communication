"""
Created on Oct 8, 2018

Author: Lifeng Fan

Description:

"""

import os
from os import listdir
from os.path import isfile
import shutil
import numpy as np
import utils


def ReadAnnot(paths, mode):
    ant_files = [f for f in sorted(listdir(os.path.join(paths.data_root, mode, 'ant'))) if
                 isfile(os.path.join(paths.data_root, mode, 'ant', f))]

    if os.path.isdir(os.path.join(paths.data_root,mode,'img_list')):

        os.system('rm -rf '+os.path.join(paths.data_root,mode,'img_list'))
        os.system('mkdir '+os.path.join(paths.data_root,mode,'img_list'))


    for file_ind in range(len(ant_files)):

        vid=ant_files[file_ind][7:-4]
        print('video id : {}'.format(vid))

        # if os.path.isfile(os.path.join(paths.data_root,mode,'ant_processed','vid_{}_classes.npy'.format(vid))):
        #
        #     continue

        ant_f = ant_files[file_ind]
        with open(os.path.join(paths.data_root, mode, 'ant', ant_f), 'r') as to_read:
            lines = [line.split(" ") for line in to_read.readlines()]

        frame_num = int(lines[-1][5]) + 1

        ant_all=list()
        det_box_all=list()
        det_class_all=list()
        adj_mat_all=list()

        small_attr_all=list()
        big_attr_all=list()

        for frame_ind in range(frame_num):

            ant_tmp=list()
            record_tmp=list()
            person_tmp = list()
            obj_tmp = list()

            det_box_tmp = np.empty((0, 4))
            det_class_tmp = list()

            small_attr_tmp=list()
            big_attr_tmp=list()


            # get record_tmp for this frame
            for line_ind in range(len(lines)):
                if int(lines[line_ind][5])==frame_ind and lines[line_ind][6]=='0' and lines[line_ind][7]=='0':
                   record_tmp.append(lines[line_ind])

            # get person_tmp and obj_tmp for this frame
            for rec_ind in range(len(record_tmp)):

                pos=record_tmp[rec_ind][1:5]
                label=record_tmp[rec_ind][9].strip()

                if label[0:6]=="Person":
                    person_tmp.append(
                        {'pos': pos, 'label': label, 'frame_ind': record_tmp[rec_ind][5], 'focus': record_tmp[rec_ind][12].strip(),
                         'BigAtt': record_tmp[rec_ind][10], 'SmallAtt': record_tmp[rec_ind][11],'vid':vid})

                elif label[0:6]=="Object":
                    obj_tmp.append({'pos': pos, 'label': label, 'frame_ind': record_tmp[rec_ind][5],'vid':vid})

            # get det_box_tmp and det_class_tmp for this frame, starts with person, then object
            for per_ind in range(len(person_tmp)):
                    det_box_tmp=np.vstack((det_box_tmp,person_tmp[per_ind]['pos']))
                    det_class_tmp.append(1)
                    ant_tmp.append(person_tmp[per_ind])

            for obj_ind in range(len(obj_tmp)):
                    det_box_tmp=np.vstack((det_box_tmp,obj_tmp[obj_ind]['pos']))
                    det_class_tmp.append(2)
                    ant_tmp.append(obj_tmp[obj_ind])

            # get the adj_mat_tmp for this frame
            adj_mat_tmp = np.zeros((len(record_tmp), len(record_tmp)))

            for per_ind in range(len(person_tmp)):

                focus=person_tmp[per_ind]['focus']

                if focus[0]=='P':
                    for per_ind_2 in range(len(person_tmp)):
                        if person_tmp[per_ind_2]['label']=='Person'+focus[1]:
                            adj_mat_tmp[per_ind,per_ind_2]=1
                elif focus[0]=='O':
                    for obj_ind_2 in range(len(obj_tmp)):
                        if obj_tmp[obj_ind_2]['label']=='Object'+focus[1]:
                            adj_mat_tmp[per_ind,len(person_tmp)+obj_ind_2]=1


            # get the small attribute and big attribute for humans in this frame
            for per_ind in range(len(person_tmp)):

                if person_tmp[per_ind]['SmallAtt']=='NA':
                    small_attr_tmp.append(0)
                elif person_tmp[per_ind]['SmallAtt']=='single':
                    small_attr_tmp.append(1)
                elif person_tmp[per_ind]['SmallAtt']=='mutual':
                    small_attr_tmp.append(2)
                elif person_tmp[per_ind]['SmallAtt']=='avert':
                    small_attr_tmp.append(3)
                elif person_tmp[per_ind]['SmallAtt']=='refer':
                    small_attr_tmp.append(4)
                elif person_tmp[per_ind]['SmallAtt']=='follow':
                    small_attr_tmp.append(5)
                elif person_tmp[per_ind]['SmallAtt']=='share':
                    small_attr_tmp.append(6)


                if person_tmp[per_ind]['BigAtt']=='NA':
                    big_attr_tmp.append(0)
                elif person_tmp[per_ind]['BigAtt']=='SingleGaze':
                    big_attr_tmp.append(1)
                elif person_tmp[per_ind]['BigAtt']=='GazeFollow':
                    big_attr_tmp.append(2)
                elif person_tmp[per_ind]['BigAtt']=='AvertGaze':
                    big_attr_tmp.append(3)
                elif person_tmp[per_ind]['BigAtt']=='MutualGaze':
                    big_attr_tmp.append(4)
                elif person_tmp[per_ind]['BigAtt']=='JointAtt':
                    big_attr_tmp.append(5)


            for obj_ind in range(len(obj_tmp)):

                small_attr_tmp.append(0)
                big_attr_tmp.append(0)


            det_box_all.append(det_box_tmp)
            det_class_all.append(det_class_tmp)
            adj_mat_all.append(adj_mat_tmp)

            #todo: here no longer add empty records for ant_all, cuz already encode vid and imgid info inside
            if len(ant_tmp)>0:
                ant_all.append(ant_tmp)

            small_attr_all.append(small_attr_tmp)
            big_attr_all.append(big_attr_tmp)

            # should generate img list here to avoid empty frame

            if len(record_tmp)>0:

                with open(os.path.join(paths.data_root,mode,'img_list',vid+'.txt'),'a') as towrite:

                    towrite.write(os.path.join(paths.data_root,mode,'img',vid,str((frame_ind+1)).zfill(5)+'.png'))
                    towrite.write('\n')

        #save data for this frame
        np.save(os.path.join(paths.data_root,mode,'ant_processed','vid_{}_classes'.format(vid)),det_class_all)
        np.save(os.path.join(paths.data_root, mode, 'ant_processed', 'vid_{}_boxes'.format(vid)),det_box_all)
        np.save(os.path.join(paths.data_root, mode, 'ant_processed', 'vid_{}_adjmat'.format(vid)),adj_mat_all)
        np.save(os.path.join(paths.data_root, mode, 'ant_processed', 'vid_{}_ant_all'.format(vid)),ant_all)

        np.save(os.path.join(paths.data_root,mode,'ant_processed','vid_{}_small_att'.format(vid)),small_attr_all)
        np.save(os.path.join(paths.data_root, mode, 'ant_processed', 'vid_{}_big_att'.format(vid)), big_attr_all)


def main(paths,mode):

    ReadAnnot(paths,mode)
    pass


if __name__ == '__main__':

    paths=utils.Paths()
    #mode='train'
    #mode='validate'
    #mode='test'
    for mode in ['train','validate','test']:

        main(paths,mode)

    pass

