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


# be careful with frame id!

def ReadAnnot(paths):

    ant_files = [f for f in sorted(listdir(os.path.join(paths.data_root, 'annotation_cleaned'))) if
                 isfile(os.path.join(paths.data_root, 'annotation_cleaned', f))]

    # if os.path.isdir(os.path.join(paths.data_root,mode,'img_list')):
    #
    #     os.system('rm -rf '+os.path.join(paths.data_root,mode,'img_list'))
    #     os.system('mkdir '+os.path.join(paths.data_root,mode,'img_list'))


    for file_ind in range(len(ant_files)):

        vid=ant_files[file_ind][7:-4]
        print('video id : {}'.format(vid))

        # if os.path.isfile(os.path.join(paths.data_root,mode,'ant_processed','vid_{}_classes.npy'.format(vid))):
        #
        #     continue

        ant_f = ant_files[file_ind]
        with open(os.path.join(paths.data_root, 'annotation_cleaned', ant_f), 'r') as to_read:
            lines = [line.split(" ") for line in to_read.readlines()]

        frame_num = int(lines[-1][5]) + 1

        ant_all=list()


        for frame_ind in range(frame_num):

            ant_tmp=list()
            record_tmp=list()
            person_tmp = list()
            obj_tmp = list()

            # get record_tmp for this frame
            for line_ind in range(len(lines)):

                assert lines[line_ind][6]=='0'
                assert lines[line_ind][7]=='0'

                if int(lines[line_ind][5])==int(frame_ind):
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
                    obj_tmp.append({'pos': pos, 'label': label, 'frame_ind': record_tmp[rec_ind][5],
                                    'focus':'NA','BigAtt': 'NA', 'SmallAtt': 'NA','vid':vid})


            # get det_box_tmp and det_class_tmp for this frame, starts with person, then object
            for per_ind in range(len(person_tmp)):
                    ant_tmp.append(person_tmp[per_ind])

            for obj_ind in range(len(obj_tmp)):
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



            if len(ant_tmp)>0:
                ant_all.append({'ant':ant_tmp, 'attmat':adj_mat_tmp})

                # with open(os.path.join(paths.data_root,'all','img_list',vid+'.txt'),'a') as towrite:
                #
                #     towrite.write(os.path.join(paths.data_root,'all','img',vid,str((frame_ind+1)).zfill(5)+'.png'))
                #     towrite.write('\n')


        # np.save(os.path.join(paths.data_root, mode, 'ant_processed', 'vid_{}_adjmat'.format(vid)),adj_mat_all)
        # np.save(os.path.join(paths.data_root, mode, 'ant_processed', 'vid_{}_ant_all'.format(vid)),ant_all)

        np.save(os.path.join(paths.data_root, 'ant_processed', 'vid_{}_ant_all'.format(vid)),ant_all)

def main(paths):

    ReadAnnot(paths)


if __name__ == '__main__':

    paths=utils.Paths()

    paths.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'

    main(paths)


    #tmp=np.load(os.path.join(paths.data_root, 'all', 'ant_processed', 'vid_{}_ant_all.npy'.format(66)))

    pass

