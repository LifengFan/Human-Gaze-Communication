"""
Created on Oct 8, 2018

Author: Lifeng Fan

Description:

"""

import os
from os import listdir
from os.path import isfile
import dataset_config


def gen_img_list(paths):


        with open(paths.train_img_list_file,'w') as to_write:

             folders=sorted(listdir(paths.train_img))
             for folder_ind in range(len(folders)):
                 imgs=[os.path.join(paths.train_img,folders[folder_ind],f) for f in sorted(listdir(os.path.join(paths.train_img,folders[folder_ind]))) if isfile(os.path.join(paths.train_img,folders[folder_ind],f))]

                 for img_ind in range(len(imgs)-2):
                     to_write.write(imgs[img_ind])
                     to_write.write('\n')

        print("finished generating train img list!")


        with open(paths.val_img_list_file,'w') as to_write:

             folders=sorted(listdir(paths.val_img))
             for folder_ind in range(len(folders)):
                 imgs=[os.path.join(paths.val_img,folders[folder_ind],f) for f in sorted(listdir(os.path.join(paths.val_img,folders[folder_ind]))) if isfile(os.path.join(paths.val_img,folders[folder_ind],f))]
                 for img_ind in range(len(imgs)-2):
                     to_write.write(imgs[img_ind])
                     to_write.write('\n')

        print("finished generating validate img list!")

        with open(paths.test_img_list_file,'w') as to_write:

             folders=sorted(listdir(paths.test_img))
             for folder_ind in range(len(folders)):
                 imgs=[os.path.join(paths.test_img,folders[folder_ind],f) for f in sorted(listdir(os.path.join(paths.test_img,folders[folder_ind]))) if isfile(os.path.join(paths.test_img,folders[folder_ind],f))]
                 for img_ind in range(len(imgs)-2):
                     to_write.write(imgs[img_ind])
                     to_write.write('\n')

        print("finished generating test img list!")


def gen_img_list_per_vid(paths):

    # train

    folders = sorted(listdir(paths.train_img))

    for folder_ind in range(len(folders)):

        imgs = [os.path.join(paths.train_img, folders[folder_ind], f) for f in
                sorted(listdir(os.path.join(paths.train_img, folders[folder_ind]))) if
                isfile(os.path.join(paths.train_img, folders[folder_ind], f))]



        with open(os.path.join(paths.data_root, 'train', 'img_list', folders[folder_ind]+'.txt'), 'w') as to_write:

             for img_ind in range(len(imgs) - 2):

                 to_write.write(imgs[img_ind])
                 to_write.write('\n')

    print("finished generating train img list!")


    # validate

    folders = sorted(listdir(paths.val_img))

    for folder_ind in range(len(folders)):

        imgs = [os.path.join(paths.val_img, folders[folder_ind], f) for f in
                sorted(listdir(os.path.join(paths.val_img, folders[folder_ind]))) if
                isfile(os.path.join(paths.val_img, folders[folder_ind], f))]


        with open(os.path.join(paths.data_root, 'validate', 'img_list', folders[folder_ind]+'.txt'), 'w') as to_write:

             for img_ind in range(len(imgs) - 2):

                 to_write.write(imgs[img_ind])
                 to_write.write('\n')

    print("finished generating validate img list!")


    # test

    folders = sorted(listdir(paths.test_img))

    for folder_ind in range(len(folders)):

        imgs = [os.path.join(paths.test_img, folders[folder_ind], f) for f in
                sorted(listdir(os.path.join(paths.test_img, folders[folder_ind]))) if
                isfile(os.path.join(paths.test_img, folders[folder_ind], f))]


        with open(os.path.join(paths.data_root, 'test', 'img_list', folders[folder_ind]+'.txt'), 'w') as to_write:

             for img_ind in range(len(imgs) - 2):

                 to_write.write(imgs[img_ind])
                 to_write.write('\n')

    print("finished generating test img list!")




def main(paths):

    gen_img_list_per_vid(paths)
    pass


if __name__=='__main__':
    paths=dataset_config.Paths()
    main(paths)