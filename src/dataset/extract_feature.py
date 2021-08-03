"""
Created on Oct 8, 2018

Author: Lifeng Fan

Description:

"""

import torch
import os
import scipy.io as scio
import dataset.utils
import roi_feature_model
import metadata
from os import listdir
import numpy as np
import scipy.misc
import cv2
import torch.autograd
import torchvision
import metadata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def extract_node_features(paths,mode):

    input_h, input_w=224,224
    node_feature_len=1000
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(metadata.train_mean_value,metadata.train_std_value)])
    feature_network = roi_feature_model.Resnet152(num_classes=len(metadata.node_classes))
    feature_network = torch.nn.DataParallel(feature_network).cuda()

    # get the processed annotation and corresponding original image

    ant_files=[f for f in sorted(listdir(os.path.join(paths.data_root,mode,'ant_processed'))) if f.endswith('_ant_all.npy')]

    #for ant_file_ind in [3]:

    for ant_file_ind in range(len(ant_files)):
        ant_f=ant_files[ant_file_ind]

        vid=ant_f[4:-12]

        # if os.path.isfile(os.path.join(paths.data_root, mode, 'node_feature_1000', 'vid_{}_resnet_node_feature.npy'.format(vid))):
        #     continue


        print('node feature vid {}'.format(vid))

        ant_all=np.load(os.path.join(paths.data_root,mode,'ant_processed',ant_f))

        frame_num=len(ant_all)

        node_feature_all=list()

        for frame_ind in range(frame_num):
            ant=ant_all[frame_ind]
            orig_img=scipy.misc.imread(os.path.join(paths.data_root,mode,'img',vid,'{}.png'.format(str(frame_ind+1).zfill(5))),mode='RGB')

            node_feature_tmp=np.zeros((len(ant),node_feature_len))

            for ant_ind in range(len(ant)):

                pos=ant[ant_ind]['pos']
                roi_img=orig_img[int(pos[1]):(int(pos[3])+1), int(pos[0]):(int(pos[2])+1), :]

                # fig, ax = plt.subplots(1)
                # ax.imshow(roi_img)
                #
                # plt.show()

                roi_img=transform(cv2.resize(roi_img,(input_h,input_w),interpolation=cv2.INTER_LINEAR))
                roi_img=torch.autograd.Variable(roi_img.unsqueeze(0)).cuda()
                feature,_=feature_network(roi_img)

                #node_feature_tmp.append(feature.data.cpu().numpy())
                node_feature_tmp[ant_ind,:]=feature.data.cpu().numpy()

            node_feature_all.append(node_feature_tmp)

        np.save(os.path.join(paths.data_root, mode, 'node_feature_1000', 'vid_{}_resnet_node_feature'.format(vid)), node_feature_all)



def extract_edge_features(paths,mode):

    input_h, input_w = 224, 224
    edge_feature_len = 1000
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(metadata.train_mean_value,
                                                                            metadata.train_std_value)])

    # get the finetuned feature network ready!

    feature_network = roi_feature_model.Resnet152(num_classes=len(metadata.node_classes))

    feature_network = torch.nn.DataParallel(feature_network).cuda()

    # checkpoint_dir = os.path.join(paths.tmp_root, 'checkpoints', 'finetune_resnet')
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    #
    # best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
    #
    # if os.path.isfile(best_model_file):
    #     checkpoint = torch.load(best_model_file)
    #     feature_network.load_state_dict(checkpoint['state_dict'])
    #     print "Loading trained model successfully!"

    # get the processed annotation and corresponding original image

    ant_files = [f for f in sorted(listdir(os.path.join(paths.data_root, mode, 'ant_processed'))) if
                 f.endswith('_ant_all.npy')]

    #for ant_file_ind in [3]:

    for ant_file_ind in range(len(ant_files)):

        ant_f = ant_files[ant_file_ind]

        vid = ant_f[4:-12]

        # if os.path.isfile(os.path.join(paths.data_root, mode, 'edge_feature_1000', 'vid_{}_resnet_edge_feature.npy'.format(vid))):
        #
        #     continue


        print('edge feature vid {}'.format(vid))

        ant_all = np.load(os.path.join(paths.data_root, mode, 'ant_processed', ant_f))

        frame_num = len(ant_all)

        edge_feature_all = list()

        for frame_ind in range(frame_num):
            ant = ant_all[frame_ind]

            # get the human node amount and object node amount in the current frame
            human_num=0
            obj_num=0

            for i in range(len(ant)):
                if ant[i]['label'].startswith('Person'):
                    human_num+=1
                elif ant[i]['label'].startswith('Object'):
                    obj_num+=1

            orig_img = scipy.misc.imread(
                os.path.join(paths.data_root, mode, 'img', vid, '{}.png'.format(str(frame_ind + 1).zfill(5))),
                mode='RGB')


            edge_feature_tmp_per_frame=np.zeros((human_num+obj_num,human_num+obj_num,edge_feature_len))

            #edge_feature_tmp_per_frame = list()

            for ant_ind1 in range(human_num):
                #edge_feature_tmp_per_person=list()

                for ant_ind2 in range(human_num+obj_num):

                    if ant_ind2==ant_ind1:
                        continue

                    pos1=ant[ant_ind1]['pos']
                    pos2=ant[ant_ind2]['pos']

                    min_xy=np.minimum([int(pos1[0]),int(pos1[1])],[int(pos2[0]),int(pos2[1])])
                    max_xy=np.maximum([int(pos1[2]),int(pos1[3])],[int(pos2[2]),int(pos2[3])])

                    scipy.misc.imshow(orig_img[min_xy[1]:(max_xy[1]+1),min_xy[0]:(max_xy[0]+1),:])
                    roi_img = orig_img[min_xy[1]:(max_xy[1]+1),min_xy[0]:(max_xy[0]+1),:]

                    pos1_x_center=(int(pos1[0])+int(pos1[2]))*1.0/2
                    pos2_x_center=(int(pos2[0])+int(pos2[2]))*1.0/2

                    if pos1_x_center>pos2_x_center:

                        roi_img=np.fliplr(roi_img)

                    scipy.misc.imshow(roi_img)

                    roi_img = transform(cv2.resize(roi_img, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
                    roi_img = torch.autograd.Variable(roi_img.unsqueeze(0)).cuda()
                    feature, _ = feature_network(roi_img)

                    edge_feature_tmp_per_frame[ant_ind1, ant_ind2,:]=feature.data.cpu().numpy()

                    #edge_feature_tmp_per_person.append(feature.data.cpu().numpy())

                #edge_feature_tmp_per_frame.append(edge_feature_tmp_per_person)

            edge_feature_all.append(edge_feature_tmp_per_frame)

        np.save(os.path.join(paths.data_root, mode, 'edge_feature_1000', 'vid_{}_resnet_edge_feature'.format(vid)), edge_feature_all)


def main():

    paths=dataset.utils.Paths()

    for mode in ['train','validate']:

        extract_node_features(paths,mode)
        extract_edge_features(paths,mode)

    pass


if __name__=='__main__':
    main()





