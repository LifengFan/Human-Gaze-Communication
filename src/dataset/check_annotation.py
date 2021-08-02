import dataset_config
import os
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np



def check_annotation(path):


    files=sorted(os.listdir(os.path.join(path,'ant')))

    for ind in range(len(files)):

        with open(os.path.join(path,'ant',files[ind]),'r') as f:
            lines=f.readlines()

        vid=files[ind].strip('NewAnt_').strip('.txt')

        imgs=sorted(os.listdir(os.path.join(path,'img',vid)))

        for idx in range(len(imgs)):

            frame=int(imgs[idx].strip('.png'))-1

            #rgb_img = scipy.misc.imread(os.path.join(path, 'img', vid, imgs[idx]), mode='RGB')
            rgb_img=np.array(Image.open(os.path.join(path, 'img', vid, imgs[idx])),dtype=np.uint8)

            fig, ax=plt.subplots(1)
            ax.imshow(rgb_img)

            record_tmp=list()

            for rec_ind in range(len(lines)):

                if int(lines[rec_ind].split(' ')[5])==frame:
                    record_tmp.append(lines[rec_ind].strip().split(' '))


            for tmp_ind in range(len(record_tmp)):

                xmin=int(record_tmp[tmp_ind][1])
                ymin=int(record_tmp[tmp_ind][2])
                xmax=int(record_tmp[tmp_ind][3])
                ymax=int(record_tmp[tmp_ind][4])

                # the up and left point (xmin, ymin)  width height

                rect=patches.Rectangle((xmin,ymin),width=xmax-xmin+1,height=ymax-ymin+1,linewidth=2,edgecolor='r',facecolor='none')

                ax.add_patch(rect)

            plt.show()


            pass

            #scipy.misc.imshow(rgb_img)
    pass



def main():

    path=dataset_config.Paths()

    check_annotation(os.path.join(path.data_root,'all'))

    pass



if __name__=='__main__':

    main()
