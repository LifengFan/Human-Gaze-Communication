import os
from os import listdir
from os.path import isfile, join
import numpy as np
import os.path as op
from PIL import Image,  ImageFont
import PIL.ImageDraw as ImageDraw

def attmat_sim(temp, attmat):

    if attmat.shape[0]==temp.shape[0] and attmat.shape[0]>1:
        return (attmat==temp).all()
    elif attmat.shape[0]==temp.shape[0] and attmat.shape[0]==1:
        return (attmat == temp)
    elif attmat.shape[0]!=temp.shape[0]:
        return False


def compress_seq():

    for v in range(1,302):

        if v==255 or v==291:
            continue

        vid=str(v)

        ant = np.load(op.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/ant_processed/',
                              'vid_{}_ant_all.npy'.format(vid)))

        seq_all = []
        seq = []
        temp = None
        min_len = np.inf

        for i in range(len(ant)):

            attmat = ant[i]['attmat']

            if temp is None:
                temp = attmat

            if attmat_sim(temp,attmat):
                seq.append(ant[i])
            else:
                seq_all.append(seq)
                min_len = min(min_len, len(seq))
                seq = []
                temp = attmat
                seq.append(ant[i])

        seq_all.append(seq)

        print('Min len for vid {} is {}'.format(vid, min_len))

        # sampling
        final_seq = []
        for i in range(len(seq_all)):
            tmp = seq_all[i]

            idx = np.random.randint(0, len(tmp), size=min(len(tmp), 10))
            for j in range(len(idx)):
                final_seq.append(tmp[idx[j]])

        visualize_seq(final_seq)

        #np.save('/home/lfan/Dropbox/Projects/ICCV19/DATA/compressed/{}.npy'.format(vid), final_seq)



def visualize_seq(seq):

    vid = seq[0]['ant'][0]['vid']

    if not op.exists('/home/lfan/Dropbox/Projects/ICCV19/DATA/viz_compressed/' + vid):
        os.mkdir('/home/lfan/Dropbox/Projects/ICCV19/DATA/viz_compressed/' + vid)

    for isq in range(len(seq)):
        frame = seq[isq]
        fid = frame['ant'][0]['frame_ind']

        img = np.load(op.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/img_np/{}/{}.npy'.format(vid, str(
            int(fid) + 1).zfill(5))))

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
            draw.line([(left1, top1), (left1, bottom1), (right1, bottom1), (right1, top1), (left1, top1)],
                      width=4,
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

        im.save('/home/lfan/Dropbox/Projects/ICCV19/DATA/viz_compressed/' + vid + '/' + fid + '.jpg')



def main():

    compress_seq()
    #tmp=np.load('/home/lfan/Dropbox/Projects/ICCV19/RunComm_out/data/all/ant_processed/vid_1_ant_all.npy')

    pass

if __name__=='__main__':

    main()

