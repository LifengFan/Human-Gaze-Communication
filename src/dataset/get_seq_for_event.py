import os.path as op
from os import listdir
import numpy as np
import os
from PIL import Image,  ImageFont
import PIL.ImageDraw as ImageDraw
import pickle
import matplotlib.pyplot as plt

# Big label
#-------------------------------------
# 'SingleGaze','GazeFollow','AvertGaze','MutualGaze','JointAtt'

data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'
files = [op.join(data_root, 'ant_processed', f) for f in sorted(listdir(op.join(data_root, 'ant_processed')))]

def test_SingleGaze(rec1, rec2):

    if (rec1['label'].startswith('Person') and rec2['label'].startswith('Person')) and \
        (rec1['BigAtt'] == 'SingleGaze' and rec2['BigAtt'] == 'SingleGaze'):
        return True
    else:
        return False

def test_GazeFollow(rec1, rec2):

    if (rec1['label'].startswith('Person') and rec2['label'].startswith('Person')) and \
        (rec1['BigAtt'] == 'GazeFollow' and rec2['BigAtt'] == 'GazeFollow'):
        return True
    else:
        return False

def test_AvertGaze(rec1, rec2):

    if (rec1['label'].startswith('Person') and rec2['label'].startswith('Person')) and \
        (rec1['BigAtt'] == 'AvertGaze' and rec2['BigAtt'] == 'AvertGaze'):
        return True
    else:
        return False

def test_MutualGaze(rec1, rec2):

    if (rec1['label'].startswith('Person') and rec2['label'].startswith('Person')) and \
        (rec1['BigAtt'] == 'MutualGaze' and rec2['BigAtt'] == 'MutualGaze'):
        return True
    else:
        return False

def test_JointAtt(rec1, rec2):

    if (rec1['label'].startswith('Person') and rec2['label'].startswith('Person')) and \
        (rec1['BigAtt'] == 'JointAtt' and rec2['BigAtt'] == 'JointAtt'):
        return True
    else:
        return False

#----------------------------------------------------------------------------------------

def test5_single(rec1, rec2):

    flag=True
    for k in range(5):
        if rec1[k]['SmallAtt'] != 'single' or rec2[k]['SmallAtt'] != 'single':
            flag = False

    return flag

def test5_mutual(rec1, rec2):

    flag=True
    for k in range(5):
        if rec1[k]['focus'].strip('P')!=rec2[k]['label'].strip('Person') or rec2[k]['focus'].strip('P')!=rec1[k]['label'].strip('Person'):
            flag=False

    return flag

def test5_share(rec1, rec2):

    flag = True
    for k in range(5):
        if rec1[k]['focus']=='NA' or rec2[k]['focus']=='NA' or rec1[k]['focus']!=rec2[k]['focus']:
            flag=False

    return flag

def test5_avert(rec1, rec2):

    flag = False
    for k in range(5):
        if rec1[k]['SmallAtt'] == 'avert' or rec2[k]['SmallAtt'] == 'avert':
            flag = True

    return flag

def test5_follow(rec1, rec2):
    # follow in "GazeFollow" event or "JointAtt" event
    flag = False
    for k in range(5):
        if rec1[k]['SmallAtt'] == 'follow' or rec2[k]['SmallAtt'] == 'follow':
            flag = True

    return flag


def test5_refer(rec1, rec2):

    flag = False
    for k in range(5):
        if rec1[k]['SmallAtt'] == 'refer' or rec2[k]['SmallAtt'] == 'refer':
            flag = True

    return flag

#----------------------------------------------------------------------------------------
def find_event_SingleGaze():

    SingleGaze = []

    for file_i in range(len(files)):
        seq=[]

        f=files[file_i]
        video=np.load(f)
        vid = video[0]['ant'][0]['vid']
        print('vid {}'.format(vid))

        fid = 0
        while (True):
            #print('fid {}'.format(fid))
            print('SingleGaze vid {} fid {}'.format(vid, fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None
            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_SingleGaze(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid

                        t_fid = start_fid
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_SingleGaze(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid - 1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        seq.append([vid, nid1, nid2, start_fid, end_fid])

                        if min_end_fid is None:
                            min_end_fid = end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid = fid + 1

        if len(seq) > 0:
            SingleGaze.append(seq)
            visualize_seq(seq, op.join(data_root, 'viz_event', 'SingleGaze_and'))

        np.save(op.join(data_root, 'event', 'SingleGaze_and.npy'), SingleGaze)

def find_event_GazeFollow():

    GazeFollow = []

    for file_i in range(len(files)):
        seq=[]

        f=files[file_i]
        video=np.load(f)
        vid = video[0]['ant'][0]['vid']
        print('vid {}'.format(vid))

        fid = 0
        while (True):
            #print('fid {}'.format(fid))
            print('GazeFollow vid {} fid {}'.format(vid, fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None
            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_GazeFollow(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid

                        t_fid = start_fid
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_GazeFollow(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid - 1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        seq.append([vid, nid1, nid2, start_fid, end_fid])

                        if min_end_fid is None:
                            min_end_fid = end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid = fid + 1

        if len(seq) > 0:
            GazeFollow.append(seq)
            visualize_seq(seq, op.join(data_root, 'viz_event', 'GazeFollow'))

        np.save(op.join(data_root, 'event', 'GazeFollow.npy'), GazeFollow)

def find_event_AvertGaze():

    AvertGaze = []

    for file_i in range(len(files)):
        seq=[]

        f=files[file_i]
        video=np.load(f)
        vid = video[0]['ant'][0]['vid']
        print('vid {}'.format(vid))

        fid = 0
        while (True):
            #print('fid {}'.format(fid))
            print('AvertGaze vid {} fid {}'.format(vid, fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None
            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_AvertGaze(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid

                        t_fid = start_fid
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_AvertGaze(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid - 1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        seq.append([vid, nid1, nid2, start_fid, end_fid])

                        if min_end_fid is None:
                            min_end_fid = end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid = fid + 1

        if len(seq) > 0:
            AvertGaze.append(seq)
            visualize_seq(seq, op.join(data_root, 'viz_event', 'AvertGaze'))

        np.save(op.join(data_root, 'event', 'AvertGaze.npy'), AvertGaze)

def find_event_MutualGaze():

    MutualGaze = []

    for file_i in range(len(files)):
        seq=[]

        f=files[file_i]
        video=np.load(f)
        vid = video[0]['ant'][0]['vid']
        print('vid {}'.format(vid))

        fid = 0
        while (True):
            #print('fid {}'.format(fid))
            print('MutualGaze vid {} fid {}'.format(vid, fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None
            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_MutualGaze(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid

                        t_fid = start_fid
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_MutualGaze(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid - 1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        seq.append([vid, nid1, nid2, start_fid, end_fid])

                        if min_end_fid is None:
                            min_end_fid = end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid = fid + 1

        if len(seq) > 0:
            MutualGaze.append(seq)
            visualize_seq(seq, op.join(data_root, 'viz_event', 'MutualGaze'))

        np.save(op.join(data_root, 'event', 'MutualGaze.npy'), MutualGaze)


def find_event_JointAtt():

    JointAtt = []

    for file_i in range(len(files)):
        seq=[]

        f=files[file_i]
        video=np.load(f)
        vid = video[0]['ant'][0]['vid']
        #print('vid {}'.format(vid))

        fid = 0
        while (True):

            print('JointAtt vid {} fid {}'.format(vid, fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None
            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_JointAtt(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid

                        t_fid = start_fid
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_JointAtt(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid - 1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        seq.append([vid, nid1, nid2, start_fid, end_fid])

                        if min_end_fid is None:
                            min_end_fid = end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid = fid + 1

        if len(seq) > 0:
            JointAtt.append(seq)
            visualize_seq(seq, op.join(data_root, 'viz_event', 'JointAtt'))

        np.save(op.join(data_root, 'event', 'JointAtt.npy'), JointAtt)


def visualize_seq(seq, save_path):

    vid=seq[0][0]
    video = np.load(op.join(data_root,'ant_processed', 'vid_{}_ant_all.npy'.format(vid)))

    if not op.exists(op.join(save_path, vid)):
        os.mkdir(op.join(save_path, vid))

    for i_seq in range(len(seq)):


        vid, nid1, nid2, start_fid, end_fid = seq[i_seq]


        if not op.exists(op.join(save_path, vid, str(i_seq))):
            os.mkdir(op.join(save_path, vid, str(i_seq)))

        for f_index in range(start_fid, end_fid+1):

            frame = video[f_index]
            fid = frame['ant'][0]['frame_ind']

            img = np.load(op.join(data_root,'img_np',vid, '{}.npy'.format(str(int(fid) + 1).zfill(5))))

            im = Image.fromarray(img)
            draw = ImageDraw.Draw(im)

            pos1 = frame['ant'][nid1]['pos']
            BL1 = frame['ant'][nid1]['BigAtt']
            SL1 = frame['ant'][nid1]['SmallAtt']

            left1 = int(pos1[0])
            top1 = int(pos1[1])
            right1 = int(pos1[2])
            bottom1 = int(pos1[3])
            draw.line([(left1, top1), (left1, bottom1), (right1, bottom1), (right1, top1), (left1, top1)], width=4,
                      fill='red')

            draw.text((left1, top1), BL1, fill=(255, 255, 255, 255))
            draw.text((left1, top1 + 10), SL1, fill=(255, 255, 255, 255))

            try:
                pos2 = frame['ant'][nid2]['pos']
                BL2 = frame['ant'][nid2]['BigAtt']
                SL2 = frame['ant'][nid2]['SmallAtt']

                left2 = int(pos2[0])
                top2 = int(pos2[1])
                right2 = int(pos2[2])
                bottom2 = int(pos2[3])
                draw.line([(left2, top2), (left2, bottom2), (right2, bottom2), (right2, top2), (left2, top2)],
                          width=4, fill='red')

                draw.text((left2, top2), BL2, fill=(255, 255, 255, 255))
                draw.text((left2, top2 + 10), SL2, fill=(255, 255, 255, 255))
            except:
                pass

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

            im.save(op.join(save_path, vid, str(i_seq), str(f_index)+'.jpg'))



def get_event_seq():

    seq_dict={'SingleGaze':[],'GazeFollow':[],'AvertGaze':[],'MutualGaze':[],'JointAtt':[]}

    mode_all=['SingleGaze','GazeFollow','AvertGaze','MutualGaze','JointAtt']

    for mode in mode_all:

        seq_all=np.load(op.join(data_root, 'event', '{}.npy'.format(mode)))

        for i in range(len(seq_all)):
            seq=seq_all[i]

            vid=seq[0][0]
            video=np.load(op.join(data_root,'ant_processed', 'vid_{}_ant_all.npy'.format(vid)))

            for j in range(len(seq)):

                vid, nid1, nid2, start_fid, end_fid=seq[j]

                alab_seq=[]

                t_fid = start_fid
                while (True):
                    if t_fid + 4 <= end_fid:

                        rec1_5=[]
                        rec1_5.append(video[t_fid]['ant'][nid1])
                        rec1_5.append(video[t_fid+1]['ant'][nid1])
                        rec1_5.append(video[t_fid+2]['ant'][nid1])
                        rec1_5.append(video[t_fid+3]['ant'][nid1])
                        rec1_5.append(video[t_fid+4]['ant'][nid1])

                        rec2_5=[]
                        rec2_5.append(video[t_fid]['ant'][nid2])
                        rec2_5.append(video[t_fid+1]['ant'][nid2])
                        rec2_5.append(video[t_fid+2]['ant'][nid2])
                        rec2_5.append(video[t_fid+3]['ant'][nid2])
                        rec2_5.append(video[t_fid+4]['ant'][nid2])


                        if test5_mutual(rec1_5,rec2_5):
                            alab='mutual'
                        elif test5_share(rec1_5,rec2_5):
                            alab='share'
                        elif test5_avert(rec1_5, rec2_5):
                            alab='avert'
                        elif test5_refer(rec1_5, rec2_5):
                            alab='refer'
                        elif test5_follow(rec1_5, rec2_5):
                            alab='follow'
                        elif test5_single(rec1_5, rec2_5):
                            alab='single'
                        else:
                            alab='NA'

                        alab_seq.append(alab)

                        t_fid += 5

                    else:
                        e_end=t_fid-1
                        break

                seq_dict[mode].append(([vid, nid1, nid2, start_fid, e_end, mode],alab_seq))

    f = open(op.join(data_root, 'event', "event_sample.pkl"), "wb")
    pickle.dump(seq_dict, f)
    f.close()


def sample_event_seq(sq_len=20):
    pass

    f=open(op.join(data_root, 'event', "event_sample.pkl"), "rb")
    event_dict = pickle.load(f)

    pass


def main():
    pass

    find_event_SingleGaze()
    find_event_GazeFollow()
    find_event_AvertGaze()
    find_event_MutualGaze()
    find_event_JointAtt()

    sample_event_seq()



if __name__=='__main__':

    main()
