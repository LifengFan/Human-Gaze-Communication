import os.path as op
from os import listdir
import numpy as np
import os
from PIL import Image,  ImageFont
import PIL.ImageDraw as ImageDraw
import pickle

# Small label
#-------------------------------------
# 'single','mutual','avert','refer','follow','share'
# single: single in SingleGaze
# mutual: mutual in any event
# avert: avert in any event -- check mutual format
# refer: refer in any event --check mutual format
# follow: follow in any event -- check refer format
# share: share in any event

data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'

def test_atomic_single(rec1, rec2):

    # if rec['BigAtt']=='SingleGaze' and rec['SmallAtt']=='single':
    #     return True
    # else:
    #     return False

    if (rec1['label'].startswith('Person') and rec2['label'].startswith('Person')) and ((rec1['BigAtt'] == 'SingleGaze' and rec1['SmallAtt'] == 'single') or (rec2['BigAtt'] == 'SingleGaze' and rec2['SmallAtt'] == 'single')):
        return True
    else:
        return False

def test_atomic_mutual(rec1, rec2):

    if rec1['SmallAtt']=='mutual' and rec2['SmallAtt']=='mutual':
        return True
    else:
        return False

def test_atomic_share(rec1, rec2):

    if rec1['SmallAtt']=='share' and rec2['SmallAtt']=='share':
        return True
    else:
        return False

def test_atomic_avert(rec1, rec2):

    if rec1['BigAtt']=='AvertGaze' and rec2['BigAtt']=='AvertGaze':
        return True
    else:
        return False

def test_atomic_follow(rec1, rec2):
    # follow in "GazeFollow" event or "JointAtt" event

    if rec1['BigAtt']=='GazeFollow' and rec2['BigAtt']=='GazeFollow':
        return True
    elif rec1['BigAtt']=='JointAtt' and rec2['BigAtt']=='JointAtt' and rec1['SmallAtt']=='follow':
        return True
    elif rec1['BigAtt']=='JointAtt' and rec2['BigAtt']=='JointAtt' and rec2['SmallAtt']=='follow':
        return True
    else:
        return False

def test_atomic_refer(rec1, rec2):

    if rec1['BigAtt']=='JointAtt' and rec2['BigAtt']=='JointAtt' and rec1['SmallAtt']=='refer':
        return True
    elif rec1['BigAtt']=='JointAtt' and rec2['BigAtt']=='JointAtt' and rec2['SmallAtt']=='refer':
        return True
    else:
        return False

def find_atomic_single():

    single_all = []
    # finding all valid seqs without the len limit 5. can use sampling or interpolation to satisfy the len 5 limit. no worry.
    files = [op.join(data_root, 'ant_processed', f) for f in sorted(listdir(op.join(data_root, 'ant_processed')))]

    for i in range(len(files)):
        seq = []

        video = np.load(files[i])
        vid = video[0]['ant'][0]['vid']

        print('vid {}'.format(vid))

        fid = 0
        while (True):
            print('fid {}'.format(fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None
            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_atomic_single(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid

                        t_fid = start_fid
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_atomic_single(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
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
            single_all.append(seq)
            #visualize_seq(seq, op.join(data_root, 'viz_atomic', 'single'), mode='single')

        np.save(op.join(data_root, 'atomic', 'single.npy'), single_all)


def find_atomic_mutual():

    mutual_all=[]

    # finding all valid seqs without the len limit 5. can use sampling or interpolation to satisfy the len 5 limit. no worry.

    files = [op.join(data_root, 'ant_processed', f) for f in sorted(listdir(op.join(data_root, 'ant_processed')))]

    for i in range(len(files)):
        seq = []

        video = np.load(files[i])
        vid = video[0]['ant'][0]['vid']

        print('vid {}'.format(vid))

        fid = 0
        while (True):
            print('fid {}'.format(fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None
            for nid1 in range(len(frame['ant'])-1):
                for nid2 in range(nid1+1, len(frame['ant'])):
                    if  nid1!=nid2 and test_atomic_mutual(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid

                        t_fid = start_fid
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_atomic_mutual(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid-1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        seq.append([vid, nid1, nid2, start_fid, end_fid])

                        if min_end_fid is None:
                            min_end_fid=end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)


            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid=fid+1

        if len(seq) > 0:
            mutual_all.append(seq)
            #visualize_seq(seq, op.join(data_root, 'viz_atomic', 'mutual'), mode='mutual')

        np.save(op.join(data_root, 'atomic', 'mutual.npy'), mutual_all)


def find_atomic_share():

    share_all=[]

    # finding all valid seqs without the len limit 5. can use sampling or interpolation to satisfy the len 5 limit. no worry.

    files = [op.join(data_root, 'ant_processed', f) for f in sorted(listdir(op.join(data_root, 'ant_processed')))]

    for i in range(len(files)):
        seq = []

        video = np.load(files[i])
        vid = video[0]['ant'][0]['vid']

        print('vid {}'.format(vid))

        fid = 0
        while (True):
            print('fid {}'.format(fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None

            for nid1 in range(len(frame['ant'])-1):
                for nid2 in range(nid1+1, len(frame['ant'])):
                    if  nid1!=nid2 and test_atomic_share(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid

                        t_fid = start_fid
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_atomic_share(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid-1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        seq.append([vid, nid1, nid2, start_fid, end_fid])
                        if min_end_fid is None:
                            min_end_fid=end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid=fid+1

        if len(seq) > 0:
            share_all.append(seq)
            #visualize_seq(seq, op.join(data_root, 'viz_atomic', 'share'), mode='share')

        np.save(op.join(data_root, 'atomic', 'share.npy'), share_all)

def find_atomic_avert():

    avert_all = []

    # finding all valid seqs without the len limit 5. can use sampling or interpolation to satisfy the len 5 limit. no worry.

    files = [op.join(data_root, 'ant_processed', f) for f in sorted(listdir(op.join(data_root, 'ant_processed')))]

    for i in range(len(files)):
        seq = []

        video = np.load(files[i])
        #video=np.load(op.join(data_root,'ant_processed','vid_107_ant_all.npy'))
        vid = video[0]['ant'][0]['vid']

        print('vid {}'.format(vid))

        fid = 0
        # outside loop
        while (True):
            print('fid {}'.format(fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None

            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_atomic_avert(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid
                        t_fid = start_fid

                        # inside loop
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_atomic_avert(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid - 1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        seq.append([vid, nid1, nid2, start_fid, end_fid])
                        if min_end_fid is None:
                            min_end_fid=end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid=fid+1


        if len(seq) > 0:
            avert_all.append(seq)
            #visualize_seq(seq, op.join(data_root, 'viz_atomic', 'avert'), mode='avert')

        np.save(op.join(data_root, 'atomic', 'avert.npy'), avert_all)

def find_atomic_follow():

    follow_all = []

    # finding all valid seqs without the len limit 5. can use sampling or interpolation to satisfy the len 5 limit. no worry.
    files = [op.join(data_root, 'ant_processed', f) for f in sorted(listdir(op.join(data_root, 'ant_processed')))]

    for i in range(len(files)):
        seq = []

        video = np.load(files[i])
        #video=np.load(op.join(data_root,'ant_processed','vid_107_ant_all.npy'))
        vid = video[0]['ant'][0]['vid']

        print('vid {}'.format(vid))

        fid = 0
        # outside loop
        while (True):
            print('fid {}'.format(fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None

            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_atomic_follow(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid
                        t_fid = start_fid

                        # inside loop
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break

                            try:
                                if not test_atomic_follow(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid - 1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break

                        if video[start_fid]['ant'][nid1]['BigAtt']=='GazeFollow':
                            seq.append([vid, nid1, nid2, start_fid, end_fid])
                        elif start_fid - 5 >= 0 and len(video[start_fid - 5]['ant']) > nid2 and len(video[start_fid - 4]['ant']) > nid2 \
                                    and len(video[start_fid - 3]['ant']) > nid2 and len(video[start_fid - 2]['ant']) > nid2 \
                                    and len(video[start_fid - 1]['ant']) > nid2:

                                seq.append([vid, nid1, nid2, start_fid - 5, end_fid])
                        elif start_fid - 3 >= 0 and len(video[start_fid - 3]['ant']) > nid2 and len(video[start_fid - 2]['ant']) > nid2 \
                                    and len(video[start_fid - 1]['ant']) > nid2:
                                seq.append([vid, nid1, nid2, start_fid - 3, end_fid])
                        else:
                                seq.append([vid, nid1, nid2, start_fid, end_fid])


                        if min_end_fid is None:
                            min_end_fid=end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid=fid+1


        if len(seq) > 0:
            follow_all.append(seq)
            #visualize_seq(seq, op.join(data_root, 'viz_atomic', 'follow'), mode='follow')

        np.save(op.join(data_root, 'atomic', 'follow.npy'), follow_all)


def find_atomic_refer():

    refer_all = []

    # finding all valid seqs without the len limit 5. can use sampling or interpolation to satisfy the len 5 limit. no worry.
    files = [op.join(data_root, 'ant_processed', f) for f in sorted(listdir(op.join(data_root, 'ant_processed')))]

    for i in range(len(files)):
        seq = []

        video = np.load(files[i])
        #video=np.load(op.join(data_root,'ant_processed','vid_107_ant_all.npy'))
        vid = video[0]['ant'][0]['vid']

        print('vid {}'.format(vid))

        fid = 0
        # outside loop
        while (True):
            print('fid {}'.format(fid))

            # need to update start_fid for a new round
            if fid >= len(video):
                break

            frame = video[fid]
            min_end_fid = None

            for nid1 in range(len(frame['ant']) - 1):
                for nid2 in range(nid1 + 1, len(frame['ant'])):
                    if nid1 != nid2 and test_atomic_refer(frame['ant'][nid1], frame['ant'][nid2]):

                        start_fid = fid
                        t_fid = start_fid

                        # inside loop
                        while (True):
                            t_fid += 1

                            if t_fid >= len(video):
                                end_fid = t_fid - 1
                                break
                            try:
                                if not test_atomic_refer(video[t_fid]['ant'][nid1], video[t_fid]['ant'][nid2]):
                                    end_fid = t_fid - 1
                                    break
                            except:
                                end_fid = t_fid - 1
                                break
                        #todo: invalid try here!

                        if start_fid-5>=0 and len(video[start_fid-5]['ant'])>nid2 and len(video[start_fid-4]['ant'])>nid2 \
                            and len(video[start_fid-3]['ant'])>nid2 and len(video[start_fid-2]['ant'])>nid2 \
                            and len(video[start_fid-1]['ant'])>nid2:

                            seq.append([vid, nid1, nid2, start_fid-5, end_fid])
                        elif start_fid-3>=0 and len(video[start_fid-3]['ant'])>nid2 and len(video[start_fid-2]['ant'])>nid2 \
                            and len(video[start_fid-1]['ant'])>nid2:

                                seq.append([vid, nid1, nid2, start_fid-3, end_fid])
                        else:
                                seq.append([vid, nid1, nid2, start_fid, end_fid])

                        if min_end_fid is None:
                            min_end_fid=end_fid
                        else:
                            min_end_fid = min(min_end_fid, end_fid)

            if min_end_fid is not None:
                fid = min_end_fid + 1
            else:
                fid=fid+1


        if len(seq) > 0:
            refer_all.append(seq)
            #visualize_seq(seq, op.join(data_root, 'viz_atomic', 'refer'), mode='refer')

        np.save(op.join(data_root, 'atomic', 'refer.npy'), refer_all)


def visualize_seq(seq, save_path, mode):

    vid=seq[0][0]
    video = np.load(op.join(data_root,'ant_processed', 'vid_{}_ant_all.npy'.format(vid)))

    if not op.exists(op.join(save_path, vid)):
        os.mkdir(op.join(save_path, vid))

    for i_seq in range(len(seq)):

        if mode=='single':
            #vid, nid, start_fid, end_fid=seq[i_seq]
            vid, nid1, nid2, start_fid, end_fid = seq[i_seq]
        elif mode=='mutual' or mode=='share' or mode=='avert' or mode=='follow' or mode=='refer':
            vid, nid1, nid2, start_fid, end_fid = seq[i_seq]

        if not op.exists(op.join(save_path, vid, str(i_seq))):
            os.mkdir(op.join(save_path, vid, str(i_seq)))

        for f_index in range(start_fid, end_fid+1):

            frame = video[f_index]
            fid = frame['ant'][0]['frame_ind']

            img = np.load(op.join(data_root,'img_np',vid, '{}.npy'.format(str(int(fid) + 1).zfill(5))))

            im = Image.fromarray(img)
            draw = ImageDraw.Draw(im)

            if mode=='single':
                # pos = frame['ant'][nid]['pos']
                # BL = frame['ant'][nid]['BigAtt']
                # SL = frame['ant'][nid]['SmallAtt']
                #
                # left1 = int(pos[0])
                # top1 = int(pos[1])
                # right1 = int(pos[2])
                # bottom1 = int(pos[3])
                # draw.line([(left1, top1), (left1, bottom1), (right1, bottom1), (right1, top1), (left1, top1)], width=4,
                #           fill='red')
                #
                # draw.text((left1, top1), BL, fill=(255, 255, 255, 255))
                # draw.text((left1, top1 + 10), SL, fill=(255, 255, 255, 255))
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

            elif mode=='mutual' or mode=='share' or mode=='avert' or mode=='follow' or mode=='refer':

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
                              width=4,fill='red')

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


def main():

    find_atomic_single()
    find_atomic_mutual()
    find_atomic_share()
    find_atomic_avert()
    find_atomic_follow()
    find_atomic_refer()



if __name__=='__main__':

    main()


