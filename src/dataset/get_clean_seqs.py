import utils
import os
import numpy as np


def get_clean_seqs(paths):

    file_dir=os.path.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/','ant_processed')
    files=[f for f in sorted(os.listdir(file_dir)) if os.path.isfile(os.path.join(file_dir,f))]


    for fidx in range(len(files)):
        file=files[fidx]
        vid=file.strip('vid_').strip('_ant_all.npy')

        #print('vid: '+vid)

        sequences_all=[]
        seq_cnt=0

        data=np.load(os.path.join(file_dir,file))

        num_frame=len(data)


        for frame_id in range(num_frame):

            recs=data[frame_id]['ant']
            attmat=data[frame_id]['attmat']

            if frame_id==0:
                node_num_old= len(recs)
                sequence=[{'ant':recs, 'attmat':attmat}]
                real_frame_id=int(recs[0]['frame_ind'])
            else:

                if len(recs)==node_num_old and int(recs[0]['frame_ind'])==real_frame_id+1:
                    sequence.append({'ant':recs, 'attmat':attmat})
                    real_frame_id=int(recs[0]['frame_ind'])

                else:
                    sequences_all.append(sequence)
                    seq_cnt+=1

                    node_num_old=len(recs)
                    sequence=[{'ant':recs, 'attmat':attmat}]
                    real_frame_id = int(recs[0]['frame_ind'])

        sequences_all.append(sequence)
        seq_cnt+=1

        # save sequences for this video
        #print('vid: '+ vid + ' seq_cnt: '+ str(seq_cnt))

        seq_save_cnt=0

        for seq_id in range(seq_cnt):

            if len(sequences_all[seq_id])>=20:

                #print('vid: ' + vid + ' seq_cnt: ' + str(seq_cnt))
                res=assert_REID(sequences_all[seq_id], vid, str(seq_id))

                if res:

                    cnt=len(sequences_all[seq_id])/20

                    for i in range(cnt):

                        seq_sec=sequences_all[seq_id][i*20:(i+1)*20]

                        save_name = os.path.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/', 'seqs_20', 'vid_' + vid + '_seq_' + str(seq_save_cnt) + '.npy')
                        np.save(save_name, seq_sec)
                        seq_save_cnt+=1


def assert_REID(seq, vid, seq_id):

    node_num=len(seq[0]['ant'])

    ent=list()

    for idx in range(node_num):
        ent.append(seq[0]['ant'][idx]['label'])


    for fid in range(len(seq)):

        if fid!=0:

            try:
                assert int(seq[fid]['ant'][0]['frame_ind'])==int(seq[fid-1]['ant'][0]['frame_ind'])+1
            except:
                print('SEQ ERROR: vid: {}, sqid: {}, fid: {}'.format(vid, seq_id, str(fid)))
                return False

        for nid in range(node_num):

            try:
                assert seq[fid]['ant'][nid]['label']==ent[nid]
            except:
                print('REID ERROR: vid: {}, sqid: {}, fid: {}, nid: {}'.format(vid, seq_id, str(fid), str(nid)))

                return False

    return True

def main(paths):



    get_clean_seqs(paths)

    ## ERROR: vid: 94, sqid: 1, fid: 135, nid: 0
    #
    # SEQ
    # ERROR: vid: 107, sqid: 1, fid: 2
    # SEQ
    # ERROR: vid: 108, sqid: 1, fid: 23
    # REID
    # ERROR: vid: 92, sqid: 3, fid: 7, nid: 0
    #
    # vid='92'
    # seq_id='4'
    #
    # tmp=np.load(os.path.join(paths.data_root,'all','seqs','vid_'+vid+'_seq_'+str(seq_id)+'.npy'))
    #
    # assert_REID(tmp, vid, seq_id)

    pass


if __name__=='__main__':

    paths= utils.Paths()

    main(paths)