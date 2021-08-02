import numpy as np
import torch

def collate_fn_test_atomic_in_event(batch):

    N = len(batch)
    max_node_num = 4
    sq_len=5

    head_batch = np.zeros((N, sq_len, 4, 3, 224, 224))
    pos_batch = np.zeros((N, sq_len, max_node_num, 6))
    attmat_batch = np.zeros((N, sq_len, max_node_num, max_node_num))
    atomic_label_batch = np.zeros((N))
    ID_batch=np.zeros((N,8))
    # event_label_batch= np.zeros((N))
    # pos_in_sq_batch= np.zeros((N))

    #for i, (head_patch_sq, pos_sq, attmat_sq, atomic_label, face_kp_sq, ID_rec) in enumerate(batch):
    for i, (head_patch_sq, pos_sq, attmat_sq, atomic_label, ID_rec) in enumerate(batch):
        #head_patch_sq, pos_sq, attmat_sq, atomic_label, event_label, pos_in_sq

            head_batch[i, ...] = head_patch_sq
            pos_batch[i, ...] = pos_sq
            attmat_batch[i, ...] = attmat_sq
            atomic_label_batch[i] = atomic_label
            # event_label_batch[i]=event_label
            # pos_in_sq_batch[i]=pos_in_sq
            #face_kp_batch[i,...]=face_kp_sq
            ID_batch[i,:]=ID_rec

    head_batch = torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    attmat_batch = torch.FloatTensor(attmat_batch)
    atomic_label_batch = torch.LongTensor(atomic_label_batch)
    # event_label_batch=torch.LongTensor(event_label_batch)
    # pos_in_sq_batch=torch.LongTensor(pos_in_sq_batch)
    #face_kp_batch=torch.FloatTensor(face_kp_batch)
    ID_batch=torch.LongTensor(ID_batch)

    #return head_batch, pos_batch, attmat_batch, atomic_label_batch, face_kp_batch, ID_batch
    return head_batch, pos_batch, attmat_batch, atomic_label_batch, ID_batch #event_label_batch, pos_in_sq_batch  #, ID_batch


def collate_fn_atomic(batch):

    N = len(batch)
    max_node_num = 4
    sq_len=5

    head_batch = np.zeros((N, sq_len, 4, 3, 224, 224))
    pos_batch = np.zeros((N, sq_len, max_node_num, 6))
    attmat_batch = np.zeros((N, sq_len, max_node_num, max_node_num))
    atomic_label_batch = np.zeros((N))
    face_kp_batch=np.zeros((N, sq_len, 2, 68, 3))
    ID_batch=np.zeros((N, 5))

    #for i, (head_patch_sq, pos_sq, attmat_sq, atomic_label, face_kp_sq, ID_rec) in enumerate(batch):
    for i, (head_patch_sq, pos_sq, attmat_sq, atomic_label) in enumerate(batch):

            head_batch[i, ...] = head_patch_sq
            pos_batch[i, ...] = pos_sq
            attmat_batch[i, ...] = attmat_sq
            atomic_label_batch[i] = atomic_label
            #face_kp_batch[i,...]=face_kp_sq
            #ID_batch[i,:]=ID_rec

    head_batch = torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    attmat_batch = torch.FloatTensor(attmat_batch)
    atomic_label_batch = torch.LongTensor(atomic_label_batch)
    #face_kp_batch=torch.FloatTensor(face_kp_batch)
    #ID_batch=torch.LongTensor(ID_batch)

    #return head_batch, pos_batch, attmat_batch, atomic_label_batch, face_kp_batch, ID_batch
    return head_batch, pos_batch, attmat_batch, atomic_label_batch #, ID_batch


def collate_fn_lstm(batch):

    N = len(batch)
    max_node_num = 6
    #sq_len=10
    sq_len=20

    img_batch = np.zeros((N, sq_len, max_node_num, 3, 224, 224))
    pos_batch = np.zeros((N, sq_len, max_node_num, 6))
    SL_batch = np.zeros((N, sq_len, max_node_num))
    num_rec_batch = np.zeros((N, sq_len))
    attmat_batch = np.zeros((N, sq_len, max_node_num, max_node_num))

    for i, (patches, poses, s_labels, node_num, attmat) in enumerate(batch):

            img_batch[i, ...] = patches
            pos_batch[i, ...] = poses
            SL_batch[i, ...] = s_labels
            num_rec_batch[i,:] = node_num
            attmat_batch[i, ...] = attmat

    img_batch = torch.FloatTensor(img_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    SL_batch = torch.LongTensor(SL_batch)
    num_rec_batch = torch.IntTensor(num_rec_batch)
    attmat_batch = torch.FloatTensor(attmat_batch)

    return img_batch, pos_batch, SL_batch, num_rec_batch, attmat_batch



def collate_fn_Nodes2SL(batch):

    N=len(batch)
    max_node_num=6

    img_batch=np.zeros((N, max_node_num,3,224,224))
    pos_batch=np.zeros((N,max_node_num,6))
    SL_batch=np.zeros((N,max_node_num))
    num_rec_batch=np.zeros((N))
    attmat_batch=np.zeros((N, max_node_num, max_node_num))

    for i, (patches, poses, s_labels, node_num, attmat) in enumerate(batch):

        img_batch[i, :node_num, ...]=patches
        pos_batch[i, :node_num, :]=poses
        SL_batch[i, :node_num]=s_labels
        num_rec_batch[i]=node_num
        attmat_batch[i,:node_num, :node_num]=attmat

    img_batch=torch.FloatTensor(img_batch)
    pos_batch=torch.FloatTensor(pos_batch)
    SL_batch=torch.LongTensor(SL_batch)
    num_rec_batch=torch.IntTensor(num_rec_batch)
    attmat_batch=torch.FloatTensor(attmat_batch)


    return img_batch, pos_batch, SL_batch, num_rec_batch, attmat_batch


def collate_fn_attmat(batch):

    N=len(batch)

    head_batch = np.zeros((N,3,224,224))
    pos_batch=np.zeros((N,12))
    att_gt_batch=np.zeros((N))


    for i, (head_patch, pos, att_gt) in enumerate(batch):

         head_batch[i,...]=head_patch
         pos_batch[i,...]=pos
         att_gt_batch[i]=att_gt


    head_batch=torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    att_gt_batch=torch.LongTensor(att_gt_batch)


    return head_batch, pos_batch, att_gt_batch


def collate_fn_headpose(batch):

    batch_size=len(batch)
    print('batch size {}'.format(batch_size))

    headpatch_batch=np.zeros((batch_size, 3,224,224))
    rpy_batch=np.zeros((batch_size,3))

    for i ,(headpatch, rpy) in enumerate(batch):

        headpatch_batch[i,...]=headpatch
        rpy_batch[i,...]=rpy

    headpatch_batch=torch.FloatTensor(headpatch_batch)
    rpy_batch=torch.FloatTensor(rpy_batch)

    return headpatch_batch, rpy_batch




def main():

    pass

if __name__=='__main__':

    main()