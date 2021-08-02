import sys
import os
import torch, torch.nn, torch.autograd
import numpy as np
import units
from . import Atomic_HGNN
import joblib


def get_data_atomic(args):

    #args.data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'
    args.data_root='/media/ramdisk/'

    f = open(op.join(args.data_root, 'atomic', "atomic_sample.pkl"), "rb")
    atomic_dict = pickle.load(f)

    # for mode in atomic_dict.keys():
    #     random.shuffle(atomic_dict[mode])

    train_dict={'NA': list(), 'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}
    val_dict = {'NA': list(), 'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}
    test_dict = {'NA': list(), 'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}

    val_seq=[]
    test_seq=[]

    for mode in atomic_dict.keys():

        L=len(atomic_dict[mode])
        train_dict[mode].extend(atomic_dict[mode][: (L//2)])
        val_dict[mode].extend(atomic_dict[mode][(L // 2):(L // 2 + L // 10)])
        test_dict[mode].extend(atomic_dict[mode][(L // 2 + L // 10):])

        random.shuffle(train_dict[mode])
        random.shuffle(val_dict[mode])
        #random.shuffle(test_dict[mode])

    for mode in atomic_dict.keys():
        val_seq.extend(val_dict[mode])
        test_seq.extend(test_dict[mode])

    random.shuffle(val_seq)
    #random.shuffle(test_seq)

    train_set=dataset.mydataset.mydataset_atomic(train_dict,is_train=True)
    val_set=dataset.mydataset.mydataset_atomic(val_dict, is_train=True)
    test_set=dataset.mydataset.mydataset_atomic(test_dict, is_train=True)

    train_loader=torch.utils.data.DataLoader(train_set,collate_fn=collate_fn.collate_fn_atomic, batch_size=args.batch_size, shuffle=False)
    val_loader=torch.utils.data.DataLoader(val_set,collate_fn=collate_fn.collate_fn_atomic, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn_atomic, batch_size=args.batch_size,  shuffle=False)

    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader),len(val_loader),len(test_loader)))

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def main(args):

    args.cuda = args.use_cuda and torch.cuda.is_available()

    train_set, validate_set, test_set, train_loader, validate_loader, test_loader = get_data.get_data_atomic(args)

    #model = models.Atomic(args)
    model=models.Atomic_2branch_lstm(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    #{'single': 0, 'mutual': 1, 'avert': 2, 'refer': 3, 'follow': 4, 'share': 5}
    criterion = [torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 0.05, 0.25, 0.25, 0.25, 0.15])), torch.nn.MSELoss()]

    # {'NA': 0, 'single': 1, 'mutual': 2, 'avert': 3, 'refer': 4, 'follow': 5, 'share': 6}

    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=1, verbose=True, mode='max')
#--------------------------------------------------
    # ------------------------
    # use multi-gpu

    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")

        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
        #model=model.cuda()
        criterion[0] = criterion[0].cuda()
        criterion[1] = criterion[1].cuda()

    elif args.cuda:
        model = model.cuda()
        criterion[0] = criterion[0].cuda()
        criterion[1] = criterion[1].cuda()

    if args.load_best_checkpoint:
        loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)

        if loaded_checkpoint:
            args, best_epoch_acc, avg_epoch_acc, model, optimizer = loaded_checkpoint

    if args.load_last_checkpoint:
        loaded_checkpoint = utils.load_last_checkpoint(args, model, optimizer, path=args.resume,
                                                       version=args.model_load_version)

        if loaded_checkpoint:
            args, best_epoch_acc, avg_epoch_acc, model, optimizer = loaded_checkpoint

            # ------------------------------------------------------------------------------
            # Start Training!

    since = time.time()

    train_epoch_acc_all = []
    val_epoch_acc_all = []

    best_acc = 0
    avg_epoch_acc = 0

    for epoch in range(args.start_epoch, args.epochs):

        train_epoch_loss, train_epoch_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        train_epoch_acc_all.append(train_epoch_acc)

        val_epoch_loss, val_epoch_acc = validate(validate_loader, model, criterion, epoch, args)
        val_epoch_acc_all.append(val_epoch_acc)

        print('Epoch {}/{} Training Acc: {:.4f} Validation Acc: {:.4f}'.format(epoch, args.epochs - 1, train_epoch_acc,
                                                                               val_epoch_acc))
        print('*' * 15)

        scheduler.step(val_epoch_acc)

        is_best = val_epoch_acc > best_acc

        if is_best:
            best_acc = val_epoch_acc

        avg_epoch_acc = np.mean(val_epoch_acc_all)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_epoch_acc': best_acc,
            'avg_epoch_acc': avg_epoch_acc,
            'optimizer': optimizer.state_dict(), 'args': args}, is_best=is_best, directory=args.resume,
            version='epoch_{}'.format(str(epoch)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {},  Final Avg Val Acc: {}'.format(best_acc, avg_epoch_acc))

    # ----------------------------------------------------------------------------------------------------------
    # test

    loaded_checkpoint = utils.load_best_checkpoint(args, model, optimizer, path=args.resume)

    if loaded_checkpoint:
        args, best_epoch_acc, avg_epoch_acc, model, optimizer = loaded_checkpoint

    test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
    test_loss, test_acc, confmat, top2_acc = test(test_loader, model, criterion, args)

    # save test results
    if not isdir(args.save_test_res):
        os.mkdir(args.save_test_res)

    with open(os.path.join(args.save_test_res, 'raw_test_results.pkl'), 'w') as f:
        pickle.dump([test_loss, test_acc, confmat, top2_acc], f)

    print("Test Acc {}".format(test_acc))
    print("Top 2 Test Acc {}".format(top2_acc))

    # todo: need to change the mode here!
    get_metric_from_confmat(confmat, 'atomic')



def test(test_loader, model, criterion, args):
    model.eval()

    total_acc = AverageMeter()
    total_loss = AverageMeter()
    confmat=np.zeros((6,6))

    confmat_transient=np.zeros((4,4))

    total_acc_top2=AverageMeter()

    for i, (head_batch, pos_batch, attmat_batch, atomic_label_batch) in enumerate(test_loader):

        batch_size = head_batch.shape[0]

        if args.cuda:
            heads = (torch.autograd.Variable(head_batch)).cuda()
            poses = (torch.autograd.Variable(pos_batch)).cuda()
            attmat_gt=(torch.autograd.Variable(attmat_batch)).cuda()
            atomic_gt = (torch.autograd.Variable(atomic_label_batch)).cuda()


        with torch.set_grad_enabled(False):

            pred_atomic = model(heads, poses, attmat_gt, args) #[N, 6, 1,1,1]

            test_loss = 0

            for bid in range(batch_size):
                # todo:check pre_atomic dim [N,6,1,1,1]??
                tmp_loss = criterion[0](pred_atomic[bid, :, 0, 0, 0].unsqueeze(0), atomic_gt[bid].unsqueeze(0))

                # print('label loss', criterion[0](sl_pred[nid][bid, :].unsqueeze(0), sl_gt[bid, nid].unsqueeze(0)))
                # print('attmat loss', criterion[1](attmat_pred, attmat_gt))

                total_loss.update(tmp_loss.item(), 1)
                test_loss = test_loss + tmp_loss

                pred = torch.argmax(pred_atomic[bid, :, 0, 0, 0], dim=0)
                bv = (pred == atomic_gt[bid].data)
                bv = bv.type(torch.cuda.FloatTensor)
                total_acc.update(bv.item(), 1)

                # todo: use top2 acc here!
                _, sort_ind = torch.sort(pred_atomic[bid, :, 0,0,0], descending=True)
                pred_2 = sort_ind[1]

                bv2 = (pred == atomic_gt[bid].data or pred_2 == atomic_gt[bid].data)
                bv2 = bv2.type(torch.cuda.FloatTensor)
                total_acc_top2.update(bv2.item(), 1)

                confmat[atomic_gt[bid].data, pred]+=1




            print('Iter: {} Testing Loss: {:.4f} Total Avg Acc: {:.4f}, Top 2 Avg Acc: {:.4f}'.format( i,test_loss.item(),total_acc.avg, total_acc_top2.avg))


    return total_loss.avg, total_acc.avg, confmat, total_acc_top2.avg


def parse_arguments():

    path = dataset.utils.Paths()

    project_name = 'train_atomic_HGNN'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('--project-name', default=project_name, help='project name')

    # path settings
    parser.add_argument('--project-root', default=path.project_root, help='project root path')
    parser.add_argument('--tmp-root', default=path.tmp_root, help='checkpoint path')
    parser.add_argument('--data-root', default=path.data_root, help='data path')
    parser.add_argument('--log-root', default=path.log_root, help='log files path')
    parser.add_argument('--resume', default=os.path.join(path.tmp_root, 'checkpoints', project_name),help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default=os.path.join(path.tmp_root, 'test_results', project_name),help='path to save test metrics')

    # model settings
    parser.add_argument('--small-attr-class-num', type=int, default=7, help='small attribute class number')
    parser.add_argument('--big-attr-class-num', type=int, default=6, help='big attribute class number')
    parser.add_argument('--roi-feature-size',type=int, default=64*3, help='node and edge feature size')
    parser.add_argument('--message-size', type=int, default=12, help='message size of the message function')
    parser.add_argument('--lstm-seq-size', type=int, default=15, help='lstm sequence length')
    parser.add_argument('--lstm-hidden-size',type=int, default=500, help='hiddden state size of lstm')
    parser.add_argument('--link-hidden-size', type=int, default=1024, help='link hidden size of the link function')
    parser.add_argument('--link-hidden-layers', type=int, default=2, help='link hidden layers of the link function')
    parser.add_argument('--propagate-layers', type=int, default=2, help='propagate layers for message passing')



    # optimization options
    parser.add_argument('--load-last-checkpoint', default=False, help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint', default=False,help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--batch-size', type=int, default=256, help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda', default=True, help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.8, help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom', default=False, help='use visdom to visualize loss curve')

    parser.add_argument('--device-ids', default=[0,1], help='gpu ids')

    return parser.parse_args()



class Event_Classifier:

    def __init__(self):
        self.best_model='./model_best.pth'
        self.model=Atomic_HGNN
        if os.path.isfile(self.best_model):
            print("====> loading best model")
            checkpoint=torch.load(self.best_model)
        self.model.load_state_dict(checkpoint['state_dict'])

    def data_loader(self):
        pass

    def test_event(self):

        data_set, test_loader=self.data_loader()

        model=self.model
        model.eval()

        for i, (head_batch, pos_batch, attmat_batch, atomic_label_batch, ID_rec) in enumerate(self.data_loader):

            print(i)

            batch_size = head_batch.shape[0]
            heads = (torch.autograd.Variable(head_batch)).cuda()
            poses = (torch.autograd.Variable(pos_batch)).cuda()
            attmat_gt = (torch.autograd.Variable(attmat_batch)).cuda()
            ID = (torch.autograd.Variable(ID_rec)).cuda()

            with torch.set_grad_enabled(False):

                pred_atomic = model(heads, poses, attmat_gt)

                for bid in range(batch_size):

                    pred = torch.argmax(pred_atomic[bid, :], dim=0)

                    pred_list.append([pred.cpu().numpy(), ID[bid][5].cpu().numpy(), ID[bid][6].cpu().numpy(),
                                      ID[bid][7].cpu().numpy()])


def extract_top_n_objs(cate_file, gazes, bbox, mask_names):
    with open(cate_file, 'rb') as f:
        category = joblib.load(f)
    top_n_names_over_frame = dict()
    for frame_id, gaze in enumerate(gazes):
        gaze = check_norm(gaze)
        mask_name = mask_names[frame_id]
        with open(mask_name, 'rb') as f:
            masks = joblib.load(f)
        distance = []
        names = []
        for obj_name, frame_cates in category.items():
            frame_cate = frame_cates[frame_id]
            if frame_cate[0]:
                cate_id = frame_cate[1]
                sub_id = frame_cate[2]
                mask = masks[cate_id][1][sub_id]
                avg_col = np.array(mask).mean(axis=1)
                if np.array(mask)[avg_col != 0, :].shape[0] == 0 or frame_id not in bbox:
                    distance.append(-1)
                    names.append(obj_name)
                else:
                    obj_center = np.array(mask)[avg_col != 0, :].mean(axis=0)
                    eye_center = bbox[frame_id][0][2]
                    obj_direct = obj_center - eye_center
                    obj_direct = check_norm(obj_direct)
                    angle = gaze.dot(obj_direct)
                    distance.append(angle)
                    names.append(obj_name)
            else:
                distance.append(-1)
                names.append(obj_name)
        idx = np.argsort(np.array(distance))[::-1][:10]
        top_n_names = []
        for ids in idx:
            top_n_names.append(names[ids])
        top_n_names_over_frame[frame_id] = top_n_names
    return top_n_names_over_frame

def extract_attention_input(img_names, person_bbox, top_n_objs_names_over_frames, obj_names):
    input_dict = {}
    obj_bbox = dict()
    for obj_name in obj_names:
        with open(obj_name, 'rb') as f:
            obj_name = obj_name.split('/')[-1]
            obj_bbox[obj_name] = joblib.load(f)
    total_features = dict()
    for frame_id, img_name in enumerate(img_names):
        img = cv2.imread(img_name)

        head_feature = []
        if frame_id in person_bbox:
            head_box = person_bbox[frame_id][0][0]
            img_patch = img[int(head_box[1]):int(head_box[3]), int(head_box[0]): int(head_box[2])]
            img_patch = cv2.resize(img_patch, (224, 224)).reshape((3, 224, 224))
            # print(img_patch.shape)
            # cv2.imshow('img', img_patch)
            # cv2.waitKey(20)
            pos_vec = np.array([head_box[0]/img.shape[1], head_box[1]/img.shape[0], head_box[2]/img.shape[1],
                                head_box[3]/img.shape[0], (head_box[0] + head_box[2])/2/img.shape[1], (head_box[1] + head_box[3])/2/img.shape[0]])
            pos_feature = np.empty((0, 12))
            for obj_id, obj_name in enumerate(top_n_objs_names_over_frames[frame_id]):
                obj_name = obj_name.split('/')[-1]
                obj_curr = obj_bbox[obj_name][frame_id]
                obj_vec = np.array([obj_curr[0]/img.shape[1], obj_curr[1]/img.shape[0], (obj_curr[2] + obj_curr[0])/img.shape[1],
                                    (obj_curr[3]+obj_curr[1])/img.shape[0], (obj_curr[0] + obj_curr[2]/2)/img.shape[1], (obj_curr[1] + obj_curr[3]/2)/img.shape[0]])
                vec = np.hstack([pos_vec, obj_vec])
                head_feature.append(img_patch)
                pos_feature = np.vstack([pos_feature, vec])
            total_features[frame_id] = [head_feature, pos_feature]
    return total_features
    #             if obj_id == 0:
    #                 cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])),
    #                                       (int(obj_curr[0] + obj_curr[2]), int(obj_curr[1] + obj_curr[3])),
    #                                       (0, 0, 255), thickness=3)
    #             else:
    #                 cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])),
    #                               (int(obj_curr[0] + obj_curr[2]), int(obj_curr[1] + obj_curr[3])),
    #                               (255, 0, 0), thickness=3)
    #             cv2.rectangle(img, (int(head_box[0]), int(head_box[1])),
    #                           (int(head_box[2]), int(head_box[3])),
    #                           (255, 0, 0), thickness=3)
    #         cv2.imshow('img', img)
    #         cv2.waitKey(20)

def get_attention(attention_input, net, img_names):
    for frame_id in attention_input.keys():
        heads, pos = attention_input[frame_id]
        pos_ori = pos
        heads = np.array(heads).astype(float)
        for c in [0, 1, 2]:
            heads[:, c, :, :] = (heads[:, c, :, :]/255. - 0.5) / 0.5
        print(heads.shape, pos.shape)
        heads = torch.from_numpy(heads).float().cuda()
        pos = torch.from_numpy(pos).float().cuda()
        predicted_val = net(torch.autograd.Variable(heads), torch.autograd.Variable(pos))
        max_score, idx = torch.max(predicted_val, 1)
        idx = idx.cpu().numpy()
        print(idx)
        img = cv2.imread(img_names[frame_id])
        for obj_id, pos_vec in enumerate(pos_ori):
            head = pos_vec[:4]
            obj = pos_vec[6:10]
            cv2.rectangle(img, (int(head[0]*img.shape[1]), int(head[1]*img.shape[0])),
                                      (int(head[2]*img.shape[1]), int(head[3]*img.shape[0])),
                                      (255, 0, 0), thickness=3)
            if idx[obj_id] == 0:
                cv2.rectangle(img, (int(obj[0] * img.shape[1]), int(obj[1] * img.shape[0])),
                              (int(obj[2] * img.shape[1]), int(obj[3] * img.shape[0])),
                              (255, 0, 0), thickness=3)
            else:
                cv2.rectangle(img, (int(obj[0] * img.shape[1]), int(obj[1] * img.shape[0])),
                              (int(obj[2] * img.shape[1]), int(obj[3] * img.shape[0])),
                              (0, 0, 255), thickness=3)
        cv2.imshow('img', img)
        cv2.waitKey(20)


def load_file():

    person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
    person_battery_bbox = '../3d_pose2gaze/record_bbox/'
    battery_gaze_path = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360_battery/'
    tracker_gaze_path = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360_tracker/'
    img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    bbox_path = {'tracker': person_tracker_bbox, 'battery': person_battery_bbox}
    gaze_path = {'tracker': tracker_gaze_path, 'battery': battery_gaze_path}
    cate_path = '/home/shuwen/data/data_preprocessing2/track_cate_with_frame/'
    mask_path = '/home/shuwen/Downloads/pointclouds/'
    obj_bbox_path = '/home/shuwen/data/data_preprocessing2/post_box_reid/'
    net = AttMat()
    net=load_best_checkpoint(net, '.')
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    clips = os.listdir(img_path)
    for clip in clips:
        print(clip)
        cate_file = os.path.join(cate_path, clip, clip + '.p')
        mask_names = sorted(glob.glob(os.path.join(mask_path, clip, '*.p')))
        img_names = sorted(glob.glob(os.path.join(img_path, clip, 'kinect/*.jpg')))
        obj_names = sorted(glob.glob(os.path.join(obj_bbox_path, clip, '*.p')))
        for key in ['tracker', 'battery']:
            print(key)
            with open(bbox_path[key] + clip + '.p', 'rb') as f:
                person_bbox = joblib.load(f)
            with open(gaze_path[key] + clip + '.p', 'rb') as f:
                gazes = pickle.load(f)
            top_n_objs_names_over_frames = extract_top_n_objs(cate_file, gazes, person_bbox, mask_names)
            attention_input = extract_attention_input(img_names, person_bbox, top_n_objs_names_over_frames, obj_names)
            get_attention(attention_input, net, img_names)




