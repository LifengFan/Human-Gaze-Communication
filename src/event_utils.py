import numpy as np
def get_metric_from_confmat(confmat):

    N=5

    recall=np.zeros(N)
    precision=np.zeros(N)
    F_score=np.zeros(N)

    correct_cnt=0.
    total_cnt=0.

    for i in range(N):

        recall[i]=confmat[i,i]/(np.sum(confmat[i,:])+1e-7)

        precision[i]=confmat[i,i]/(np.sum(confmat[:,i])+1e-7)

        F_score[i]=2*precision[i]*recall[i]/(precision[i]+recall[i]+1e-7)

        correct_cnt+=confmat[i,i]

        total_cnt+=np.sum(confmat[i,:])

    acc=correct_cnt/total_cnt

    print('===> Confusion Matrix for Event Label: \n {}'.format(confmat.astype(int)))

    print('===> Precision: \n  [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(precision[0]*100, precision[1]*100, precision[2]*100, precision[3]*100, precision[4]*100))

    print('===> Recall: \n [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(recall[0]*100, recall[1]*100, recall[2]*100, recall[3]*100, recall[4]*100))

    print('===> F score: \n [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(F_score[0]*100, F_score[1]*100, F_score[2]*100, F_score[3]*100, F_score[4]*100))

    print('===> Accuracy: {} %'.format(acc*100))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


confmat=np.zeros((5,5))
total_acc_top1= AverageMeter()
total_acc_top2=AverageMeter()


