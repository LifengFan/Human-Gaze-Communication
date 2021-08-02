import numpy as np
import os


class Paths(object):
    def __init__(self):
        self.project_root='/home/lfan/Dropbox/Projects/ICCV19/RunComm/'
        self.log_root=os.path.join(self.project_root,'log')
        self.data_root=os.path.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/')
        self.tmp_root = os.path.join(self.project_root, 'tmp')



def get_label_statistics(path, mode):

    # the last position records the total number

    NA_event=np.zeros(8)
    SingleGaze=np.zeros(8)
    MutualGaze=np.zeros(8)
    GazeFollow=np.zeros(8)
    AvertGaze=np.zeros(8)
    JointAtt=np.zeros(8)

    ant_files=sorted(os.listdir(os.path.join(path.data_root,mode,'ant')))

    for ind in range(len(ant_files)):
        file=ant_files[ind]

        with open(os.path.join(path.data_root,mode,'ant',file),'r') as to_read:
            lines=to_read.readlines()



        for line_ind in range(len(lines)):
            tmp=lines[line_ind].split(' ')

            if len(tmp)==13:
                big_label=tmp[10].strip()
                small_label=tmp[11].strip()

                if big_label=='NA':
                    NA_event[7]+=1

                    if small_label=='NA':
                        NA_event[0]+=1
                    elif small_label=='single':
                        NA_event[1]+=1
                    elif small_label=='mutual':
                        NA_event[2]+=1
                    elif small_label=='avert':
                        NA_event[3]+=1
                    elif small_label=='refer':
                        NA_event[4]+=1
                    elif small_label=='follow':
                        NA_event[5]+=1
                    elif small_label=='share':
                        NA_event[6]+=1

                elif big_label=='SingleGaze':

                    SingleGaze[7]+=1

                    if small_label=='NA':
                        SingleGaze[0]+=1
                    elif small_label=='single':
                        SingleGaze[1]+=1
                    elif small_label=='mutual':
                        SingleGaze[2]+=1
                    elif small_label=='avert':
                        SingleGaze[3]+=1
                    elif small_label=='refer':
                        SingleGaze[4]+=1
                    elif small_label=='follow':
                        SingleGaze[5]+=1
                    elif small_label=='share':
                        SingleGaze[6]+=1

                elif big_label=='GazeFollow':

                    GazeFollow[7]+=1

                    if small_label == 'NA':
                        GazeFollow[0] += 1
                    elif small_label == 'single':
                        GazeFollow[1] += 1
                    elif small_label == 'mutual':
                        GazeFollow[2] += 1
                    elif small_label == 'avert':
                        GazeFollow[3] += 1
                    elif small_label == 'refer':
                        GazeFollow[4] += 1
                    elif small_label == 'follow':
                        GazeFollow[5] += 1
                    elif small_label == 'share':
                        GazeFollow[6] += 1

                elif big_label=='AvertGaze':

                    AvertGaze[7]+=1

                    if small_label == 'NA':
                        AvertGaze[0] += 1
                    elif small_label == 'single':
                        AvertGaze[1] += 1
                    elif small_label == 'mutual':
                        AvertGaze[2] += 1
                    elif small_label == 'avert':
                        AvertGaze[3] += 1
                    elif small_label == 'refer':
                        AvertGaze[4] += 1
                    elif small_label == 'follow':
                        AvertGaze[5] += 1
                    elif small_label == 'share':
                        AvertGaze[6] += 1

                elif big_label=='MutualGaze':

                    MutualGaze[7]+=1

                    if small_label == 'NA':
                        MutualGaze[0] += 1
                    elif small_label == 'single':
                        MutualGaze[1] += 1
                    elif small_label == 'mutual':
                        MutualGaze[2] += 1
                    elif small_label == 'avert':
                        MutualGaze[3] += 1
                    elif small_label == 'refer':
                        MutualGaze[4] += 1
                    elif small_label == 'follow':
                        MutualGaze[5] += 1
                    elif small_label == 'share':
                        MutualGaze[6] += 1

                elif big_label=='JointAtt':

                    JointAtt[7]+=1

                    if small_label == 'NA':
                        JointAtt[0] += 1
                    elif small_label == 'single':
                        JointAtt[1] += 1
                    elif small_label == 'mutual':
                        JointAtt[2] += 1
                    elif small_label == 'avert':
                        JointAtt[3] += 1
                    elif small_label == 'refer':
                        JointAtt[4] += 1
                    elif small_label == 'follow':
                        JointAtt[5] += 1
                    elif small_label == 'share':
                        JointAtt[6] += 1


    return  NA_event, SingleGaze, MutualGaze, GazeFollow, AvertGaze, JointAtt


def main():

    paths = Paths()

    mode='train'

    train_NA_event, train_SingleGaze, train_MutualGaze, train_GazeFollow, train_AvertGaze, train_JointAtt=get_label_statistics(paths,mode)

    print('[Training statistics] \n NA: {} \n SingleGaze: {} \n MutualGaze: {} \n GazeFollow: {} \n AvertGaze: {} \n JointAtt: {} \n small: [NA, single, mutual, avert, refer, follow, share]'.format(train_NA_event, train_SingleGaze, train_MutualGaze, train_GazeFollow, train_AvertGaze, train_JointAtt))

    mode = 'validate'

    val_NA_event, val_SingleGaze, val_MutualGaze, val_GazeFollow, val_AvertGaze, val_JointAtt = get_label_statistics(paths, mode)

    print('[Validation statistics] \n NA: {} \n SingleGaze: {} \n MutualGaze: {} \n GazeFollow: {} \n AvertGaze: {} \n JointAtt: {} \n small: [NA, single, mutual, avert, refer, follow, share]'.format(
        val_NA_event, val_SingleGaze, val_MutualGaze, val_GazeFollow, val_AvertGaze, val_JointAtt))


    mode = 'test'

    test_NA_event, test_SingleGaze, test_MutualGaze, test_GazeFollow, test_AvertGaze, test_JointAtt = get_label_statistics(paths, mode)

    print('[Testing statistics] \n NA: {} \n SingleGaze: {} \n MutualGaze: {} \n GazeFollow: {} \n AvertGaze: {} \n JointAtt: {} \n small: [NA, single, mutual, avert, refer, follow, share]'.format(
        test_NA_event, test_SingleGaze, test_MutualGaze, test_GazeFollow, test_AvertGaze, test_JointAtt))



    pass

if __name__=='__main__':

    main()