"""
Created on Oct 8, 2018

Author: Lifeng Fan

Description:

"""
import utils
import scipy.misc
import numpy as np
import os
#from sets import Set

node_classes=['person','object']

pointing_index=[1,2,6,7,8,14,17,21,24,31,32,47,53,71,87,119,189,206,212,215,216,225,227,230,231,248]

# SmallAtt=['NA','single','mutual','avert','refer','follow','share']
# small_class={'NA':0,'single':1,'mutual':2,'avert':3,'refer':4,'follow':5,'share':6}

# BigAtt=['NA','SingleGaze','GazeFollow','AvertGaze','MutualGaze','JointAtt']
# big_class={'NA':0,'SingleGaze':1,'GazeFollow':2,'AvertGaze':3,'MutualGaze':4,'JointAtt':5}


BigAtt=['NA','SingleGaze','AvertGaze','MutualGaze','JointAtt']

big_class={'NA':0,'SingleGaze':1,'AvertGaze':2,'MutualGaze':3,'JointAtt':4,'GazeFollow':5}

#
# train_mean_value=[82.14003794, 69.78824366, 66.98968561]
# train_std_value=[58.98020917, 56.67555952, 59.52360103]

#
train_list=['1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108',
            '109', '11', '110', '111', '112', '113', '114', '115', '116', '117', '118',
            '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128',
            '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138',
            '139', '14', '140', '141', '142', '143', '144', '145', '146', '147', '148',
            '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158',
            '159', '16', '160', '161', '162', '163', '164', '165', '166', '167', '168',
            '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178',
            '179', '18', '180', '181', '182', '183', '184', '185', '186', '187', '188',
            '189', '19', '190', '191', '192', '193', '194', '195', '196', '197', '198',
            '199', '2', '20', '200', '201', '202', '203', '204', '205', '206', '207',
            '208', '209', '21', '210', '211', '212', '213', '214', '215', '216', '218',
            '219', '22', '220', '221', '223', '224', '225', '228', '229', '23', '231',
            '234', '235', '237', '24', '25', '26', '27', '28', '29', '30', '35', '37',
            '4', '43', '47', '50', '59', '66', '67', '68', '69']

val_list=['222', '226', '227', '3', '31', '32', '33', '34', '36', '38', '39', '40',
          '41', '42', '44', '45', '48', '49', '5', '51', '52', '53', '54', '55', '56',
          '57', '6', '60', '61', '70', '72', '73']

test_list=['217','230', '232', '233', '236', '46', '58', '62', '63', '64', '65', '7', '71',
           '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85',
           '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']


NA_Event={'name':'NA','count': [169, 98, 7, 5, 16, 20, 10, 128, 1, 431, 8, 5, 2, 24, 39, 156, 155, 474, 8, 2, 1, 6], 'vid': ['108', '111', '120', '123', '124', '141', '164', '196', '199', '231', '35', '37', '4', '47', '50', '59', '60', '61', '65', '8', '89', '98']}
Non_communicative_Event={'name':'Single','count': [605, 50, 42, 227, 66, 56, 301, 362, 541, 604, 322, 487, 376, 800, 555, 126, 282, 127, 950, 376, 347, 592, 420, 100, 54, 70, 240, 330, 280, 260, 748, 100, 60, 282, 223, 340, 130, 772, 360, 240, 436, 212, 152, 268, 196, 167, 62, 149, 18, 403, 116, 134, 190, 48, 250, 470, 474, 26, 456, 204, 577, 304, 180, 42, 1461, 430, 314, 216, 1628, 176, 112, 32, 142, 217, 34, 290, 168, 394, 341, 65, 10, 134, 316, 8, 86, 32, 554, 350, 260, 839, 606, 238, 179, 328, 432, 13, 97, 374, 344, 686, 319, 284, 288, 414, 92, 544, 371, 282, 328, 155, 665, 86, 844, 92, 272, 309, 82, 350, 89, 1142, 650, 812, 91, 636, 12, 89, 380, 236, 383, 757, 377, 209], 'vid': ['100', '102', '103', '104', '105', '107', '108', '110', '111', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '13', '131', '133', '134', '135', '136', '139', '141', '142', '145', '146', '148', '149', '15', '152', '157', '158', '159', '16', '162', '163', '164', '168', '169', '170', '171', '172', '174', '175', '177', '179', '180', '181', '182', '183', '185', '186', '188', '19', '190', '192', '193', '194', '195', '196', '201', '203', '208', '213', '215', '217', '218', '221', '223', '224', '229', '235', '237', '24', '25', '28', '29', '30', '32', '33', '34', '35', '36', '38', '39', '4', '43', '44', '45', '47', '48', '5', '50', '51', '52', '53', '54', '55', '59', '6', '60', '61', '65', '66', '7', '79', '8', '80', '83', '84', '86', '87', '88', '89', '9', '91', '92', '94', '95', '96', '97', '98', '99']}
Following_Event={'name':'Follow','count': [18, 166, 62, 32, 116, 578, 390, 198, 174, 336, 186, 298, 202, 301, 502, 426, 272, 298, 410, 445, 458, 200, 427, 526, 434, 140, 618, 292, 558, 832, 428, 249, 214, 508, 350, 108, 72, 402, 467, 318, 212, 124, 350, 308, 100, 128, 52, 76, 32, 28], 'vid': ['133', '144', '145', '175', '189', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '232', '233', '234', '235', '236', '237', '24', '29', '38', '5', '59', '91']}
Aversion_Event={'name':'Avert','count': [164, 226, 208, 144, 219, 60, 204, 1028, 676, 420, 166, 262, 428, 164, 190, 200, 50, 760, 600, 129, 132, 374, 1044, 942, 1104, 1346, 1126, 602, 228, 496, 455, 140, 134, 292, 304, 166, 259, 86, 108, 284], 'vid': ['107', '112', '114', '116', '118', '120', '121', '125', '127', '128', '142', '143', '153', '166', '176', '188', '192', '25', '26', '43', '50', '66', '67', '68', '69', '70', '72', '73', '74', '75', '78', '79', '80', '81', '82', '84', '85', '92', '96', '97']}
Mutual_Event={'name':'Mutual','count': [369, 208, 78, 26, 317, 150, 176, 432, 138, 58, 248, 92, 764, 632, 172, 100, 38, 32, 620, 590, 456, 288, 280, 180, 54, 106, 186, 544, 388, 386, 414, 262, 120, 120, 540, 446, 152, 148, 530, 72, 556, 120, 82, 424, 180, 730, 68, 5, 444, 410, 232, 22, 395, 302, 24, 138, 962, 682, 732, 680, 629, 740, 968, 48, 602, 544, 1710, 108, 132, 96, 630, 22, 302, 762, 306, 162, 882, 924, 76, 1246, 1180, 558, 421, 532, 26, 118, 326, 312, 496, 160, 720, 334, 310, 292, 226, 514, 532, 298, 106, 184, 192, 214, 250, 378, 198, 134, 148, 114, 310, 294, 128, 410], 'vid': ['103', '104', '105', '108', '110', '111', '113', '114', '115', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '13', '130', '131', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '146', '147', '149', '151', '152', '155', '156', '157', '159', '16', '160', '163', '164', '167', '168', '169', '170', '171', '174', '180', '181', '182', '183', '184', '185', '188', '190', '191', '192', '193', '194', '195', '196', '215', '217', '22', '229', '24', '25', '26', '27', '29', '31', '32', '34', '35', '36', '37', '4', '43', '45', '46', '47', '48', '54', '55', '57', '60', '61', '64', '66', '76', '77', '79', '8', '80', '86', '88', '89', '9', '92', '93', '94', '95', '96', '97', '98']}
Joint_Attention={'name':'Joint','count': [778, 930, 173, 362, 212, 62, 87, 128, 365, 420, 20, 376, 304, 782, 166, 1054, 842, 378, 200, 568, 360, 200, 444, 501, 344, 460, 286, 258, 2090, 84, 432, 430, 28, 150, 248, 460, 336, 134, 902, 1356, 838, 556, 204, 144, 914, 2024, 394, 1052, 864, 556, 264, 184, 36, 146, 234, 381, 1314, 208, 203, 110, 178, 1004, 368, 239, 806, 199, 159, 190, 306, 608, 298, 482, 220, 336, 348, 190, 394, 514, 631, 882, 184, 984, 36, 370, 3360, 256, 158, 484, 312, 362, 1090, 202, 623, 432, 532, 426, 80, 248, 230, 94, 51, 34, 432, 257, 225], 'vid': ['1', '10', '100', '101', '102', '103', '104', '106', '109', '11', '115', '116', '118', '12', '122', '129', '13', '132', '136', '14', '147', '15', '150', '154', '156', '16', '161', '165', '17', '170', '172', '173', '175', '177', '178', '18', '181', '183', '186', '187', '189', '19', '190', '196', '197', '2', '20', '206', '21', '211', '214', '215', '221', '224', '225', '228', '23', '234', '235', '24', '25', '27', '28', '29', '3', '30', '32', '33', '38', '39', '40', '41', '42', '43', '44', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '61', '62', '63', '64', '65', '7', '71', '8', '89', '9', '90', '91', '92', '93', '94', '97', '99']}


train_dict={'Mutual': ['103', '104', '105', '108', '110', '111', '113', '114', '115', '117',
                       '118', '119', '12', '120', '121', '122', '123', '124', '125', '126',
                       '13', '130', '131', '133', '134', '135', '136', '137', '138', '139',
                       '140', '141', '142', '143', '144', '146', '147', '149', '151', '152',
                       '155', '156', '157', '159', '16', '160', '163', '164', '167', '168',
                       '169', '170', '171', '174', '180', '181', '182', '183', '184', '185',
                       '188', '190', '191', '192', '193', '194', '195', '196', '215', '22',
                       '229', '24', '25', '26', '27', '29', '35', '37', '4', '43', '47', '66'],
            'Avert': ['107', '112', '114', '116', '118', '120', '121', '125', '127', '128',
                      '142', '143', '153', '166', '176', '188', '192', '25', '26', '43', '50',
                      '66', '67', '68', '69'],
            'NA': ['108', '111', '120', '123', '124', '141', '164', '196', '199', '231', '35',
                   '37', '4', '47', '50', '59'],
            'Joint': ['1', '10', '100', '101', '102', '103', '104', '106', '109', '11',
                      '115', '116', '118', '12', '122', '129', '13', '132', '136', '14',
                      '147', '15', '150', '154', '156', '16', '161', '165', '17', '170',
                      '172', '173', '175', '177', '178', '18', '181', '183', '186', '187',
                      '189', '19', '190', '196', '197', '2', '20', '206', '21', '211',
                      '214', '215', '221', '224', '225', '228', '23', '234', '235',
                      '24', '25', '27', '28', '29', '30', '43', '50', '59'],
            'Single': ['100', '102', '103', '104', '105', '107', '108', '110', '111',
                       '113', '114', '115', '116', '117', '118', '119', '12', '120',
                       '121', '122', '123', '124', '125', '126', '13', '131', '133', '134',
                       '135', '136', '139', '141', '142', '145', '146', '148', '149', '15',
                       '152', '157', '158', '159', '16', '162', '163', '164', '168', '169',
                       '170', '171', '172', '174', '175', '177', '179', '180', '181', '182',
                       '183', '185', '186', '188', '19', '190', '192', '193', '194', '195',
                       '196', '201', '203', '208', '213', '215', '218', '221', '223', '224',
                       '229', '235', '237', '24', '25', '28', '29', '30', '35', '4', '43',
                       '47', '50', '59', '66'],
            'Follow': ['133', '144', '145', '175', '189', '198', '199', '200', '201',
                       '202', '203', '204', '205', '206', '207', '208', '209', '210',
                       '211', '212', '213', '214', '215', '216', '218', '219', '220',
                       '221', '223', '224', '225', '228', '229', '234', '235', '237', '24', '29', '59']}



val_dict={'Mutual': ['31', '32', '34', '36', '45', '48', '54', '55', '57', '60', '61'],
          'Avert': ['70', '72', '73'],
          'NA': ['60', '61'],
          'Joint': ['3', '32', '33', '38', '39', '40', '41', '42', '44', '48', '49', '51',
                    '52', '53', '54', '55', '56', '57', '6', '61'],
          'Single': ['32', '33', '34', '36', '38', '39', '44', '45', '48', '5', '51',
                     '52', '53', '54', '55', '6', '60', '61'],
          'Follow': ['222', '226', '227', '38', '5']}



test_dict={'Mutual': ['217', '46', '64', '76', '77', '79', '8', '80', '86', '88', '89',
                      '9', '92', '93', '94', '95', '96', '97', '98'],
           'Avert': ['74', '75', '78', '79', '80', '81', '82', '84', '85', '92', '96', '97'],
           'NA': ['65', '8', '89', '98'],
           'Joint': ['58', '62', '63', '64', '65', '7', '71', '8', '89', '9', '90', '91',
                     '92', '93', '94', '97', '99'],
           'Single': ['217', '65', '7', '79', '8', '80', '83', '84', '86', '87', '88',
                      '89', '9', '91', '92', '94', '95', '96', '97', '98', '99'],
           'Follow': ['217', '230', '232', '233', '236', '91']}

#
valtest_dict={

          'Mutual': ['31', '32', '34', '36', '45', '48', '54', '55', '57', '60', '61','217', '46', '64', '76', '77', '79', '8', '80', '86', '88', '89',
                      '9', '92', '93', '94', '95', '96', '97', '98'],
          'Avert': ['70', '72', '73','74', '75', '78', '79', '80', '81', '82', '84', '85', '92', '96', '97'],
          'NA': ['60', '61', '65', '8', '89', '98'],
          'Joint': ['3', '32', '33', '38', '39', '40', '41', '42', '44', '48', '49', '51',
                    '52', '53', '54', '55', '56', '57', '6', '61','58', '62', '63', '64', '65', '7', '71', '8', '89', '9', '90', '91',
                     '92', '93', '94', '97', '99'],
          'Single': ['32', '33', '34', '36', '38', '39', '44', '45', '48', '5', '51',
                     '52', '53', '54', '55', '6', '60', '61','217', '65', '7', '79', '8', '80', '83', '84', '86', '87', '88',
                      '89', '9', '91', '92', '94', '95', '96', '97', '98', '99'],
          'Follow': ['222', '226', '227', '38', '5','217', '230', '232', '233', '236', '91']
}



def dataset_split_new():

    train_dict={'Single':list(),'Avert':list(),'Mutual':list(),'Joint':list()}
    val_dict={'Single':list(),'Avert':list(),'Mutual':list(),'Joint':list()}
    test_dict={'Single':list(),'Avert':list(),'Mutual':list(),'Joint':list()}


    for event in [NA_Event, Non_communicative_Event, Following_Event, Aversion_Event, Mutual_Event, Joint_Attention]:

        for vid in event['vid']:

            if vid in train_list:

                train_dict[event['name']].append(vid)
            elif vid in val_list:
                val_dict[event['name']].append(vid)

            elif vid in test_list:
                test_dict[event['name']].append(vid)



    print(train_dict)

    print(val_dict)

    print(test_dict)



def dataset_split_clean():

    train_set=[]
    val_set=[]
    test_set=[]

    for event in [NA_Event, Non_communicative_Event, Following_Event, Aversion_Event, Mutual_Event, Joint_Attention]:

        train_set.extend(event['train'])
        val_set.extend(event['val'])
        test_set.extend(event['test'])

    train_set=set(train_set)
    val_set=set(val_set)
    test_set=set(test_set)

    new_val_set=val_set.difference(train_set)
    new_test_set=test_set.difference(train_set).difference(val_set)


    return train_set, new_val_set, new_test_set



def dataset_split(event):

    cnt=0

    event['train'] = []
    event['val'] = []
    event['test'] = []

    for idx in range(len(event['vid'])):

        if cnt< event['total_node_num']*0.6:
           cnt+=event['count'][idx]
           event['train'].append(event['vid'][idx])
        elif cnt> event['total_node_num']*0.6 and cnt < event['total_node_num']*0.8:
            cnt += event['count'][idx]
            event['val'].append(event['vid'][idx])
        else:
            cnt += event['count'][idx]
            event['test'].append(event['vid'][idx])

    return event


def get_event_per_vid(path):

    single = dict(vid=list(),count=list())
    mutual = dict(vid=list(), count=list())
    #follow = dict(vid=list(), count=list())
    avert = dict(vid=list(), count=list())
    joint = dict(vid=list(), count=list())
    #na = dict(vid=list(), count=list())

    files=[f for f in sorted(os.listdir(os.path.join(path.data_root,'all','ant'))) if os.path.isfile(os.path.join(path.data_root,'all','ant',f))]

    for ind in range(len(files)):

        vid=files[ind].strip('.txt').strip('NewAnt_')

        with open(os.path.join(path.data_root,'all','ant',files[ind]),'r') as f:
            lines=f.readlines()

        for lind in range(len(lines)):
            list_now=lines[lind].strip().split(' ')

            if len(list_now)==10:
                continue

            event=list_now[10].strip()

            # if event=='NA':
            #     if vid in na['vid']:
            #         na['count'][-1]+=1
            #     else:
            #         na['vid'].append(vid)
            #         na['count'].append(1)

            if event=='SingleGaze':
                if vid in single['vid']:
                    single['count'][-1]+=1
                else:
                    non_comm['vid'].append(vid)
                    non_comm['count'].append(1)

            elif event=='GazeFollow':
                if vid in follow['vid']:
                    follow['count'][-1]+=1
                else:
                    follow['vid'].append(vid)
                    follow['count'].append(1)

            elif event=='AvertGaze':

                if vid in avert['vid']:
                    avert['count'][-1]+=1
                else:
                    avert['vid'].append(vid)
                    avert['count'].append(1)

            elif event=='MutualGaze':

                if vid in mutual['vid']:
                    mutual['count'][-1]+=1
                else:
                    mutual['vid'].append(vid)
                    mutual['count'].append(1)

            elif event=='JointAtt':

                if vid in joint['vid']:
                    joint['count'][-1]+=1
                else:
                    joint['vid'].append(vid)
                    joint['count'].append(1)


    return na, non_comm,follow, avert, mutual, joint



def get_normalization_stats(paths,step):

    with open(paths.train_img_list_file,'r') as to_read:

        lines=to_read.readlines()

    for idx in range(0,len(lines),step):

        img=scipy.misc.imread(lines[idx].strip(),mode='RGB')

        img=np.reshape(img,[-1,3])
        try:
             img_all=np.concatenate((img_all,img),axis=0)

        except:
            img_all=img


    mean_value=img_all.mean(axis=0)
    std_value=img_all.std(axis=0)

    return mean_value, std_value


def get_node_num_stats():

    nodes_num={}

    path='/home/lfan/Dropbox/Projects/ICCV19/RunComm/data/ant_processed/'

    files=[f for f in os.listdir(path)]

    for i in range(len(files)):
        f=files[i]

        video=np.load(os.path.join(path,f))

        for j in range(len(video)):

            n_num=len(video[j]['ant'])

            if str(n_num) in nodes_num:
                nodes_num[str(n_num)]+=1
            else:
                nodes_num[str(n_num)]=1


    print(nodes_num)

#   {'1': 2092, '3': 34720, '2': 20804, '5': 4461, '4': 11397, '7': 80, '6': 1678, '8': 502}


def main():

    # # to get the mean and std statistics of the training images
    # paths=utils.Paths()
    #
    # mean_value,std_value=get_normalization_stats(paths,step=20)
    #
    # print(mean_value)
    # print(std_value)

    #
    path = utils.Paths()
    # na, non_comm, follow, avert, mutual, joint=get_event_per_vid(path)
    #
    # print('NA Event: {}'.format(na))
    # print('Non-communicative Event: {}'.format(non_comm))
    # print('Following Event: {}'.format(follow))
    # print('Aversion Event: {}'.format(avert))
    # print('Mutual Event: {}'.format(mutual))
    # print('Joint Attention: {}'.format(joint))

    # new_NA_event=train_test_split(NA_Event)
    # new_noncomm_event = train_test_split(Non_communicative_Event)
    # new_follow_event = train_test_split(Following_Event)
    # new_avert_event = train_test_split(Aversion_Event)
    # new_mutual_event = train_test_split(Mutual_Event)
    # new_jointatt_event = train_test_split(Joint_Attention)
    #
    # print('NA Event: {}'.format(new_NA_event))
    # print('Non-communicative Event: {}'.format(new_noncomm_event))
    # print('Following Event: {}'.format(new_follow_event))
    # print('Aversion Event: {}'.format(new_avert_event))
    # print('Mutual Event: {}'.format(new_mutual_event))
    # print('Joint Attention: {}'.format(new_jointatt_event))

    # print('NA Event ==> total video number: {}, total node number: {}'.format(len(NA_Event['vid']), sum(NA_Event['count'])))
    # print('Non-communicative Event ==> total video number: {}, total node number: {}'.format(len(Non_communicative_Event['vid']), sum(Non_communicative_Event['count'])))
    # print('Following Event  ==> total video number: {}, total node number: {}'.format(len(Following_Event['vid']), sum(Following_Event['count'])))
    # print('Aversion Event ==> total video number: {}, total node number: {}'.format(len(Aversion_Event['vid']), sum(Aversion_Event['count'])))
    # print('Mutual Event ==> total video number: {}, total node number: {}'.format(len(Mutual_Event['vid']), sum(Mutual_Event['count'])))
    # print('Joint Attention ==> total video number: {}, total node number: {}'.format(len(Joint_Attention['vid']), sum(Joint_Attention['count'])))


    # for event in [NA_Event, Non_communicative_Event, Following_Event, Aversion_Event, Mutual_Event, Joint_Attention]:
    #
    #     new_event=dataset_split(event)
    #
    #     print('[{}],\n  train vid: {} \n val vid: {} \n test vid: {}'.format(new_event['name'], new_event['train'], new_event['val'], new_event['test']))




    # train_set, val_set, test_set=dataset_split_clean()
    #
    # print('training set: {}'.format(train_set))
    # print('validation set: {}'.format(val_set))
    # print('test set: {}'.format(test_set))
    #
    # print(len(train_set)+len(val_set)+len(test_set))
    #
    # print(len(train_set))
    #
    # print(len(val_set))
    #
    # print(len(test_set))
    #
    # train_list=sorted(list(train_set))
    # val_list=sorted(list(val_set))
    # test_list=sorted(list(test_set))
    #
    # print(len(train_list))
    #
    # print(len(val_list))
    #
    # print(len(test_list))
    # print(len(train_list)+len(val_list)+len(test_list))
    #
    # print(set(train_list).intersection(set(val_list)))
    # print(set(train_list).intersection(set(test_list)))
    # print(set(val_list).intersection(set(test_list)))

    # files=[f for f in os.listdir('/home/lfan/Dropbox/RunComm/data/all/split') if os.path.isfile(os.path.join('/home/lfan/Dropbox/RunComm/data/all/split',f))]
    #
    # for idx in range(len(files)):
    #
    #     vid=files[idx].strip('NewAnt_').strip('.txt')
    #
    #     if vid in train_list:
    #
    #         os.system('mv /home/lfan/Dropbox/RunComm/data/all/split/{} /home/lfan/Dropbox/RunComm/data/all/split/train'.format(files[idx]))
    #
    #     elif vid in test_list:
    #
    #         os.system('mv /home/lfan/Dropbox/RunComm/data/all/split/{} /home/lfan/Dropbox/RunComm/data/all/split/test'.format(files[idx]))
    #
    #     elif vid in val_list:
    #
    #         os.system('mv /home/lfan/Dropbox/RunComm/data/all/split/{} /home/lfan/Dropbox/RunComm/data/all/split/validate'.format(files[idx]))

    #
    # for event in [NA_Event, Non_communicative_Event, Following_Event, Aversion_Event, Mutual_Event, Joint_Attention]:
    #
    #     event['train']=list(set(event['train']).union(set(train_list)))
    #     event['val'] = list(set(event['val']).union(set(val_list)))
    #     event['test'] = list(set(event['test']).union(set(test_list)))
    #
    #     print(event)

    # train_len=0
    # val_len = 0
    # test_len = 0
    #
    # for event in [NA_Event, Non_communicative_Event, Following_Event, Aversion_Event, Mutual_Event, Joint_Attention]:
    #
    #     train_len+=len(event['train'])
    #     val_len += len(event['val'])
    #     test_len += len(event['test'])
    #
    # print('train total video {}'.format(train_len))
    # print('validate total video {}'.format(val_len))
    # print('test total video {}'.format(test_len))


    #dataset_split_new()
    # train=list()
    # val=list()
    # test=list()
    #
    # for name in ['NA','Single','Follow','Avert','Mutual','Joint']:
    #     train.extend(train_dict[name])
    #     val.extend(val_dict[name])
    #     test.extend(test_dict[name])
    #
    # print(len(set(train).intersection(set(train_list))))
    # print(len(set(train)))
    #
    # print(len(set(val).intersection(set(val_list))))
    # print(len(set(val)))
    #
    # print(len(set(test).intersection(set(test_list))))
    # print(len(set(test)))

    pass

    import os.path as op
    from os import listdir
    import matplotlib.pyplot as plt

    data_root='/home/lfan/Dropbox/Projects/ICCV19/DATA/'

    # files=[f for f in listdir(op.join(data_root, 'img'))]
    #
    # video_len=[]
    #
    # for i in range(len(files)):
    #
    #     video=files[i]
    #
    #     imgs=[f for f in listdir(op.join(data_root,'img',video))]
    #
    #     video_len.append(len(imgs))
    #
    # print('min {} max {}'.format(min(video_len), max(video_len)))
    #
    # print('mean {}'.format(sum(video_len)/len(video_len)))
    # plt.hist(video_len)
    # plt.show()

#------------------------------------------------
# object num
    #['SingleGaze', 'GazeFollow', 'AvertGaze', 'MutualGaze', 'JointAtt']

    files=[f for f in listdir(op.join(data_root,'ant_processed'))]
    #object_cnt=0
    Single_cnt=0
    Mutual_cnt=0
    Avert_cnt=0
    Follow_cnt=0
    Joint_cnt=0

    total_cnt=0

    for i in range(len(files)):

        video=np.load(op.join(data_root,'ant_processed', files[i]))

        for j in range(len(video)):

            ant=video[j]['ant']
            #frame_cnt+=1

            for k in range(len(ant)):

                if ant[k]['BigAtt']=='SingleGaze':
                    Single_cnt+=1
                    total_cnt+=1
                elif ant[k]['BigAtt']=='MutualGaze':
                    Mutual_cnt+=1
                    total_cnt+=1
                elif ant[k]['BigAtt']=='AvertGaze':
                    Avert_cnt+=1
                    total_cnt+=1
                elif ant[k]['BigAtt']=='GazeFollow':
                    Follow_cnt+=1
                    total_cnt+=1
                elif ant[k]['BigAtt']=='JointAtt':
                    Joint_cnt+=1
                    total_cnt+=1

    print('SingleGaze {}'.format(Single_cnt*1./total_cnt))
    print('MutualGaze {}'.format(Mutual_cnt * 1. / total_cnt))
    print('AvertGaze {}'.format(Avert_cnt * 1. / total_cnt))
    print('GazeFollow {}'.format(Follow_cnt * 1. / total_cnt))
    print('JointAtt {}'.format(Joint_cnt * 1. / total_cnt))

    # print(object_cnt)
    # print(object_cnt*1./frame_cnt)
    #

if __name__ == '__main__':

    #get_node_num_stats()

    main()


