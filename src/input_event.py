import pickle
import numpy as np
atomic_action = ['avert', 'mutual', 'follow', 'single', 'share', 'refer', 'NA']

with open('clean_event_seq.pkl', 'rb') as f:
    data = pickle.load(f)
f.close()

sequence_list = []
for key, item in data.items():
    for sequence in item:
        sequence_label = []
        sequence_len = []
        item_pre = []
        for i, sub_act in enumerate(sequence[1]):
            if item_pre != sub_act:
                sequence_len.append(0)
                sequence_label.append(atomic_action.index(sub_act))
            else:
                sequence_len[-1] += 1
            item_pre = sub_act
        print len(sequence_label)
        sequence_list.append({'label': key, 'data': sequence_label, 'len': sequence_len})

len_total = len(sequence_list)

train_num = int(len_total * 0.5)
ind = np.random.permutation(len_total)
train_ind = ind[:train_num]
test_ind = ind[train_num:]

with open('train_seq.pickle', 'w') as f:
    pickle.dump([sequence_list[index] for index in train_ind], f)
f.close()

with open('test_seq.pickle', 'w') as f:
    pickle.dump([sequence_list[index] for index in test_ind], f)
f.close()

with open('pred_res_for_atomic_in_event.pkl', 'rb') as f:
    data = pickle.load(f)
f.close()

sequence_list_without_gt = []
for key, item in data.items():
    for sequence in item:
        sequence_label = []
        sequence_len = []
        item_pre = []
        for i, sub_act in enumerate(sequence[1]):
            if item_pre != sub_act:
                sequence_len.append(0)
                sequence_label.append(atomic_action.index(sub_act))
            else:
                sequence_len[-1] += 1
            item_pre = sub_act
        print len(sequence_label)
        sequence_list_without_gt.append({'label': key, 'data': sequence_label, 'len': sequence_len})

len_total = len(sequence_list_without_gt)

with open('test_seq_without_gt.pickle', 'w') as f:
    pickle.dump([sequence_list_without_gt[index] for index in test_ind], f)
f.close()










