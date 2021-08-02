import sys
#sys.path.append("/home/lfan/Dropbox/RunComm/src/")
sys.path.append("/home/tangxy/RunComm/src")

import os
import torch
import torch.nn
import torch.autograd
import units
import numpy as np


class Atomic_HGNN(torch.nn.Module):

    def __init__(self,model_args, args):
        super(Atomic_HGNN, self).__init__()

        self.model_args=model_args.copy()

        self.resnet = units.Resnet.ResNet50(6)

        self._load_pretrained_weight(os.path.join(args.tmp_root, 'checkpoints', 'attmat', 'model_best.pth'))
        self.freeze_res_layer(layer_num=9)
        
        num_ftrs = self.resnet.resnet.fc.in_features
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 64*3)



        self.conv3d1=torch.nn.Conv3d(in_channels=64*3, out_channels=6,kernel_size=(5, 6,1),stride=1, padding=0) #Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3d2=torch.nn.Conv3d(in_channels=6,out_channels=6,kernel_size=(1,1,1), stride=1, padding=0)

        self.link_fun=units.LinkFunction('GraphConv',model_args)
        self.message_fun=units.MessageFunction('linear_concat_relu',model_args)
        self.update_fun=units.UpdateFunction('gru',model_args)
        self.readout_fun=units.ReadoutFunction('fc', {'readout_input_size':model_args['node_feature_size'],'output_classes':model_args['big_attr_classes']})

        self.lstm=torch.nn.LSTM(model_args['roi_feature_size'], self.model_args['lstm_hidden_size'])
        self.lstm_readout = torch.nn.Linear(self.model_args['lstm_hidden_size'], self.model_args['big_attr_classes'])
        self.propagate_layers=model_args['propagate_layers']

        self.softmax=torch.nn.Softmax()
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.maxpool=torch.nn.MaxPool2d((2,2))

    def _load_pretrained_weight(self, model_path):

        pretrained_model = torch.load(model_path)['state_dict']
        #---------------------------------------------------------------------
        # load resnet weight
        model_dict=self.resnet.state_dict()
        pretrained_dict={}

        for k,v in pretrained_model.items():
            if k[len('module.resnet.'):] in model_dict:
                pretrained_dict[k[len('module.resnet.'):]]=v

        # print(len(model_dict))
        # print(len(pretrained_dict))

        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

    def freeze_res_layer(self, layer_num=9):

        child_cnt=0
        for child in self.resnet.resnet.children():
            if child_cnt<layer_num:
                for param in child.parameters():
                    param.requires_grad=False

            child_cnt+=1

        print('Resnet has {} children totally, {} has been freezed'.format(child_cnt, layer_num))


    def forward(self,node_feature, pos, ATTMAT, args):
        #node_feature [N,seq,node_num,3,224,224]
        #edge_feature [N,seq,node_num,node_num,3,224,224]
        #node_num_rec=[N, seq]

        batch_size=node_feature.size()[0]
        seq_len=node_feature.size()[1]
        max_node_num = 6
#        valid_node_num = node_feature.size()[2]
        valid_node_num = ATTMAT.size()[2]

        node_resnet=torch.autograd.Variable(torch.zeros(batch_size,seq_len, 64*3,max_node_num))
#        edge_resnet=torch.autograd.Variable(torch.zeros(batch_size,seq_len,self.model_args['roi_feature_size'],max_node_num,max_node_num))
        edge_resnet = torch.autograd.Variable(torch.zeros(batch_size, seq_len, 12, max_node_num, max_node_num))

        if hasattr(args,'cuda') and args.cuda:
            node_resnet=node_resnet.cuda()
            edge_resnet=edge_resnet.cuda()


        for b_idx in range(batch_size):
            for sq_idx in range(seq_len):
                for h_i in range(2):
                    node_resnet[b_idx, sq_idx, :, h_i]= self.resnet(node_feature[b_idx, sq_idx,h_i, ...].unsqueeze(0))
#                valid_node_num = node_num_rec[b_idx,sq_idx]
#                for n_idx in range(valid_node_num):
#                    node_resnet[b_idx, sq_idx, :, n_idx] = self.resnet(node_feature[b_idx, sq_idx,n_idx, ...].unsqueeze(0))


        for b_idx in range(batch_size):
            for sq_idx in range(seq_len):
#                valid_node_num = max_node_num
#                valid_node_num = node_num_rec[b_idx,sq_idx]
                for n_idx1 in range(valid_node_num):
                    for n_idx2 in range(valid_node_num):
                        if n_idx2 == n_idx1:
                            continue
                        if ATTMAT[b_idx, sq_idx,n_idx1,n_idx2] ==1:
                            edge_resnet[b_idx, sq_idx, :, n_idx1, n_idx2] = torch.cat((pos[b_idx,sq_idx,n_idx1,:], pos[b_idx, sq_idx, n_idx2, :]), -1)


        hidden_node_states = [[[node_resnet[batch_i,sq_i, ...].unsqueeze(0).clone() for _ in range(self.propagate_layers + 1)] for sq_i in range(seq_len) ] for batch_i in range(batch_size)]
        hidden_edge_states = [[[edge_resnet[batch_i,sq_i, ...].unsqueeze(0).clone() for _ in range(self.propagate_layers + 1)] for sq_i in range(seq_len) ] for batch_i in range(batch_size)]

        pred_feature = torch.autograd.Variable(torch.zeros(batch_size, 64*3, 5, max_node_num, 1))
        pred_adj_mat = torch.autograd.Variable(torch.zeros(batch_size, seq_len,max_node_num, max_node_num))
#        pred_label = torch.autograd.Variable(torch.zeros(batch_size,6))

        if hasattr(args, 'cuda') and args.cuda:
            pred_adj_mat = pred_adj_mat.cuda()
            pred_feature = pred_feature.cuda()
#            pred_label = pred_label.cuda()
            # hidden_node_states=torch.autograd.Variable(torch.FloatTensor(hidden_node_states)).cuda()
            # hidden_edge_states=torch.autograd.Variable(torch.FloatTensor(hidden_edge_states)).cuda()


        #msg passing
        # actually, batch size here is only 1
        for batch_idx in range(batch_size):

            for passing_round in range(self.propagate_layers):

                for sq_idx in range(seq_len):
#                    valid_node_num = max_node_num
#                    valid_node_num = node_num_rec[batch_idx,sq_idx]


                    pred_adj_mat[batch_idx,sq_idx, :valid_node_num, :valid_node_num] = self.link_fun(hidden_edge_states[batch_idx][sq_idx][passing_round][:, :, :valid_node_num, :valid_node_num])

                    sigmoid_pred_adj_mat = self.sigmoid(pred_adj_mat[batch_idx, sq_idx, :valid_node_num, :valid_node_num].unsqueeze(0))

                    # Loop through nodes


                    for i_node in range(valid_node_num):

                        h_v = hidden_node_states[batch_idx][sq_idx][passing_round][:, :, i_node]
                        h_w = hidden_node_states[batch_idx][sq_idx][passing_round][:, :, :valid_node_num]
                        #e_vw = node_resnet=torch.autograd.Variable(torch.zeros(batch_size,seq_len, 64*3,max_node_num))
                        e_vw = edge_resnet[batch_idx,sq_idx, :, i_node, :valid_node_num].unsqueeze(0)
                        #e_vw.expand_as(h_w)

                        m_v = self.message_fun(h_w, e_vw, args)

                        # sum up messages from different nodes according to weights

                        # m_v [1,message_size,node_num]
                        # sigmoidi_ored_adj_mat [1,node_num,node_num]

                        m_v = sigmoid_pred_adj_mat[:, i_node, :valid_node_num].unsqueeze(1).expand_as(m_v) * m_v
                        hidden_edge_states[batch_idx][sq_idx][passing_round + 1][:, :, i_node, :valid_node_num] = m_v
                        m_v = torch.sum(m_v, 2)
                        h_v = self.update_fun(m_v[None], h_v[None].contiguous())

                        hidden_node_states[batch_idx][sq_idx][passing_round+1][:,:,i_node]=h_v

                        # Readout at the final round of message passing

                if passing_round == self.propagate_layers - 1:

                    # input of shape (seq_len, node_num, input_size)
                    # output of shape (seq_len, node_num, hidden_size)

                    input=torch.autograd.Variable(torch.Tensor(seq_len,max_node_num,64*3)).cuda()

                    for seq_ind in range(seq_len):

                        for n_ind in range(max_node_num):
                            pred_feature[batch_idx, :, seq_ind, n_ind, 0] = hidden_node_states[batch_idx][seq_ind][passing_round+1][0,:,n_ind]

#                            input[seq_ind,:, n_ind,:]=hidden_node_states[batch_idx][seq_ind][passing_round+1][0,:,n_ind]

#                    output, hidden=self.lstm(input)

#                    for seq_id in range(seq_len):

#                        pred_label[batch_idx] = self.readout()
#                        valid_node_num=node_num_rec[batch_idx,seq_id]


 #                       for n_id in range(valid_node_num):

 #                           pred_label[batch_idx,seq_id,n_id,:]=self.lstm_readout(output[seq_id,n_id,:])

        out=self.conv3d1(pred_feature)
        out=self.relu(out)
        out=self.conv3d2(out)

        return out


def main():

    pass

if __name__ == '__main__':

    #print(torch.__version__)

    main()