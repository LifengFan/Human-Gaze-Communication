import sys
sys.path.append("/home/lfan/Dropbox/Projects/ICCV19/RunComm/src/")
import torch
import torch.nn
import torch.autograd
import units
import os.path as op


class Atomic_GNN_lstm(torch.nn.Module):
    def __init__(self, args):
        super(Atomic_GNN_lstm, self).__init__()

        self.args = args
        #--------------- base model ------------------
        self.resnet = units.Resnet.ResNet50(3)
        # load headpose-finetuned resnet50 weights
        #self._load_pretrained_weight(op.join(args.tmp_root,'checkpoints','finetune_headpose','model_best.pth'))
        #self.freeze_res_layer(layer_num=9)
        # replace the last fc layer of resnet with a new one
        # print(self.resnet)

        # num_ftrs = self.resnet.resnet.fc.in_features
        # self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 256)
        # self.resnet.resnet.fc.weight.data.normal_(0,0.01)
        # self.resnet.resnet.fc.bias.data.zero_()

        num_ftrs = self.resnet.resnet.fc.in_features
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 6)

        self.fc1=torch.nn.Linear(18,18)
        self.fc2=torch.nn.Linear(18,2)
        # input for conv3d is [N, 5, 2, 4, 2]
        # (N,C_in ,D,H,W) and output (N, C_{out}, D_{out}, H_{out}, W_{out})
        self.conv3d1=torch.nn.Conv3d(in_channels=2, out_channels=6,kernel_size=(5,2,4),stride=1, padding=0) #Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3d2=torch.nn.Conv3d(in_channels=6,out_channels=6,kernel_size=(1,1,1), stride=1, padding=0)

        #-----------------------------------------------------
        # todo: load attmat weight
        # todo: freeze layer
        self._load_pretrained_weight(op.join(args.tmp_root, 'checkpoints', 'attmat', 'model_best.pth'))
        self.freeze_res_layer(layer_num=9)

        #-------------------------------------------------------
        self.nodefeat_n0=6  #3
        self.nodefeat_n1=6+6

        #---------------- link fun --------------------
        self.link_conv1=torch.nn.Conv2d(in_channels=self.nodefeat_n1*2, out_channels=self.nodefeat_n1*2, kernel_size=1)
        self.link_conv2=torch.nn.Conv2d(in_channels=self.nodefeat_n1*2, out_channels=1, kernel_size=1)

        #---------------- message fun ------------------
        self.message_fc=torch.nn.Linear(self.nodefeat_n1, self.nodefeat_n1, bias=True)
        self.message_fc2=torch.nn.Linear(2, self.nodefeat_n1, bias=True)

        #---------------- update fun -------------------
        self.update_gru=torch.nn.GRU(self.nodefeat_n1*2, self.nodefeat_n1, num_layers=1,bias=True,dropout=0)

        #---------------- readout fun ------------------
        self.readout_fc1=torch.nn.Linear(self.nodefeat_n1, self.nodefeat_n1, bias=True)
        self.readout_fc2=torch.nn.Linear(self.nodefeat_n1, 6, bias=True)

        #---------------- lstm fun ---------------------
        self.lstm=torch.nn.LSTM(input_size=self.nodefeat_n1*4, hidden_size=self.nodefeat_n1*2, batch_first=True,bidirectional=True)
        self.lstm.flatten_parameters()
        self.lstm_readout_fc1=torch.nn.Linear(self.nodefeat_n1*2*2, self.nodefeat_n1, bias=True)
        self.lstm_readout_fc2=torch.nn.Linear(self.nodefeat_n1, 6, bias=True)

        #-----------------------------------------------
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
        self.relu=torch.nn.ReLU()
        self.dropout=torch.nn.Dropout(p=0.5)


    def _load_pretrained_weight(self, model_path):

        model_dict=self.resnet.state_dict()
        pretrained_model=torch.load(model_path)['state_dict']

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
            #print('-'*15)
            #print('resnet child {}'.format(child_cnt))
            #print(child)
            if child_cnt<layer_num:
                for param in child.parameters():
                    param.requires_grad=False

            child_cnt+=1

        print('Resnet has {} children totally, {} has been freezed'.format(child_cnt, layer_num))

    def link_fun(self, edge_feat):
        # input: edge_feat [N, 262*2, max_node_num, max_node_num]
        # output: AttMat [N, max_node_num, max_node_num]
        out=self.link_conv1(edge_feat)
        out=self.relu(out)
        out=self.link_conv2(out)
        out=self.sigmoid(out)

        return out

    def message_fun(self, h_w, edge_feat):
        # h_w [sq_len, valid_node_num, 262]
        # out [sq_len, valid_node_num, 128]
        out=self.message_fc(h_w)
        out2=self.message_fc2(edge_feat)
        m=self.relu(torch.cat((out,out2),-1))

        return m

    def update_fun(self, m_v, h_v):
        # m_v [1, 10, 128]  (seq_len, batch, input_size)
        # h_v [1, 10, 262]
        self.update_gru.flatten_parameters()
        out, h=self.update_gru(m_v,h_v)

        return h

    def readout_fun(self, h_v):
        out=self.readout_fc1(h_v)
        out=self.relu(out)
        out=self.readout_fc2(out)

        return out

    def lstm_readout(self, h_v):

        out=self.lstm_readout_fc1(h_v)
        out=self.relu(out)
        out=self.lstm_readout_fc2(out)

        return out

    def forward(self, nodes, pos, attmat):
        # nodes [N, sq, 4, 3, 224, 224]
        # pos [N, sq, 4, 6]
        # attmat [N, sq, 4, 4]
        #-----------------------------------------------------
        # extract node feature using resnet model and pos info
        N=nodes.shape[0]
        sq_len=5
        max_node_num=4
        iterN=1

        nodes=nodes.view(N*sq_len*max_node_num, 3, 224, 224)
        nodes_feature=self.resnet(nodes).view(N, sq_len, max_node_num, self.nodefeat_n0)

        # node_feature [N, sq_len, max_node_num,262] plus pos info
        #node_feat=torch.cat((nodes_feature,pos),3)
        #------------------------------------------------------
        # get edge feature using node feature
        # edge_feat = torch.autograd.Variable(torch.zeros(N, sq_len, self.nodefeat_n1*2, max_node_num, max_node_num)).cuda()
        edge_feat = torch.autograd.Variable(torch.zeros(N, sq_len, max_node_num, max_node_num, 2)).cuda()

        # change the edge feature to the pred_attmat 2 channels

        for b_id in range(N):
            for sq_id in range(sq_len):
                valid_node_num=4
                for n_id1 in range(valid_node_num):
                    for n_id2 in range(valid_node_num):
                        #edge_feat[b_id,sq_id, :,n_id1,n_id2]=torch.cat((node_feat[b_id, sq_id, n_id1, :].clone(), node_feat[b_id, sq_id, n_id2, :].clone()),0)

                        h_out = self.tanh(nodes_feature[b_id, sq_id, n_id1, :])

                        ht_pos = torch.cat((pos[b_id, sq_id, n_id1, :], pos[b_id, sq_id, n_id2, :]), -1)
                        out = torch.cat((h_out, ht_pos), -1).unsqueeze(0)

                        # print(out.shape[:])
                        out = self.fc1(out)
                        out = self.tanh(out)
                        edge_feat[b_id, sq_id, n_id1, n_id2,:] =self.fc2(out).squeeze(0)

                        #-----------------------------------------------------
        hidden_node_state=torch.autograd.Variable(torch.zeros(iterN, N, sq_len, self.nodefeat_n1, max_node_num)).cuda() # passing round=2
        #hidden_edge_state=torch.autograd.Variable(torch.zeros(N, 128, max_node_num, max_node_num)).cuda()
        pred_label=torch.autograd.Variable(torch.zeros(N, 6)).cuda()
        pred_label0 = torch.autograd.Variable(torch.zeros(N, 6)).cuda()
        #attmat=self.link_fun(edge_feat.view(N*sq_len, 262*2,6,6)).squeeze(1).view(N,sq_len,6,6) # attmat [N, 1, max_node_num, max_node_num]

        for pass_rnd in range(iterN):
            for b_id in range(N):
                #---------------------------------------
                # message passing for the whole sequence
                #for sq_id in range(sq_len):
                valid_node_num = 4
                for n_id in range(valid_node_num):

                        if pass_rnd == 0:
                            h_v = nodes_feature[b_id, :, n_id, :] #[sq_len, 262]
                            h_w = nodes_feature[b_id, :, :valid_node_num, :]
                        # e_vw = edge_feat[b_id, :, n_id, :valid_node_num]
                        else:
                            h_v = hidden_node_state[pass_rnd - 1, b_id, :, :, n_id]
                            h_w = nodes_feature[b_id, :, :valid_node_num, :].clone()
                            for q in range(valid_node_num):
                                h_w[:, q, :] = hidden_node_state[pass_rnd-1, b_id, :, :, q]

                        m_v = self.message_fun(h_w, edge_feat[b_id,:,n_id,:valid_node_num,:])  # m_v [sq_len, valid_node_num, 128]

                        # edge_feat [N, sq_len, 2, max_node_num, max_node_num)]

                        m_v = attmat[b_id, :, n_id, :valid_node_num].unsqueeze(2).expand_as(m_v) * m_v

                        # for k in range(valid_node_num):
                        #     hidden_edge_state[b_id, :, n_id, k] = m_v[k, :]

                        m_v = torch.sum(m_v, 1)
                        # m_v [sq_len, 128]
                        # h_v [sq_len, 262]
                        h_v_new = self.update_fun(m_v[None], h_v[None].contiguous()) # [1, 10, 262]

                        hidden_node_state[pass_rnd, b_id, :, :, n_id] = h_v_new  # [2, N, sq_len, 262, max_node_num]

                # ------------------------------------
                # lstm for the sequence
                if pass_rnd==(iterN-1):
                    #sq_node_num=4
                    #for nid in range(sq_node_num):

                        # frame-only classification
                        # hidden_node_state[pass_rnd, b_id, :, :, nid] [se_len, 262]

                        #pred_label0[b_id,:,nid,:]=self.readout_fun(hidden_node_state[pass_rnd, b_id, :,:, nid].clone())

                        # input [1, sq_len, 262]
                        self.lstm.flatten_parameters()
                        lstm_input=hidden_node_state[pass_rnd, b_id, :, :,:].clone()

                        #output, _=self.lstm(hidden_node_state[pass_rnd, b_id, :, :].clone().unsqueeze(0)) # [1, sq_len,  2*262]
                        output, output_t = self.lstm(lstm_input.view(sq_len,-1).unsqueeze(0))  #

                        pred_label[b_id, :] = self.lstm_readout(output[:,-1,:]) # pred_label [N,sq_len, max_node_num,7]


        return pred_label


def main():
    pass

if __name__ == '__main__':

    main()
