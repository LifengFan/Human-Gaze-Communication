import sys
sys.path.append("/home/lfan/Dropbox/Projects/ICCV19/RunComm/src/")

import torch
import torch.nn
import torch.autograd
import units
import os.path as op


class GNN_SL(torch.nn.Module):
    def __init__(self, args):
        super(GNN_SL, self).__init__()

        self.args = args
        #--------------- base model ------------------
        self.resnet = units.Resnet.ResNet50(3)
        # load headpose-finetuned resnet50 weights
        self._load_pretrained_weight(op.join(args.tmp_root,'checkpoints','finetune_headpose','model_best.pth'))
        self.freeze_res_layer(layer_num=9)
        # replace the last fc layer of resnet with a new one
        # print(self.resnet)

        num_ftrs = self.resnet.resnet.fc.in_features
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 256)
        self.resnet.resnet.fc.weight.data.normal_(0,0.01) #todo: check the initialization here!!
        self.resnet.resnet.fc.bias.data.zero_()


        #---------------- link fun --------------------
        self.link_conv1=torch.nn.Conv2d(in_channels=262*2, out_channels=262, kernel_size=1)
        self.link_conv2=torch.nn.Conv2d(in_channels=262, out_channels=1, kernel_size=1)

        #---------------- message fun ------------------
        self.message_fc=torch.nn.Linear(262,128, bias=True)

        #---------------- update fun -------------------
        self.update_gru=torch.nn.GRU(128,262,num_layers=1,bias=True,dropout=0)

        #---------------- readout fun ------------------
        self.readout_fc1=torch.nn.Linear(262,128, bias=True)
        self.readout_fc2=torch.nn.Linear(128,7, bias=True)

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

    def message_fun(self, h_w):
        # h_w [valid_node_num, 262]
        # out [valid_node_num, 128]
        out=self.message_fc(h_w)

        return out

    def update_fun(self, m_v, h_v):
        # m_v [128]
        # h_v [262]
        self.update_gru.flatten_parameters()
        out, h=self.update_gru(m_v,h_v)

        return h

    def readout_fun(self, h_v):
        out=self.readout_fc1(h_v)
        out=self.relu(out)
        out=self.readout_fc2(out)

        return out

    def forward(self, nodes, pos, num_rec, attmat):
        # nodes [N, max_node_num, 3, 224, 224]
        # pos [N, max_node_num, 6]
        # suppose max_node_num=6
        #-----------------------------------------------------
        # extract node feature using resnet model and pos info
        N=nodes.shape[0]
        max_node_num=nodes.shape[1]
        nodes=nodes.view(N*max_node_num, 3, 224, 224)
        nodes_feature=self.resnet(nodes).view(N,max_node_num,256)

        # node_feature [N, max_node_num,262] plus pos info
        node_feat=torch.cat((nodes_feature,pos),2)
        #------------------------------------------------------
        # get edge feature using node feature
        edge_feat = torch.autograd.Variable(torch.zeros(N, 262*2, max_node_num, max_node_num)).cuda()

        for b_id in range(N):
            valid_node_num=num_rec[b_id]
            for n_id1 in range(valid_node_num):
                for n_id2 in range(valid_node_num):
                        edge_feat[b_id,:,n_id1,n_id2]=torch.cat((node_feat[b_id, n_id1, :].clone(), node_feat[b_id, n_id2, :].clone()),0)

        #-----------------------------------------------------
        hidden_node_state=torch.autograd.Variable(torch.zeros(2, N, 262, max_node_num)).cuda() # passing round=2

        #hidden_edge_state=torch.autograd.Variable(torch.zeros(N, 128, max_node_num, max_node_num)).cuda()

        pred_label=torch.autograd.Variable(torch.zeros(N,max_node_num,7)).cuda()


        attmat=self.link_fun(edge_feat).squeeze(1) # attmat [N, 1, max_node_num, max_node_num]


        for pass_rnd in range(2):
            for b_id in range(N):
                valid_node_num = num_rec[b_id]
                for n_id in range(valid_node_num):

                    if pass_rnd==0:
                        h_v = node_feat[b_id, n_id, :]
                        h_w = node_feat[b_id, :valid_node_num, :]
                    #e_vw = edge_feat[b_id, :, n_id, :valid_node_num]
                    else:
                        h_v=hidden_node_state[pass_rnd-1, b_id,:,n_id]
                        h_w = node_feat[b_id, :valid_node_num, :].clone()
                        for q in range(valid_node_num):
                            h_w[q,:]=hidden_node_state[pass_rnd-1, b_id,:,q]

                    m_v = self.message_fun(h_w)  # m_v [valid_node_num, 128]
                    m_v = attmat[b_id, n_id, :valid_node_num].unsqueeze(1).expand_as(m_v) * m_v

                    # for k in range(valid_node_num):
                    #     hidden_edge_state[b_id, :, n_id, k] = m_v[k, :]

                    m_v = torch.sum(m_v, 0)
                    h_v_new = self.update_fun(m_v[None][None], h_v[None][None].contiguous())

                    hidden_node_state[pass_rnd, b_id, :, n_id] = h_v_new

                    if pass_rnd==1:

                        pred_label[b_id,n_id,:]=self.readout_fun(h_v_new)

        return pred_label, attmat


def main():
    pass

if __name__ == '__main__':

    main()
