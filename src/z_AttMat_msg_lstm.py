import os

import torch
import torch.autograd
import torch.nn

import units


class AttMat_msg_lstm(torch.nn.Module):
    def __init__(self, model_args, args):
        super(AttMat_msg_lstm, self).__init__()

        self.model_args = model_args.copy()

        # TODO: I don't think resnet34 is sufficient for this task,
        # TODO: you'd better adopt resnet50 for feature extraction at least.
        self.resnet = units.Resnet34(model_args['roi_feature_size'])
        # TODO: The init of the ConvLSTM is important, double check the init procedure.
        self.link_fun = units.LinkFunction('GraphConv', model_args)
        self.base_model = torch.nn.ModuleList([])

        self.load_base_model = args.load_base_model
        self.base_model_weight = args.base_model_weight
        self.freeze_base_model = args.freeze_base_model

        self.message_fun = units.MessageFunction('linear_concat_relu', model_args)
        self.update_fun = units.UpdateFunction('gru', model_args)
        self.readout_fun = units.ReadoutFunction('fc', {'readout_input_size': model_args['node_feature_size'],
                                                        'output_classes': model_args['big_attr_classes']})

        self.base_model.append(self.resnet)
        self.base_model.append(self.link_fun)
        self.base_model.append(self.message_fun)
        self.base_model.append(self.update_fun)

        self.lstm = torch.nn.LSTM(model_args['roi_feature_size'], self.model_args['lstm_hidden_size'])
        self.lstm_readout = units.ReadoutFunction('fc', {'readout_input_size': model_args['lstm_hidden_size'],
                                                         'output_classes': model_args['big_attr_classes']})
        self.propagate_layers = model_args['propagate_layers']

        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.maxpool = torch.nn.MaxPool2d((2, 2))

        if self.load_base_model:
            self._load_base_model_weight(self.base_model_weight)

        if self.freeze_base_model:

            for i, param in self.base_model[0].named_parameters():
                param.requires_grad = False

    def forward(self, node_feature, edge_feature, AttMat, node_num_rec, args):

        # node_feature [N,seq,node_num,3,224,224]
        # edge_feature [N,seq,node_num,node_num,3,224,224]
        # node_num_rec=[N, seq]

        batch_size = node_feature.size()[0]
        seq_len = node_feature.size()[1]
        max_node_num = node_feature.size()[2]

        node_resnet = torch.autograd.Variable(
            torch.zeros(batch_size, seq_len, self.model_args['roi_feature_size'], max_node_num))
        edge_resnet = torch.autograd.Variable(
            torch.zeros(batch_size, seq_len, self.model_args['roi_feature_size'], max_node_num, max_node_num))

        if hasattr(args, 'cuda') and args.cuda:
            node_resnet = node_resnet.cuda()
            edge_resnet = edge_resnet.cuda()

        for b_idx in range(batch_size):
            for sq_idx in range(seq_len):
                valid_node_num = node_num_rec[b_idx, sq_idx]
                for n_idx in range(valid_node_num):
                    node_resnet[b_idx, sq_idx, :, n_idx] = self.base_model[0](
                        node_feature[b_idx, sq_idx, n_idx, ...].unsqueeze(0))

        for b_idx in range(batch_size):
            for sq_idx in range(seq_len):
                valid_node_num = node_num_rec[b_idx, sq_idx]
                for n_idx1 in range(valid_node_num):
                    for n_idx2 in range(valid_node_num):
                        if n_idx2 == n_idx1:
                            continue
                        edge_resnet[b_idx, sq_idx, :, n_idx1, n_idx2] = self.base_model[0](
                            edge_feature[b_idx, sq_idx, n_idx1, n_idx2, ...].unsqueeze(0))

        hidden_node_states = [
            [[node_resnet[batch_i, sq_i, ...].unsqueeze(0).clone() for _ in range(self.propagate_layers + 1)] for sq_i
             in range(seq_len)] for batch_i in range(batch_size)]
        hidden_edge_states = [
            [[edge_resnet[batch_i, sq_i, ...].unsqueeze(0).clone() for _ in range(self.propagate_layers + 1)] for sq_i
             in range(seq_len)] for batch_i in range(batch_size)]

        pred_adj_mat = torch.autograd.Variable(torch.zeros(batch_size, seq_len, max_node_num, max_node_num))
        pred_label = torch.autograd.Variable(
            torch.zeros(batch_size, seq_len, max_node_num, self.model_args['big_attr_classes']))

        if hasattr(args, 'cuda') and args.cuda:
            pred_adj_mat = pred_adj_mat.cuda()
            pred_label = pred_label.cuda()
            # hidden_node_states=torch.autograd.Variable(torch.FloatTensor(hidden_node_states)).cuda()
            # hidden_edge_states=torch.autograd.Variable(torch.FloatTensor(hidden_edge_states)).cuda()

        # msg passing
        # actually, batch size here is only 1
        for batch_idx in range(batch_size):

            for passing_round in range(self.propagate_layers):

                for sq_idx in range(seq_len):

                    valid_node_num = node_num_rec[batch_idx, sq_idx]

                    pred_adj_mat[batch_idx, sq_idx, :valid_node_num, :valid_node_num] = self.base_model[1](
                        hidden_edge_states[batch_idx][sq_idx][passing_round][:, :, :valid_node_num, :valid_node_num])

                    sigmoid_pred_adj_mat = self.sigmoid(
                        pred_adj_mat[batch_idx, sq_idx, :valid_node_num, :valid_node_num].unsqueeze(0))

                    # Loop through nodes

                    for i_node in range(valid_node_num):
                        h_v = hidden_node_states[batch_idx][sq_idx][passing_round][:, :, i_node]
                        h_w = hidden_node_states[batch_idx][sq_idx][passing_round][:, :, :valid_node_num]
                        e_vw = edge_resnet[batch_idx, sq_idx, :, i_node, :valid_node_num].unsqueeze(0)

                        m_v = self.base_model[2](h_w, e_vw, args)

                        # sum up messages from different nodes according to weights

                        # m_v [1,message_size,node_num]
                        # sigmoidi_ored_adj_mat [1,node_num,node_num]

                        m_v = sigmoid_pred_adj_mat[:, i_node, :valid_node_num].unsqueeze(1).expand_as(m_v) * m_v
                        hidden_edge_states[batch_idx][sq_idx][passing_round + 1][:, :, i_node, :valid_node_num] = m_v
                        m_v = torch.sum(m_v, 2)
                        h_v = self.base_model[3](m_v[None], h_v[None].contiguous())

                        hidden_node_states[batch_idx][sq_idx][passing_round + 1][:, :, i_node] = h_v

                        # Readout at the final round of message passing

                if passing_round == self.propagate_layers - 1:

                    # input of shape (seq_len, node_num, input_size)
                    # output of shape (seq_len, node_num, hidden_size)

                    input = torch.autograd.Variable(
                        torch.Tensor(seq_len, max_node_num, self.model_args['roi_feature_size'])).cuda()

                    for seq_ind in range(seq_len):

                        for n_ind in range(max_node_num):
                            input[seq_ind, n_ind, :] = hidden_node_states[batch_idx][seq_ind][passing_round + 1][0, :,
                                                       n_ind]

                    output, hidden = self.lstm(input)

                    for seq_id in range(seq_len):

                        valid_node_num = node_num_rec[batch_idx, seq_id]

                        for n_id in range(valid_node_num):
                            pred_label[batch_idx, seq_id, n_id, :] = self.lstm_readout(output[seq_id, n_id, :])
                            # pred_label[batch_idx,seq_id,n_id,:]=self.readout_fun(input[seq_id,n_id,:])

        return self.sigmoid(pred_adj_mat), pred_label

    def _load_base_model_weight(self, best_model_file):

        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)

            pretrained_dict = checkpoint['state_dict']

            model_dict = self.base_model.state_dict()

            new_pretrained_dict = {k.strip('base_model.'): v for k, v in pretrained_dict.items() if
                                   k.strip('base_model.') in model_dict}

            model_dict.update(new_pretrained_dict)

            self.base_model.load_state_dict(model_dict)

            print('loaded base model from {}'.format(best_model_file))


def main():
    pass


if __name__ == '__main__':

    # print(torch.__version__)
    main()
