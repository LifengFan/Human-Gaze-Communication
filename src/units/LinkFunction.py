
import torch
import torch.nn
from . import ConvLSTM


class LinkFunction(torch.nn.Module):
    def __init__(self,link_def,args):
        super(LinkFunction,self).__init__()
        self.l_definition=''
        self.l_function=None
        self.learn_args=torch.nn.ParameterList([])
        self.learn_modules=torch.nn.ModuleList([])
        self.__set_link(link_def,args)

    def forward(self,edge_features):
        return self.l_function(edge_features)

    def __set_link(self,link_def,args):

        self.l_definition=link_def.lower()
        self.args=args

        self.l_function={
            'graphconv':self.l_graph_conv,
            'graphconvlstm':self.l_graph_conv_lstm,
        }.get(self.l_definition,None)

        if self.l_function is None:
            print('WARNING!:Link Function has not been set correctly!\n\tIncorrect definition '+link_def)
            quit()

        init_parameters={
            'graphconv':self.init_graph_conv,
            'graphconvlstm':self.init_graph_conv_lstm,
        }.get(self.l_definition,lambda x:(torch.nn.ParameterList([]),torch.nn.ModuleList([]),{}))

        init_parameters() # todo: to check out why

    def l_graph_conv(self,edge_features):
        last_layer_output=edge_features
        for layer in self.learn_modules:
            last_layer_output=layer(last_layer_output)

        return last_layer_output  # todo: double check

    def init_graph_conv(self):
        input_size=self.args['edge_feature_size']
        hidden_size=self.args['link_hidden_size']

        for _ in range(self.args['link_hidden_layers']-1):

            self.learn_modules.append(torch.nn.Conv2d(in_channels=input_size,out_channels=hidden_size,kernel_size=1))
            self.learn_modules.append(torch.nn.ReLU())

            input_size=hidden_size

        self.learn_modules.append(torch.nn.Conv2d(in_channels=input_size,out_channels=1,kernel_size=1))


    def l_graph_conv_lstm(self,edge_features):

        last_layer_output=self.ConvLSTM(edge_features)

        for layer in self.learn_modules:
            last_layer_output=layer(last_layer_output)

        return last_layer_output  # todo: double check

    def init_graph_conv_lstm(self):

        input_size=self.args['edge_feature_size']
        hidden_size=self.args['link_hidden_size']
        hidden_layers=self.args['link_hidden_layers']

        self.ConvLSTM=ConvLSTM.ConvLSTM(input_size,hidden_size,hidden_layers)
        self.learn_modules.append(torch.nn.Conv2d(hidden_size,1,1))
        self.learn_modules.append(torch.nn.Sigmoid())



def main():
    pass

if __name__=='__main__':

    main()
