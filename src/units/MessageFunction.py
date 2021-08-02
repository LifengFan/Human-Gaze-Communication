
import torch
import torch.nn
import torch.autograd

class MessageFunction(torch.nn.Module):
    def __init__(self,message_def,args):
        super(MessageFunction,self).__init__()
        self.m_definition=''
        self.m_function=None
        self.args={}
        self.learn_args=torch.nn.ParameterList([])
        self.learn_modules=torch.nn.ModuleList([])
        self.__set_message(message_def,args)

    def forward(self,h_w,e_vw,args):

        return self.m_function(h_w,e_vw,args)


    def __set_message(self,message_def,args):
        self.m_definition=message_def.lower()
        self.args=args

        self.m_function={
            'linear': self.m_linear,
            'linear_edge': self.m_linear_edge,
            'linear_concat':self.m_linear_concat,
            'linear_concat_relu':self.m_linear_concat_relu,
        }.get(self.m_definition,None)

        if self.m_definition is None:
           print('WARNING!:Message Function has not been set correctly!\n\tIncorrect definition '+message_def)
           quit()

        init_parameters={
            'linear':   self.init_linear,
            'linear_edge':  self.init_linear_edge,
            'linear_concat':    self.init_linear_concat,
            'linear_concat_relu':   self.init_linear_concat_relu,
        }.get(self.m_definition,lambda x:(torch.nn.ParameterList([]),torch.nn.ModuleList([]),{}))


        init_parameters()

    def m_linear(self,h_w,e_vw,args):
        message=torch.autograd.Variable(torch.zeros(e_vw.size()[0],self.args['message_size'],e_vw.size()[2]))

        if hasattr(args,'cuda') and args.cuda:
            message=message.cuda(0)

        for i_node in range(e_vw.size()[2]):
            message[:,:,i_node]=self.learn_modules[0](e_vw[:,:,i_node])+self.learn_modules[1](h_w[:,:,i_node])

        return message

    def init_linear(self):

        edge_feature_size=self.args['edge_feature_size']
        node_feature_size=self.args['node_feature_size']

        message_size=self.args['message_size']

        self.learn_modules.append(torch.nn.Linear(edge_feature_size,message_size,bias=True))
        self.learn_modules.append(torch.nn.Linear(node_feature_size,message_size,bias=True))

    def m_linear_edge(self,h_w,e_vw,args):
        message=torch.autograd.Variable(torch.zeros(e_vw.size()[0],self.args['message_size'],e_vw.size()[2]))
        if hasattr(args,'cuda') and args.cuda:
            message=message.cuda(0)
        for i_node in range(e_vw.size()[2]):
            message[:,:,i_node]=self.learn_modules[0](e_vw[:,:,i_node])

        return message

    def init_linear_edge(self):
        edge_feature_size=self.args['edge_feature_size']
        message_size=self.args['message_size']
        self.learn_modules.append(torch.nn.Linear(edge_feature_size,message_size,bias=True))

    def m_linear_concat(self,h_w,e_vw,args):

        message=torch.autograd.Variable(torch.zeros(e_vw.size()[0],self.args['message_size'],e_vw.size()[2]))
        if hasattr(args,'cuda') and args.cuda:
            message=message.cuda(0)

        for i_node in range(e_vw.size()[2]):
            message[:,:,i_node]=torch.cat([self.learn_modules[0](e_vw[:,:,i_node]),self.learn_modules[1](h_w[:,:,i_node])],1)

        return message


    def init_linear_concat(self):

        edge_feature_size=self.args['edge_feature_size']
        node_feature_size=self.args['node_feature_size']

        message_size=self.args['message_size']/2

        self.learn_modules.append(torch.nn.Linear(edge_feature_size,message_size,bias=True))
        self.learn_modules.append(torch.nn.Linear(node_feature_size,message_size,bias=True))


    def m_linear_concat_relu(self,h_w,e_vw,args):

        message=torch.autograd.Variable(torch.zeros(e_vw.size()[0],self.args['message_size'],e_vw.size()[2]))

        if hasattr(args,'cuda') and args.cuda:
            message=message.cuda(0)

        for i_node in range(e_vw.size()[2]):
            message[:,:,i_node]=self.learn_modules[2](torch.cat([self.learn_modules[0](e_vw[:,:,i_node]),self.learn_modules[1](h_w[:,:,i_node])],1))

        return message

    def init_linear_concat_relu(self):

        edge_feature_size=self.args['edge_feature_size']
        node_feature_size=self.args['node_feature_size']
        message_size=self.args['message_size']/2

        self.learn_modules.append(torch.nn.Linear(edge_feature_size,int(message_size), bias=True))
        self.learn_modules.append(torch.nn.Linear(node_feature_size,int(message_size), bias=True))
        self.learn_modules.append(torch.nn.ReLU())

        #todo: check the relu here


def main():
    pass

if __name__=='__main__':

    main()




