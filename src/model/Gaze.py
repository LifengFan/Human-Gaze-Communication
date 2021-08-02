import torch
import torch.nn
import torch.autograd
import units


class Gaze(torch.nn.Module):
    def __init__(self,args):
        super(Gaze, self).__init__()

        self.resnet = units.Resnet.ResNet50(3)

        self.tanh=torch.nn.Tanh()

        self.args = args

    def forward(self, head_patch):

        #head_patch [N,3,224,224]
        #rpy [N,3]
        return self.tanh(self.resnet(head_patch))


def main():

    pass

if __name__ == '__main__':


    main()




