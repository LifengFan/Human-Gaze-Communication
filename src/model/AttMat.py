import sys
import os
sys.path.append("/home/lfan/Dropbox/Projects/ICCV19/RunComm/src/")

import torch
import torch.nn
import torch.autograd
import units
import os.path as op


class AttMat(torch.nn.Module):
    def __init__(self, args):
        super(AttMat, self).__init__()

        self.args = args
        self.resnet = units.Resnet.ResNet50(3)

        self._load_pretrained_weight(op.join(args.tmp_root,'checkpoints','finetune_headpose','model_best.pth'))

        # replace the last fc layer of resnet with a new one
        num_ftrs = self.resnet.resnet.fc.in_features
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 6)

        self.resnet.resnet.fc.weight.data.normal_(0,0.01)
        self.resnet.resnet.fc.bias.data.zero_()

        self.tanh=torch.nn.Tanh()
        self.fc1=torch.nn.Linear(18,18)
        self.fc2=torch.nn.Linear(18,2)


    def _load_pretrained_weight(self, model_path):

        model_dict=self.resnet.state_dict()
        pretrained_model=torch.load(model_path)['state_dict']

        pretrained_dict={}

        for k,v in pretrained_model.items():
            if k[len('module.resnet.'):] in model_dict:
                pretrained_dict[k[len('module.resnet.'):]]=v

        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

    def freeze_res_layer(self, layer_num=9):

        child_cnt=0
        for child in self.resnet.children():
            if child_cnt<layer_num:
                for param in child.parameters():
                    param.requires_grad=False

            child_cnt+=1

            print('Resnet has {} children totally, {} has been freezed'.format(child_cnt, layer_num))


    def forward(self, head_patch, pos):
        # head_patch [N, 3,224,224]
        # pos [N, 12]

        out=self.resnet(head_patch)
        out=self.tanh(out)
        out=torch.cat((out, pos),1)

        out=self.fc1(out)
        out=self.tanh(out)
        out=self.fc2(out)

        return out


def main():

    pass

if __name__ == '__main__':


    main()

