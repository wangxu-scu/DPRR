import torch
import torch.nn as nn
import numpy as np
import time


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage)
            )
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = self.module_name + "_"
            name = time.strftime(prefix + "%m%d_%H:%M:%S.pth")
        torch.save(self.state_dict(), "checkpoint/" + name)
        return name

    def forward(self, *input):
        pass


class ImgModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024 * 8, mid_num2=1024 * 8, 
                 hiden_layer=3, num_classes=10):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.norm = norm

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        

    def forward(self, x):
        feature = self.fc(x)
        out = torch.tanh(feature)
        
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
            
        return out


class TxtModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024 * 8, mid_num2=1024 * 8, 
                 hiden_layer=3, num_classes=10):
        super(TxtModule, self).__init__()
        self.module_name = "text_model"
        self.bit = bit
        self.num_classes = num_classes
        self.norm = norm

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)


    def forward(self, x):
        feature = self.fc(x)
        out = torch.tanh(feature)
        
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
            
        return out


class CrossModalNet(nn.Module):
    def __init__(self, txt_feature_dim, img_feature_dim, bit,num_classes,hiden_layer):
        super(CrossModalNet, self).__init__()

        self.img_net=ImgModule(img_feature_dim, bit, hiden_layer=hiden_layer,num_classes=num_classes)
        self.txt_net=TxtModule(txt_feature_dim, bit, hiden_layer=hiden_layer,num_classes=num_classes)




        W1 = torch.Tensor(bit, num_classes)
        W1 = torch.nn.init.orthogonal_(W1, gain=1)
        W1 = torch.tensor(W1, requires_grad=True)
        self.W1 = torch.nn.Parameter(W1)

        W2 = torch.Tensor(bit, num_classes)
        W2 = torch.nn.init.orthogonal_(W2, gain=1)
        W2 = torch.tensor(W2, requires_grad=True)
        self.W2 = torch.nn.Parameter(W2)
    def forward(self, img, txt):

        img_feature = self.img_net(img)
        txt_feature = self.txt_net(txt)
        
        img_prob_logits = torch.matmul(img_feature, self.W1)
        txt_prob_logits = torch.matmul(txt_feature, self.W1)

        return img_feature, txt_feature, img_prob_logits,txt_prob_logits