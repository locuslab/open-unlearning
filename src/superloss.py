import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

loss_div_wd = np.float32(
    [-1000, -0.7357585932962737, -0.7292385198866751, -0.7197861042909649,
     -0.7060825529685993, -0.6862159572880272, -0.6574145455480526, -0.6156599675844636,
     -0.5551266577364037, -0.46736905653740307, -0.34014329294487, -0.15569892914556094,
     0.11169756647530316, 0.4993531412919867, 1.0613531942004133, 1.8761075276533326,
     3.0572900212223724, 4.769698321281568, 7.252246278161051, 10.851297017399714,
     16.06898724880869, 23.63328498268829, 34.599555050301056, 50.497802769609315,
     73.54613907594951, 106.96024960367691, 155.40204460004963, 225.63008495214464,
     327.4425312511471, 475.0441754009414, 689.0282819387658, 999.249744])

conf = np.float32(
    [1, 0.9991138577461243, 0.8724386692047119, 0.8048540353775024, 0.7398145198822021,
     0.6715637445449829, 0.5973713397979736, 0.5154045820236206, 0.42423248291015625,
     0.3226756751537323, 0.20976418256759644, 0.08473344892263412, -0.05296758562326431,
     -0.2036692053079605, -0.3674810528755188, -0.5443023443222046, -0.7338425517082214,
     -0.9356498718261719, -1.149145483970642, -1.3736592531204224, -1.6084641218185425,
     -1.8528070449829102, -2.1059343814849854, -2.367111921310425, -2.6356399059295654,
     -2.910861015319824, -3.1921679973602295, -3.479003667831421, -3.770861864089966,
     -4.067285060882568, -4.367861747741699, -4.67222261428833])


# Credit for LamberW function: Thibault Castells (author of SuperLoss)
class LambertW(nn.Module):
    def __init__(self, weight_decay):
        super().__init__()
        self.weight_decay = weight_decay
        # transformation from: loss_div_wd[1:] --> [0, ..., len(loss_div_wd)-2]
        log_loss_on_wd = torch.log(torch.from_numpy(loss_div_wd[1:]) + 0.750256)
        step = (log_loss_on_wd[-1] - log_loss_on_wd[0]) / (len(log_loss_on_wd) - 1)
        offset = log_loss_on_wd[0]

        # now compute step and offset such that [0,30] --> [-1,1]
        self.log_step = step * (len(log_loss_on_wd) - 1) / 2
        self.log_offset = offset + self.log_step
        self.register_buffer('optimal_conf', torch.from_numpy(conf[1:]).view(1, 1, 1, -1))

    def forward(self, loss):
        loss = loss.detach()

        l = loss / self.weight_decay
        # print(l.shape,loss.shape,self.weight_decay.shape)
        # using grid_sampler in the log-space of loss/wd
        l = (torch.log(l + 0.750256) - self.log_offset) / self.log_step
        # print(l.dtype)
        # print(l.shape)
        # print(torch.isnan(l).shape)
        l[torch.isnan(l)] = -1  # not defined before -0.75
        l = torch.stack((l, l.new_zeros(l.shape)), dim=-1).view(1, 1, -1, 2)
        r = F.grid_sample(self.optimal_conf.to("cuda"), l, padding_mode="border", align_corners=True)
        return torch.exp(r.view(loss.shape))


class SuperLoss(nn.Module):

    def __init__(self, training_type, C=2, lam=0.5,
                 mode='constant'):  # mode = 'avg' is replaced with mode= 'constant' for tau = math.log(C)
        super(SuperLoss, self).__init__()
        self.mode = mode
        self.register_buffer('sum', None)
        self.register_buffer('count', torch.tensor(0.))
        self.tau = math.log(C)
        self.tau_adjusted = math.log(C)
        self.lambertw = LambertW(lam)

        self.training_type = training_type
        print("using superloss in superloss.py = {} ".format(self.training_type))

    def forward(self, loss, alpha, trend_tensor, *args):
        if self.mode == 'avg':
            if self.sum is None:
                self.tau = loss.mean()
                self.sum = torch.tensor(0.).to(loss.device)
            else:
                self.tau = self.sum / self.count

        if self.training_type == "sl":
            conf = self.lambertw(loss - self.tau)
            self.tau_adjusted = self.tau
        else: # self.training_type == "sl_trend"
            self.tau_adjusted = self.tau - alpha * trend_tensor
            conf = self.lambertw(loss - self.tau_adjusted)

        if self.mode == 'avg':
            self.sum += loss.data.sum()
            self.count = self.count.to("cuda") + conf.to("cuda").sum()

        return conf, self.tau, self.tau_adjusted