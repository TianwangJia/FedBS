import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import *
from utils import *

class FFALayer(nn.Module):
    '''
    FFAlayer of model in fedfa
    '''
    def __init__(self, prob=0.5, eps=1e-6, momentum1=0.99, momentum2=0.99, nfeat=None):
        super(FFALayer, self).__init__()
        self.prob = prob
        self.eps = eps
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.nfeat = nfeat

        self.register_buffer('running_var_mean_bmic', torch.ones(self.nfeat))
        self.register_buffer('running_var_std_bmic', torch.ones(self.nfeat))
        self.register_buffer('running_mean_bmic', torch.zeros(self.nfeat))
        self.register_buffer('running_std_bmic', torch.ones(self.nfeat))

    def forward(self, x):
        if not self.training: return x
        if np.random.random() > self.prob: return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps)
        std = std.sqrt()

        self.momentum_updating_running_mean_and_std(mean, std)

        var_mu = self.var(mean)
        var_std = self.var(std)

        running_var_mean_bmic = 1 / (1 + 1 / (self.running_var_mean_bmic + self.eps))
        gamma_mu = x.shape[1] * running_var_mean_bmic / sum(running_var_mean_bmic)

        running_var_std_bmic = 1 / (1 + 1 / (self.running_var_std_bmic + self.eps))
        gamma_std = x.shape[1] * running_var_std_bmic / sum(running_var_std_bmic)

        var_mu = (gamma_mu + 1) * var_mu
        var_std = (gamma_std + 1) * var_std

        var_mu = var_mu.sqrt().repeat(x.shape[0], 1)
        var_std = var_std.sqrt().repeat(x.shape[0], 1)

        beta = self.gaussian_sampling(mean, var_mu)
        gamma = self.gaussian_sampling(std, var_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    def gaussian_sampling(self, mu, std):
        e = torch.randn_like(std)
        z = e.mul(std).add_(mu)
        return z

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def var(self, x):
        t = x.var(dim=0, keepdim=False) + self.eps
        return t

    def momentum_updating_running_mean_and_std(self, mean, std):
        with torch.no_grad():
            self.running_mean_bmic = self.running_mean_bmic * self.momentum1 + \
                                     mean.mean(dim=0, keepdim=False) * (1 - self.momentum1)
            self.running_std_bmic = self.running_std_bmic * self.momentum1 + \
                                    std.mean(dim=0, keepdim=False) * (1 - self.momentum1)

    def momentum_updating_running_var(self, var_mean, var_std):
        with torch.no_grad():
            self.running_var_mean_bmic = self.running_var_mean_bmic * self.momentum2 + var_mean * (1 - self.momentum2)
            self.running_var_std_bmic = self.running_var_std_bmic * self.momentum2 + var_std * (1 - self.momentum2)

class EEGNetFedFa(EEGNet):
    '''
    Adding ffalayer to EEGNet
    '''
    def __init__(self,
                sample_rate,
                channels=22,
                F1=8,
                D=2,
                F2=16,
                time=1001,
                class_num=4,
                drop_out=0.25,
                bn_track=True,
                prob=0.5) -> None:
        super(EEGNetFedFa, self).__init__(sample_rate=sample_rate,channels=channels,F1=F1,D=D,F2=F2,time=time,class_num=class_num,
                                          drop_out=drop_out,bn_track=bn_track)
        self.drop_out = drop_out
        self.class_num = class_num
        self.sample_rate = sample_rate 
        self.c = channels
        self.time = time  
        self.F1 = F1  # number of temporal filters
        self.D = D  #D = depth multiplier (number of spatial filters)

        self.F2 = F2  # number of pointwise filters
        self.bn_track = bn_track # track_running_state of BatchNorm2d, if fedbs, bn_track = False

        self.ffa_layer1 = FFALayer(prob=prob,nfeat=self.F1)
        self.ffa_layer2 = FFALayer(prob=prob, nfeat=self.D*self.F1)
        self.ffa_layer3 = FFALayer(prob=prob, nfeat=self.F2)
        
        self.ffa_layer0 = FFALayer(prob=prob, nfeat=1)

    def forward(self, x):
        x = self.ffa_layer0(x)

        x = self.block_1(x)
        x = self.ffa_layer1(x)

        x = self.block_2(x)
        x = self.ffa_layer2(x)

        x = self.block_3(x)
        x = self.ffa_layer3(x)

        # x = self.block_1[0:2](x)
        # x = self.ffa_layer1(x)
        # x = self.block_1[2:](x)

        # x = self.block_2[0](x)
        # x = self.ffa_layer2(x)
        # x = self.block_2[1:](x)

        # x = self.block_3(x)

        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # out = F.softmax(x, dim=1)
        return out

class DeepConvNetFa(DeepConvNet):
    '''
    Adding ffalayer to DeepConvNet
    '''
    def __init__(self, n_classes: int, Chans: int, Samples: int, dropoutRate: Optional[float] = 0.5, bn_track=True, TemporalKernel_Times=1,prob=0.5):
        super(DeepConvNetFa,self).__init__(n_classes, Chans, Samples, dropoutRate, bn_track, TemporalKernel_Times)

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.bn_track = bn_track # track_running_state of BatchNorm2d, if fedbs, bn_track = False
        self.TemporalKernel_Times = TemporalKernel_Times # multiplier for temporal convolution kernel size compared to original

        self.ffalayer0 = FFALayer(prob=prob, nfeat=1)
        self.ffalayer1 = FFALayer(prob=prob, nfeat=25)
        self.ffalayer2 = FFALayer(prob=prob, nfeat=50)
        self.ffalayer3 = FFALayer(prob=prob, nfeat=100)
        self.ffalayer4 = FFALayer(prob=prob, nfeat=200)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffalayer0(x)

        output = self.block1(x)
        output = self.ffalayer1(output)

        output = self.block2(output)
        output = self.ffalayer2(output)

        output = self.block3(output)
        output = self.ffalayer3(output)

        output = self.block4(output)
        output = self.ffalayer4(output)

        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        # output = F.softmax(output, dim=1)

        return output
    
class ShallowConvNetFa(ShallowConvNet):
    '''
    Adding ffalayer to ShallowConvNet
    '''
    def __init__(self, n_classes: int, Chans: int, Samples: int, dropoutRate: Optional[float] = 0.5, bn_track=True, TemporalKernel_Times=1, prob=0.5):
        super(ShallowConvNetFa,self).__init__(n_classes, Chans, Samples, dropoutRate, bn_track, TemporalKernel_Times)
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.bn_track = bn_track # track_running_state of BatchNorm2d, if fedbs, bn_track = False
        self.TemporalKernel_Times = TemporalKernel_Times # multiplier for temporal convolution kernel size compared to original

        self.ffalayer0 = FFALayer(prob=prob, nfeat=1)
        self.ffalayer1 = FFALayer(prob=prob, nfeat=40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffalayer0(x)

        output = self.block1(x)
        output = self.ffalayer1(output)
        
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        # output = F.softmax(output, dim=1)

        return output


def refine_weight_dict_by_GA(weight_dict, candidates_id_list, site_before_results_dict, site_after_results_dict, step_size=0.1, fair_metric='loss'):
    '''
    Adjust model aggregation weights based on generalization error in GA
    '''
    if fair_metric == 'acc':
        signal = -1.0
    elif fair_metric == 'loss':
        signal = 1.0
    else:
        raise ValueError('fair_metric must be acc or loss')
    
    value_list = []
    for id in candidates_id_list:
        value_list.append((site_after_results_dict[id].cpu() - site_before_results_dict[id].cpu()).item())

    value_list = np.array(value_list)
    
    step_size = 1/len(weight_dict) * step_size 
    norm_gap_list = value_list / np.max(np.abs(value_list))
    
    for i in range(len(candidates_id_list)):
        weight_dict[candidates_id_list[i]] += signal * norm_gap_list[i] * step_size 

    weight_dict = weight_clip(weight_dict) 
    
    return weight_dict


def weight_clip(weight_dict):
    new_total_weight = 0.0
    for key_name in weight_dict.keys():
        weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
        new_total_weight += weight_dict[key_name]
    
    for key_name in weight_dict.keys():
        weight_dict[key_name] /= new_total_weight
    
    return weight_dict