import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict
from typing import Optional

def CalculateOutSize(blocks, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    x = torch.rand(1, 1, channels, samples)
    for block in blocks:
        block.eval()
        x = block(x)
    shape = x.shape[-2] * x.shape[-1]
    return shape

class Activation(nn.Module):
    '''
    ShallowConcNet Activation Function
    '''
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6)) 
        else:
            raise Exception('Invalid type !')

        return output


class FFALayer(nn.Module):
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


class EEGNet(nn.Module):

    def __init__(self,
                sample_rate,
                channels=22,
                F1=8,
                D=2,
                F2=16,
                time=1001,
                class_num=4,
                drop_out=0.25,
                bn_track=True) -> None:
        super(EEGNet, self).__init__()
        self.drop_out = drop_out
        self.class_num = class_num
        self.sample_rate = sample_rate 
        self.c = channels
        self.time = time  
        self.F1 = F1  # number of temporal filters
        self.D = D  #D = depth multiplier (number of spatial filters)

        self.F2 = F2  # number of pointwise filters
        self.bn_track = bn_track # track_running_state of BatchNorm2d

        # block1
        # in (1, C, T) out(F1, C, T)
        self.block_1 = nn.Sequential(OrderedDict([
            ('zeropad',nn.ZeroPad2d((int(self.sample_rate/4), int(self.sample_rate/4), 0, 0))), 
            ('conv',nn.Conv2d(1, self.F1, (1, int(self.sample_rate/2)))), #sample rate/2 ##125
            ('bn',nn.BatchNorm2d(self.F1,track_running_stats=self.bn_track))]))

        # block2 DepthwiseConv2d
        # in (F1, C, T) out (D*F1, 1, T//4)
        self.block_2 = nn.Sequential(OrderedDict([
            ('conv',nn.Conv2d(self.F1,
                      self.D * self.F1,
                        kernel_size=(self.c, 1),
                      groups=self.F1)), ('bn',nn.BatchNorm2d(self.D * self.F1, track_running_stats=self.bn_track)),
            ('elu',nn.ELU()), ('avgpool',nn.AvgPool2d((1, 4))), ('drop',nn.Dropout(self.drop_out))]))

        # block3 SeparableConv2d
        # in (D*F1, 1, T//4) out (F2, 1, T//32)
        # F2 = D*F1
        self.block_3 = nn.Sequential(OrderedDict([
            ('zeropad',nn.ZeroPad2d((int(self.sample_rate/16), int(self.sample_rate/16), 0, 0))), #(15,15,0,0)
            ('conv1',nn.Conv2d(self.D * self.F1,
                      self.D * self.F1,
                        kernel_size=(1, int(self.sample_rate/8)), 
                      groups=self.D * self.F1)),  # Depthwise Convolution
            ('conv2',nn.Conv2d(self.D * self.F1, self.F2,
                        kernel_size=(1, 1))),  # Pointwise Convolution
            ('bn',nn.BatchNorm2d(self.F2, track_running_stats=self.bn_track)),
            ('elu',nn.ELU()),
            ('avgpool',nn.AvgPool2d((1, 8))),
            ('drop',nn.Dropout(self.drop_out))]))

        self.fc = nn.Linear((self.F2 * (self.time // 32)), self.class_num, bias=True)

        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # self.fc1 = nn.Linear((self.F2 * (self.time // 32)), (self.F2 * (self.time // 32)), bias=True)
        # self.fc2 = nn.Linear((self.F2 * (self.time // 32)), 256, bias=True)

        # self.fc = nn.Linear(256,self.class_num, bias=True)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        out = self.fc(x)
        # out = F.softmax(x, dim=1)
        return out
    
    def MaxNormConstraint(self):
        for n, p in self.block_2.named_parameters():
            if n == 'conv.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        for n, p in self.fc.named_parameters():
            if n == 'weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


class DeepConvNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5,
                 bn_track=True,
                 TemporalKernel_Times=1):
        super(DeepConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.bn_track = bn_track
        self.TemporalKernel_Times = TemporalKernel_Times # multiplier for temporal convolution kernel size compared to original

        self.block1 = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 10*self.TemporalKernel_Times))),
            ('conv2',nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(Chans, 1))),
            ('bn',nn.BatchNorm2d(num_features=25,track_running_stats=self.bn_track)), ('elu',nn.ELU()),
            ('maxpool',nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))),
            ('dropout',nn.Dropout(self.dropoutRate))]))

        self.block2 = nn.Sequential(OrderedDict([
            ('conv',nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 10*self.TemporalKernel_Times))),
            ('bn',nn.BatchNorm2d(num_features=50, track_running_stats=self.bn_track)), ('elu',nn.ELU()),
            ('maxpool',nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))),
            ('dropout',nn.Dropout(self.dropoutRate))]))

        self.block3 = nn.Sequential(OrderedDict([
            ('conv',nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 10*self.TemporalKernel_Times))),
            ('bn',nn.BatchNorm2d(num_features=100, track_running_stats=self.bn_track)), ('elu',nn.ELU()),
            ('maxpool',nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))),
            ('dropout',nn.Dropout(self.dropoutRate))]))
        
        self.block4 = nn.Sequential(OrderedDict([
            ('conv',nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1,10*self.TemporalKernel_Times))),
            ('bn',nn.BatchNorm2d(num_features=200, track_running_stats=self.bn_track)), ('elu',nn.ELU()),
            ('maxpool',nn.MaxPool2d(kernel_size=(1,3),stride=(1,3))),
            ('dropout',nn.Dropout(self.dropoutRate))])
        )

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=200 *
                      CalculateOutSize([self.block1, self.block2, self.block3, self.block4],
                                       self.Chans, self.Samples),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        # output = F.softmax(output, dim=1)
        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3, self.block4]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)


class ShallowConvNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5,
                 bn_track = True,
                 TemporalKernel_Times=1):
        super(ShallowConvNet, self).__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.bn_track = bn_track
        self.TemporalKernel_Times = TemporalKernel_Times # multiplier for temporal convolution kernel size compared to original
        

        self.block1 = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 25*self.TemporalKernel_Times))),
            ('conv2',nn.Conv2d(in_channels=40,
                      out_channels=40,
                      kernel_size=(self.Chans, 1))),
            ('bn',nn.BatchNorm2d(num_features=40, track_running_stats=self.bn_track)), ('activation1',Activation('square')),
            ('avgpool',nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))),
            ('activation2',Activation('log')),
            ('dropout',nn.Dropout(self.dropoutRate))]))

        self.classifier_block = nn.Sequential(
            nn.Linear(
                in_features=40 *
                CalculateOutSize([self.block1], self.Chans, self.Samples),
                out_features=self.n_classes,
                bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        # output = F.softmax(output, dim=1)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)

class EEGNetFedFa(EEGNet):
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
        self.bn_track = bn_track # track_running_state of BatchNorm2d

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
    def __init__(self, n_classes: int, Chans: int, Samples: int, dropoutRate: Optional[float] = 0.5, bn_track=True, TemporalKernel_Times=1,prob=0.5):
        super(DeepConvNetFa,self).__init__(n_classes, Chans, Samples, dropoutRate, bn_track, TemporalKernel_Times)

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.bn_track = bn_track
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
    def __init__(self, n_classes: int, Chans: int, Samples: int, dropoutRate: Optional[float] = 0.5, bn_track=True, TemporalKernel_Times=1, prob=0.5):
        super(ShallowConvNetFa,self).__init__(n_classes, Chans, Samples, dropoutRate, bn_track, TemporalKernel_Times)
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.bn_track = bn_track
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


if __name__ == '__main__':
    x = torch.randn(64, 1, 22, 1001)

    # model = EEGNet(sample_rate=250)
    # model = DeepConvNet(4,22,1001,0.5,True)
    # model = ShallowConvNet(4,22,1001,0.5,True)
    model = EEGNetFedFa(sample_rate=250)
    print(model)
    model.train()
    out = model(x)
    print(out.shape)


