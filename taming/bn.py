import torch
import torch.nn as nn
from torch.nn import GroupNorm

class myBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(nn.BatchNorm2d, self).__init__(
            num_channels, eps, momentum, affine, track_running_stats)
            
    def forward(self, input):                             # input(N,C,H,W)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:    # training和track_running_stats都为True才更新BN的参数
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1             # 记录完成前向传播的batch数目
                if self.momentum is None:                 # momentum为None，用1/num_batches_tracked代替
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:                                     # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates 计算均值和方差的过程
        if self.training:
            mean = input.mean([0, 2, 3])                  #计算均值，出来的维度大小等于channel的数目
            # torch.var 默认算无偏估计，这里先算有偏的，因此需要手动设置unbiased=False
            var = torch.mean(torch.abs(input-mean[None, :, None, None]), dim=[0, 2, 3]) # input.var([0, 2, 3], unbiased=False)    # 计算的是有偏方差
            # sub_mean = input - mean[None, :, None, None]
            # sub_mean = sub_mean.permute([1, 0, 2, 3]).contiguous()
            # var = sub_mean.pow(4).sum([1, 2, 3])
            n = input.numel() / input.size(1)             # size(1)是指channel的数目  n=N*H*W
            with torch.no_grad():                         # 计算均值和方差的过程不需要梯度传输
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var 这里通过有偏方差和无偏方差的关系，又转换成了无偏方差
                self.running_var = exponential_average_factor * var  \
                                   + (1 - exponential_average_factor) * self.running_var
        else:                                             # 不处于训练模式就固定running_mean和running_var的值
            mean = self.running_mean
            var = self.running_var
        # 用None扩充维度，然后与原输入tensor做相应运算实现规范化
        # input = self.alpha * (input - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps)
        # input = self.alpha * (input - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps).pow(1/4)
        input = (input - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps)
  
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class myGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        #  super(nn.GroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        super().__init__(num_groups, num_channels, eps, affine)
        self.num_groups = num_groups; 
        self.num_channels = num_channels
        self.affine = affine
        self.eps = eps
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        # var = x.var(-1, keepdim=True, unbiased=False)
        # var = torch.mean(torch.abs(x - mean), -1, keepdim=True)
        var = torch.mean((x-mean)*torch.tanh((x-mean)), -1, keepdim=True)

        # x = (x - mean) / (var + self.eps).sqrt()
        x = (x - mean) / (var + self.eps)
        x = x.view(N, C, H, W)

        # return self.gn(x)
        if self.affine:
            return x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        else:
            return x