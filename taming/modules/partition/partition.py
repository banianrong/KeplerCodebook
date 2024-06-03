import torch.nn as nn


class GroupPartition(nn.Module):
    def __init__(self, partitions):
        super(GroupPartition, self).__init__()
        self.partitions = partitions
        self.shape = None

    def partition(self, x): # B HW C
        self.shape = x.shape
        x = x.reshape(-1, x.shape[-1] // self.partitions)
        return x

    def unpartition(self, x): # L C
        x = x.reshape(self.shape)
        return x


class LayerPartition(nn.Module):
    def __init__(self, partitions):
        super(LayerPartition, self).__init__()
        self.partitions = partitions
        self.shape = None

    def partition(self, x): # B HW C
        x = x.permute(0, 3, 1, 2).contiguous()
        self.shape = x.shape
        x = x.reshape(-1,  x.shape[-1] * x.shape[-2] // self.partitions)
        return x

    def unpartition(self, x): # L HW
        x = x.reshape(self.shape)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class CustomPartition(nn.Module):
    def __init__(self, partitions):
        super(CustomPartition, self).__init__()
        self.partitions = partitions
        self.shape = None

    def partition(self, x):
        assert x.shape[-1] % 2 == 0
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2, 2)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        self.shape = x.shape
        x = x.reshape(-1, x.shape[-1] * 2 // self.partitions)
        return x

    def unpartition(self, x):
        x = x.reshape(self.shape)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        return x
