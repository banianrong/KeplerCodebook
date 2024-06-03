import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

def kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q + 1e-5), dim=(1)).mean()

class KeplerLoss(nn.Module):
    def __init__(self, use, kl_weight,n_e):
        super(KeplerLoss, self).__init__()
        self.use = use
        self.kl_weight=kl_weight
        self.n_e=n_e

        self.prior_prob=self.create_high_dimensional_grid()

    def create_high_dimensional_grid(self):
        num_points = 2048
        dimensions = 64
        sub_dimensions = 16

        points_per_dim = int(np.ceil(num_points ** (1 / sub_dimensions)))
        low_dim_grid = np.indices([points_per_dim] * sub_dimensions).reshape(sub_dimensions, -1).T
        high_dim_grid = np.tile(low_dim_grid, (1, dimensions // sub_dimensions))
    
        if high_dim_grid.shape[0] < num_points:
            raise ValueError("The number of points in the grid is insufficient to generate the required number of sphere centers.")
    
        start_index = np.random.randint(0, high_dim_grid.shape[0] - num_points + 1)
        selected_points = high_dim_grid[start_index:start_index + self.n_e, :]
    
        return torch.from_numpy(selected_points).float()
    
        
    def forward(self, z):
        p = z.view(z.shape[0], -1)  
        p = (p - p.mean(dim=1, keepdim=True)) / p.var(dim=1, keepdim=True)
        p = F.softmax(p, dim=1) 

        q = self.prior_prob.reshape(1, -1).repeat(p.shape[0], 1)  
        q = (q - q.mean(dim=1, keepdim=True)) / q.var(dim=1, keepdim=True)
        q = F.softmax(q[:, :p.shape[1]], dim=1).to(p.device)
        kl_loss = kl_divergence(p, q) * self.kl_weight  
        return kl_loss
