from random import sample
import torch
from torch.nn import Module
import os
import numpy as np

from .common import *
from .encoders import *
from .diffusion import *
from .flow import *


class MyVAE(Module):

    def __init__(self, args, point_clouds=None, seed=2023, sample_num=2048):
        super().__init__()
        np.random.seed(seed)
        self.sample_num = sample_num
        self.args = args
        if point_clouds == None:
            self.read_point_clouds()
        else:
            self.point_clouds = point_clouds.to(self.args.device)
        self.points_num = self.point_clouds.shape[1]
        self.pcd_index = np.zeros(self.points_num, dtype=bool)
        self.pcd_index[np.random.choice(np.arange(self.points_num), self.sample_num, replace=False)] = True
        self.sampled_pcd = self.point_clouds[:,self.pcd_index]

        self.zs = nn.Parameter(torch.rand((self.point_clouds.shape[0], args.latent_dim), requires_grad=True))
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def enlarge_sample_num(self):
        if self.sample_num < self.points_num:
            increase_num = min(self.sample_num, self.points_num - self.sample_num)
            self.pcd_index[np.random.choice(np.where(self.pcd_index==False)[0], increase_num, replace=False)] = True
            self.sampled_pcd = self.point_clouds[:,self.pcd_index]
            self.sample_num += increase_num
        
        return
    
    def read_point_clouds(self):
        self.point_clouds = []
        for path in os.listdir(self.args.src_dir):
            # append tensor of shape [1, N, 3]
            self.point_clouds.append(torch.load(os.path.join(self.args.src_dir, path)).unsqueeze(0))
        self.point_clouds = torch.cat(self.point_clouds, 0).to(self.args.device)

        return

    def get_loss(self):
        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion.get_loss(self.sampled_pcd, self.zs)
        # Loss
        loss = neg_elbo

        return loss

    def sample(self):
        samples = self.diffusion.sample(self.sample_num, context=self.zs, flexibility=self.args.flexibility)

        return samples
