import os
import math
import argparse
from sklearn.metrics import mean_absolute_percentage_error
import torch
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_trained_z import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--latent_dim', type=int, default=2048)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=400*THOUSAND)

# Training
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--increase_freq', type=int, default=20000)
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--src_dir', type=str, default='/content/drive/MyDrive/K-D_Tree_NeRF/diffusion-point-cloud/data/nerf')
args = parser.parse_args()

def plot(pts):
    np_pcd = pts.numpy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np_pcd[:,0], np_pcd[:,1], np_pcd[:,2])

    return fig

def train(model, optimizer, scheduler, it):
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss()

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    wandb.log({
        'iters': it,
        'learning rate': optimizer.param_groups[0]['lr'],
        'Grad': orig_grad_norm,
        'loss': loss.item()
    })

    return loss.item()

if __name__ == '__main__':
    wandb.init(config=args)
    # Model
    print('Building model...')
    model = MyVAE(args).to(args.device)
    print(repr(model))
    # if args.spectral_norm:
    #     add_spectral_norm(model, logger=logger)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr
    )

    print('Start training...')
    log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
    ckpt_mgr = CheckpointManager(log_dir)
    try:
        it = 0
        acc_loss = 0.0
        while it < args.max_iters:
            loss_val = train(model, optimizer, scheduler, it)
            acc_loss += loss_val
            if it % args.val_freq == args.val_freq - 1 or it == args.max_iters - 1:
                print('iteration: {}, loss = {}'.format(it, acc_loss / args.val_freq))
                acc_loss = 0.0
                # opt_states = {
                #     'optimizer': optimizer.state_dict(),
                #     'scheduler': scheduler.state_dict(),
                # }
                # ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
                with torch.no_grad():
                    samples = model.sample()[0].detach().cpu()
                fig = plot(samples)
                image = wandb.Image(fig, caption="generated point cloud")
                wandb.log({"gen_pcd": image})
            if it % args.increase_freq == args.increase_freq - 1:
                with torch.no_grad():
                    model.enlarge_sample_num()
            it += 1

    except KeyboardInterrupt:
        print('Terminating...')
