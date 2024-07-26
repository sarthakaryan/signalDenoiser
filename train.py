import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from network import DPATD
from dataset import load_CleanNoisyPairDataset
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from util import find_max_epoch, loss_fn


# Train function
def train(num_gpus, rank, group_name, exp_path, log, optimization, network_config, trainset_config, dist_config):

    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)
    
    # Create tensorboard logger
    log_directory = os.path.join(log["directory"], exp_path)
    if rank == 0:
        tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # load training data
    trainloader = load_CleanNoisyPairDataset(**trainset_config, 
                            subset='training',
                            batch_size=optimization["batch_size_per_gpu"], 
                            num_gpus=num_gpus)
    print('Data loaded')
    
    # predefine model
    model = DPATD(**network_config).cuda()
    print_size(model)

    # apply gradient all reduce
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimization["learning_rate"])

    # load checkpoint
    time0 = time.time()
    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    else:
        ckpt_iter = log["ckpt_iter"]
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # feed model dict and optimizer state
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
        except:
            print('Something went wrong when loading model at iteration %s' % ckpt_iter)
            raise


    # training loop
    for i in range(max(0, ckpt_iter), optimization["max_epoch"]):
        model.train()
        for num, X in enumerate(trainloader):
            # feed input
            X = [x.cuda(non_blocking=True) for x in X[:2]]
            
            # zero optim gradients
            optimizer.zero_grad()

            # apply model to input
            loss, loss_dic = loss_fn(model, X)

            # optimize loss
            loss.backward()
            optimizer.step()

            # reduce losses over all gpus for logging purposes
            if num_gpus > 1:
                loss_reduced = reduce_tensor(loss.data, num_gpus).item()
            else:
                loss_reduced = loss.item()

            # print info
            if rank == 0:
                if num % 10 == 0:
                    print('iteration %d, epoch %d: loss %.6f' % (num, i, loss_reduced))

                # record loss
                if tb:
                    tb.add_scalar('loss_train_total', loss_reduced, i * len(trainloader) + num)
                    for key in loss_dic:
                        tb.add_scalar(key, loss_dic[key], i * len(trainloader) + num)

        # save model
        if rank == 0:
            checkpoint_path = os.path.join(ckpt_directory, '%d.pkl' % (i+1))
            torch.save({
                'iteration': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_time_seconds': time.time()-time0
                }, checkpoint_path)
            print('model at iteration %s is saved' % (i+1))
            
def print_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {num_params / 1e6:.2f} million parameters")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech enhancement")
    parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus for DDP training')
    parser.add_argument('--rank', default=0, type=int, help='global rank of DDP training')
    parser.add_argument('--group_name', default='group_name', type=str, help='process group name for DDP training')
    parser.add_argument('--config', default='config.json', type=str, help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    torch.backends.cudnn.benchmark = True

    train(args.num_gpus, args.rank, args.group_name, **config)
