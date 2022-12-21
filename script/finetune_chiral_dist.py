import socket
import argparse
import numpy as np
import random
import os
import pickle5 as pickle
import yaml
from easydict import EasyDict

import torch
import sys
sys.path.append('.')
from torch_geometric.transforms import Compose
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from mgp import utils, layers, models
import torch.multiprocessing as mp
from time import time
from collections import OrderedDict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F # from denoise_prednoise.py
from torch import optim
import torch_geometric
from copy import deepcopy
import pandas as pd
from rdkit.Chem.rdchem import BondType

from data.dataset import BatchDatapoint, GEOMDataset, AtomOnehot, EdgeHop, Cutoff
import json

from data.geom_drugs import rdmol_to_data

BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

mp.set_sharing_strategy('file_system')

device = torch.device("cuda")
dtype = torch.float32

class Dataset_3D_GNN(Dataset):
    ## THIS IS FROM Keir Adams --> ChIRo code
    # https://github.com/keiradams/ChIRo/blob/2686d3a1801db8fb3dec10b61fb0cb0cec047c7c/model/datasets_samplers.py#L56
    def __init__(self, df, transforms = None, regression = 'top_score'):
        super(Dataset_3D_GNN, self).__init__()
        self.df = df
        self.regression = regression
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, key):
        mol = deepcopy(self.df.iloc[key].rdkit_mol_cistrans_stereo)
        data = rdmol_to_data(mol)

        if self.transforms:
            data = self.transforms(data)
        
        y = torch.tensor(deepcopy(self.df.iloc[key][self.regression]))

        return (data, y) 


def gen_edge_onehot(config, edge_types):
    # from denoise_prednoise.py --> data processing
    config_edge_types = 0 if config.model.no_edge_types else config.model.order + 1
    if not config_edge_types:
        return None
    return F.one_hot(edge_types.long(), config_edge_types)


def load_model(model, model_path):
    # from finetune_qm9
    # needed to load the pretrained weights
    state = torch.load(model_path, map_location=device)
    new_dict = OrderedDict()
    for k, v in state['model'].items():
        if k.startswith('module.model.'):
            new_dict[k[13:]] = v
        if k.startswith('model.'):
            new_dict[k[6:]] = v
        # if k.startswith('module.node_dec.'):
        #     new_dict[k[7:]] = v
    model.load_state_dict(new_dict, strict=False)
    return new_dict


def forward_batch(config, model, rank, l1loss, batch):
    """Take a batch of data from dataloader and run a forward pass and compute loss."""
    # unpack batched data
    batch_data, label = batch
    batch_data = batch_data.to(rank)
    label = label.type(torch.float32)
    label = label.to(rank)

    # batched data attributes
    node2graph = batch_data.batch
    pos = batch_data.pos
    node_feature = batch_data.node_feature
    edge_index = batch_data.edge_index
    edge_attr = gen_edge_onehot(config, batch_data.edge_type)
    # forward pass
    pred = model(node_feature, pos, edge_index, edge_attr, node2graph)

    # loss
    loss = l1loss(pred.squeeze(), label.squeeze())

    return loss

def train(rank, config, world_size=1, verbose=1):
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print('Rank: ', rank)
    if rank != 0:
        verbose = 0

    train_start = time()

    if config.model.no_edge_types:
        transform = Compose([
            Cutoff(cutoff_length=config.model.cutoff),
            AtomOnehot(max_atom_type=config.model.max_atom_type, charge_power=config.model.charge_power),
            ])
    else:
        transform = Compose([
            EdgeHop(max_hop=config.model.order),
            AtomOnehot(max_atom_type=config.model.max_atom_type, charge_power=config.model.charge_power),
            ])

    # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)        
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True

    # set up data processing for train/val datasets
    data_dir = config.data.block_dir

    with open('/data/people/kabeywar/datasets/chiral/docking/train_small_enantiomers_stable_full_screen_docking_MOL_margin3_234622_48384_24192.pkl', 'rb') as f:
        train_df = pickle.load(f)
    train_df = pd.DataFrame(train_df)
    
    with open('/data/people/kabeywar/datasets/chiral/docking/validation_small_enantiomers_stable_full_screen_docking_MOL_margin3_49878_10368_5184.pkl', 'rb') as f:
        val_df = pickle.load(f)
    val_df = pd.DataFrame(val_df)

    train_dataset = Dataset_3D_GNN(train_df, transforms=transform)
    val_dataset = Dataset_3D_GNN(val_df, transforms=transform)

    if world_size > 1:
        data_sampler = DistributedSampler(train_dataset, world_size, rank, drop_last=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, world_size, rank, drop_last=True) if world_size > 1 else None
        dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=(data_sampler is None), sampler=data_sampler)
        valloader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=(val_sampler is None), sampler=val_sampler)

    edge_types = 0 if config.model.no_edge_types else config.model.order + 1

    model = layers.EGNN_finetune_last_drugs(in_node_nf=config.model.max_atom_type*(config.model.charge_power + 1), in_edge_nf=edge_types, hidden_nf=config.model.hidden_dim, n_layers=config.model.n_layers, attention=config.model.attention, use_layer_norm=config.model.layernorm)

    if config.train.restore_path:
        encoder_param = load_model(model, config.train.restore_path)
        print('Loading model from', config.train.restore_path)
    
    model = model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True) 

    optimizer = optim.Adam([param for name, param in model.named_parameters()], lr=config.train.lr, weight_decay=float(config.train.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.train.epochs, eta_min=float(config.train.min_lr))

    num_epochs = config.train.epochs

    train_losses = []
    val_losses = []
    
    start_epoch = 0

    L1_loss = torch.nn.L1Loss()
    best_loss = 100

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time()
        batch_losses = []
        batch_cnt = 0

        if world_size > 1:
            data_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        for batch in dataloader:
            batch_cnt += 1

            loss = forward_batch(config, model, rank, L1_loss, batch)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            # average_gradients(model)
            optimizer.step()
            batch_losses.append(loss.item())
            
            if verbose and (batch_cnt % config.train.log_interval == 0 or (epoch==0 and batch_cnt <= 10)):
                print('Epoch: %d | Step: %d | loss: %.5f| Lr: %.5f' % \
                        (epoch + start_epoch, batch_cnt, batch_losses[-1], optimizer.param_groups[0]['lr']))
            
        average_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(average_loss)

        if epoch % 10 == 0: 
            fname = os.path.join(config.train.save_train_val, config.train.save_train_file)
            with open(fname, 'wb') as f:
                np.save(f, np.array(train_losses))

        if verbose and world_size==1:
            print('Epoch: %d | Train Loss: %.5f | Time: %.5f' % (epoch + start_epoch, average_loss, time() - epoch_start))
        if world_size > 1:
            print('Rank', dist.get_rank(), ', epoch ', epoch, ': ', average_loss, ' | Time: ', time() - epoch_start)

        scheduler.step()

        if world_size > 1:
            dist.barrier()

        model.eval()
        eval_start = time()
        eval_losses = []
        with torch.no_grad():
            for batch in valloader:
                loss = forward_batch(config, model, rank, L1_loss, batch)
                eval_losses.append(loss.item())
            average_loss = sum(eval_losses) / len(eval_losses)

        if rank == 0:
            print('Evaluate val Loss: %.5f | Time: %.5f' % (average_loss, time() - eval_start))
            
            val_losses.append(average_loss)

            if epoch % 10 == 0: 
                fname = os.path.join(config.train.save_train_val, config.train.save_val_file)
                with open(fname, 'wb') as f:
                    np.save(f, np.array(val_losses))

            if epoch % 10 == 0 or epoch == 0 or loss.item() < best_loss:
                if loss.item() < best_loss:
                    best_loss = loss.item()

                if config.train.save:
                    state = {
                        "model": model.state_dict(),
                        "config": config,
                        'cur_epoch': epoch + start_epoch,
                        'best_loss': val_losses[-1] if loss.item() > best_loss else best_loss,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "avg_train_loss": train_losses[-1]
                    }
                    epoch = str(epoch) if epoch is not None else ''
                    # save checkpoints of model for each epoch
                    checkpoint = os.path.join(config.train.save_path ,'checkpoint%s' % epoch)

                    torch.save(state, checkpoint)

        if world_size > 1:
            dist.barrier()
    
    if rank == 0:
        start_epoch = start_epoch + num_epochs               
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))
    if world_size > 1:
        dist.destroy_process_group()


def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def main():
    # This is a modified version of the main function from pretrain_3dmgp.py

    torch.set_printoptions(profile="full")

    parser = argparse.ArgumentParser(description='Chiral with distribution')
    parser.add_argument('--config_path', type=str, default='.', metavar='N',
                        help='Path of config yaml.')
    parser.add_argument('--model_name', type=str, default='', metavar='N',
                        help='Model name.')
    parser.add_argument('--restore_path', type=str, default='', metavar='N',
                        help='Restore path.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.model_name != '':
        config.model.name = args.model_name

    if args.restore_path != '':
        config.train.restore_path = args.restore_path
    
    if args.epochs != config.train.epochs:
        config.train.epochs = args.epochs

    if config.train.save_path is not None:
        config.train.save_path = os.path.join(config.train.save_path, config.model.name)
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path, exist_ok=True)

    print(config)

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ["MASTER_PORT"] = str(get_free_port())
        mp.spawn(train, args=(config, world_size), nprocs=world_size)
    else:
        train(args.local_rank, config, world_size)


def init_process(rank, world_size, func, config):
    """ Initialize the distributed environment. """
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f'Rank: {rank} - Distribution was initialized: {torch.distributed.is_initialized()}')
    func(rank, config, world_size)

if __name__ == '__main__':
    main()
