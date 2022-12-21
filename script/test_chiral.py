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

from script.chiro import evaluate_binary_ranking_regression_loop_alpha, get_ranking_accuracies

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


def load_model(model, model_path):
    # from finetune_qm9
    # needed to load the pretrained weights
    state = torch.load(model_path, map_location=device)
    if 'model' in list(state.keys()):
        state = state['model']
    new_dict = OrderedDict()
    for k, v in state.items():
        if k.startswith('module.model.'):
            new_dict[k[13:]] = v
        if k.startswith('model.'):
            new_dict[k[6:]] = v
        # if k.startswith('module.node_dec.'):
        #     new_dict[k[7:]] = v
    model.load_state_dict(new_dict, strict=False)
    return new_dict

def test(rank, config, world_size=1, chkpt:str='', verbose=1):
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print('Rank: ', rank)
    if rank != 0:
        verbose = 0

    test_start = time()
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

    with open('/data/people/kabeywar/datasets/chiral/docking/test_small_enantiomers_stable_full_screen_docking_MOL_margin3_50571_10368_5184.pkl', 'rb') as f:
        test_df = pickle.load(f)
    test_df = pd.DataFrame(test_df)
    # test_df.sort_values(['SMILES_nostereo'])

    test_dataset = Dataset_3D_GNN(test_df, transforms=transform)

    if world_size > 1:
        data_sampler = DistributedSampler(test_dataset, world_size, rank, drop_last=True) if world_size > 1 else None
        dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, sampler=data_sampler, shuffle=(data_sampler is None))
    else:
        dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=True)

    print('Loader done')

    edge_types = 0 if config.model.no_edge_types else config.model.order + 1

    model = layers.EGNN_finetune_last_drugs(in_node_nf=config.model.max_atom_type*(config.model.charge_power + 1), in_edge_nf=edge_types, hidden_nf=config.model.hidden_dim, n_layers=config.model.n_layers, attention=config.model.attention, use_layer_norm=config.model.layernorm)

    if config.test.restore_path:
        checkpoint_path = os.path.join(config.test.restore_path, config.model.name, chkpt)
        encoder_param = load_model(model, checkpoint_path)
        print('Loading model from', checkpoint_path)
    
    model = model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True) 
    print('Start Testing')

    with torch.no_grad():
        targets, outputs = evaluate_binary_ranking_regression_loop_alpha(rank, config, world_size, model, dataloader, len(test_dataset))

    results_df = deepcopy(test_df[['ID', 'SMILES_nostereo', 'top_score']])
    results_df['targets'] = targets
    results_df['outputs'] = outputs
        
    margins, ranking_accuracy, random_baseline_means, random_baseline_stds = get_ranking_accuracies(results_df, mode = config.test.margin)
        
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print(f'Checkpoint: {chkpt}')
        print(f'optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - test_start))

    print(f'\tMargins: \t\t{margins}\n\tRanking Accuracy: \t{ranking_accuracy}\n\tRandom Baseline Means: \t{random_baseline_means}\n\tRandom Baseline Stddev: \t{random_baseline_stds}')
    save_path = os.path.join(config.test.save_path, chkpt, 'test_results.npz')
    print(f'\nSaved to {save_path}\n')
    np.savez(save_path, margins=margins, ranking_accuracy=ranking_accuracy, random_baseline_means=random_baseline_means, random_baseline_stds=random_baseline_stds)

    return margins, ranking_accuracy, random_baseline_means, random_baseline_stds

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def main():
    # This is a modified version of the main function from pretrain_3dmgp.py

    torch.set_printoptions(profile="full")

    parser = argparse.ArgumentParser(description='Docking Ranking Test')
    parser.add_argument('--config_path', type=str, default='.', metavar='N',
                        help='Path of config yaml.')
    parser.add_argument('--model_name', type=str, default='', metavar='N',
                        help='Model name.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.model_name != '':
        config.model.name = args.model_name

    if config.test.save_path is not None:
        config.test.save_path = os.path.join(config.test.save_path, config.model.name)
        if not os.path.exists(config.test.save_path):
            os.makedirs(config.test.save_path, exist_ok=True)
        if config.test.margin is not None:
            if config.test.margin == '>=':
                margin = 'geq'
            elif config.test.margin == '<=':
                margin = 'leq'
            elif config.test.margin == '==':
                margin = 'eq'
            else:
                NameError('config.test.margin must be one of these: ">=", "<=", "==".')
            marginpath = os.path.join(config.test.save_path, margin)
            config.test.save_path = os.path.join(config.test.save_path, margin)
            if not os.path.exists(marginpath):
                os.makedirs(marginpath, exist_ok=True)

    print(config)

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    for chkpt in config.test.checkpoints:
        if not os.path.exists(os.path.join(config.test.save_path, chkpt)):
            os.makedirs(os.path.join(config.test.save_path, chkpt), exist_ok=True)
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ["MASTER_PORT"] = str(get_free_port())
            mp.spawn(test, args=(config, world_size), nprocs=world_size)
        else:
            test(args.local_rank, config, world_size, chkpt)


def init_process(rank, world_size, func, config):
    """ Initialize the distributed environment. """
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f'Rank: {rank} - Distribution was initialized: {torch.distributed.is_initialized()}')
    func(rank, config, world_size)

if __name__ == '__main__':
    main()