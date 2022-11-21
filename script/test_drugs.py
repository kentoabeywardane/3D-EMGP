import argparse
import numpy as np
import random
import os
import pickle
import yaml
from easydict import EasyDict

import torch
import sys
sys.path.append('.')
from torch_geometric.transforms import Compose
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from mgp import utils, layers, models
from torch_geometric.nn import DataParallel
import torch.multiprocessing as mp
from time import time
from collections import OrderedDict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.nn.functional as F # from denoise_prednoise.py
from torch import optim


from data.dataset import BatchDatapoint, GEOMDataset, AtomOnehot, EdgeHop, Cutoff
import json

mp.set_sharing_strategy('file_system')

device = torch.device("cuda")
dtype = torch.float32


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

def test(rank, config, world_size, chkpt:int, verbose=1):
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

    # set random seed
    # np.random.seed(config.test.seed)
    # random.seed(config.test.seed)
    # torch.manual_seed(config.test.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(config.test.seed)        
    #     torch.cuda.manual_seed_all(config.test.seed)
    torch.backends.cudnn.benchmark = True

    data_dir = config.data.block_dir

    with open(os.path.join(data_dir, 'summary.json'), 'r') as f:
        summ = json.load(f)

    test_block_num = summ['test block num'] - 1
    test_block_size = summ['test block size']
    test_blocks = [BatchDatapoint(os.path.join(data_dir,'test_block_%d.pkl'%i), test_block_size) for i in range(test_block_num)]

    for d in test_blocks:
        d.load_datapoints()

    test_dataset = GEOMDataset(test_blocks, test_block_size, transforms=transform)

    edge_types = 0 if config.model.no_edge_types else config.model.order + 1

    model = layers.EGNN_finetune_last_drugs(in_node_nf=config.model.max_atom_type * (config.model.charge_power + 1),
                                        in_edge_nf=edge_types, hidden_nf=config.model.hidden_dim, n_layers=config.model.n_layers,
                                        attention=config.model.attention, use_layer_norm = config.model.layernorm)
    
    load_file = f'checkpoint{str(chkpt)}'
    load_path = os.path.join(config.test.restore_path, load_file)
    encoder_param = load_model(model, load_path)
    print('load model from', load_file)

    if world_size == 1:
        dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False,
                            num_workers=config.test.num_workers, pin_memory = False)
    else:
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,
                                        rank=rank)
        dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False,
                                sampler=test_sampler, num_workers=config.test.num_workers, pin_memory = False)

    model = model.to(rank)
    if world_size>1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    print(f'Rank {rank} start testing for checkpoint {chkpt}...')
    L1_loss = torch.nn.L1Loss(reduction='mean')

    model.eval()
    start_time = time()
    batch_losses = []
    all_smiles = []
    all_labels = []
    all_losses = []
    all_preds = []
    batch_cnt = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_cnt += 1
            batch = batch.to(rank)

            # Energy Prediction
            node2graph = batch.batch
            # edge2graph = node2graph[batch.edge_index[0]]
            pos = batch.pos
            edge_attr = gen_edge_onehot(config, batch.edge_type)
            num_nodes = batch.num_nodes

            # predict
            nrg_pred = model(batch.node_feature, pos, batch.edge_index, edge_attr, num_nodes, node2graph)
            label = batch[config.test.property].to(device, dtype) # label --> test target

            all_smiles.extend(batch['smiles']) # maybe save to file as we go?
            all_labels.extend(label.tolist())
            all_preds.extend(nrg_pred.tolist())
            all_losses.extend(torch.subtract(label, nrg_pred).view(1, -1).tolist()) # returns a tensor {l1, ..., lN}'
            loss = L1_loss(nrg_pred, label)

            batch_losses.append(loss.item())

            if verbose and (batch_cnt % config.test.log_interval == 0) or batch_cnt <= 10: 
                print('Step: %d | Loss: %.5f '% (batch_cnt, batch_losses[-1])) 

    average_loss = sum(batch_losses) / len(batch_losses)
    variance = sum([((x - average_loss) ** 2) for x in batch_losses]) / len(batch_losses)
    stddev = variance ** 0.5

    print(f'Testing finished for checkpoint {chkpt}')
    if verbose:
        print('Average Test Loss: %.5f | Loss Stddev: %.5f | Total Time: %.5f'% (average_loss, stddev, time() - start_time))

    if world_size > 1:
        dist.destroy_process_group()
    
    # need a way to save the rdmols and losses to files
    d = {'smiles' : all_smiles,
         'labels' : all_labels,
         'preds' : all_preds,
         'losses': all_losses,
         'config': config,
         'checkpoint': chkpt}
    filen = os.path.join(config.test.save_path, f'chkpt{chkpt}.p')
    with open(filen, 'wb') as fp:
        pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    # This is a modified version of the main function from finetune_drugs.py

    torch.set_printoptions(profile="full")

    parser = argparse.ArgumentParser(description='Drugs Test')
    parser.add_argument('--config_path', type=str, default='.', metavar='N',
                        help='Path of config yaml.')
    parser.add_argument('--property', type=str, default='totalenergy', metavar='N',
                        help='Property to predict.')
    parser.add_argument('--model_name', type=str, default='', metavar='N',
                        help='Model name.')
    parser.add_argument('--restore_path', type=str, default='', metavar='N',
                        help='Restore path for a checkpoint.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.property != 'totalenergy':
        config.test.property = args.property

    if args.model_name != '':
        config.model.name = args.model_name

    if args.restore_path != '':
        config.test.restore_path = args.restore_path
    
    if config.test.save_path is not None:
        config.test.save_path = os.path.join(config.test.save_path, config.model.name)
        if not os.path.exists(config.test.save_path):
            os.makedirs(config.test.save_path, exist_ok=True)

    print(config)

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    print(dist.is_available())
    if world_size > 1:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group('nccl', rank=args.local_rank, world_size=world_size)
    # print(f'Was dist initialized: {torch.distributed.is_initialized()}')

    # Test!
    for chkpt in config.test.checkpoints:
        test(args.local_rank, config, world_size, chkpt)

if __name__ == '__main__':
    main()