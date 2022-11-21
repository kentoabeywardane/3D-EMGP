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
from torch_geometric.data import Data
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


def train(rank, config, world_size, verbose=1):

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

    with open(os.path.join(data_dir,'summary.json'),'r') as f:
        summ = json.load(f)
    
    ###
    #Kento: added '- 1' b/c the last train block is not the same block size as the others
    # based on how I modified geom_drugs.py's gen_GEOM_blocks func
    # So we do not use the last train block...
    # I was getting an AssertionError in "d.load_datapoints()":
    ## File "./data/dataset.py", line XX, in load_datapoints
    ##      assert len(self.datapooints) == self.n_samples
    train_block_num = summ['train block num'] - 1 
    ###
    train_block_size = summ['train block size']
    val_block_size = summ['val block size']
    val_block = BatchDatapoint(os.path.join(data_dir,'val_block.pkl'),val_block_size)
    val_block.load_datapoints()
    val_dataset = GEOMDataset([val_block],val_block_size, transforms=transform)

    train_blocks = [BatchDatapoint(os.path.join(data_dir,'train_block_%d.pkl'%i),train_block_size) for i in range(train_block_num)]

    for d in train_blocks:
        d.load_datapoints()

    train_dataset = GEOMDataset(train_blocks, train_block_size, transforms=transform)

    edge_types = 0 if config.model.no_edge_types else config.model.order + 1

    ### We use EGNN_finetune_last instead of models.EquivariantDenoisePred for the model
    # This uses EGNN backbone with a final output layer with dim=1
    model = layers.EGNN_finetune_last_drugs(in_node_nf=config.model.max_atom_type * (config.model.charge_power + 1),
                                        in_edge_nf=edge_types, hidden_nf=config.model.hidden_dim, n_layers=config.model.n_layers,
                                        attention=config.model.attention, use_layer_norm = config.model.layernorm)
    ###

    # checkpoint model
    if config.train.restore_path:
        encoder_param = load_model(model, config.train.restore_path)
        print('load model from', config.train.restore_path)

    num_epochs = config.train.epochs
    if world_size == 1:
        dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                        num_workers=config.train.num_workers, pin_memory = False)
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                        rank=rank)
        dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                sampler=train_sampler, num_workers=config.train.num_workers, pin_memory = False)

    valloader = DataLoader(val_dataset, batch_size=config.train.batch_size, \
                            shuffle=False, num_workers=config.train.num_workers)

    model = model.to(rank)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True) 
    optimizer = optim.Adam([param for name, param in model.named_parameters()], lr=config.train.lr, weight_decay=float(config.train.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.train.epochs, eta_min = float(config.train.min_lr))
    train_losses = []
    val_losses = []
    ckpt_list = []
    max_ckpt_maintain = 10
    best_loss = 100.0
    start_epoch = 0
    
    print(f'Rank {rank} start training...')
    L1_loss = torch.nn.L1Loss()

    # train epochs
    for epoch in range(num_epochs):
        # train
        if world_size>1:
            train_sampler.set_epoch(epoch)
        model.train() # training mode
        optimizer.zero_grad()
        epoch_start = time()
        batch_losses = []
        batch_cnt = 0

        for batch in dataloader:
            batch_cnt += 1
            batch = batch.to(rank)
            
            ### Energy Prediction
            # similar to denoise_prednoise.py
            node2graph = batch.batch
            # edge2graph = node2graph[batch.edge_index[0]]
            pos = batch.pos
            edge_attr = gen_edge_onehot(config, batch.edge_type)
            num_nodes = batch.num_nodes

            nrg_pred = model(batch.node_feature, pos, batch.edge_index, edge_attr, num_nodes, node2graph)

            # design loss function
            label = batch[config.train.property].to(device, dtype) # label --> training target
            
            loss = L1_loss(nrg_pred, label)
            ###

            # back to pretrain_3dmgp recipe
            loss.backward()
            # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip)
            # if not norm.isnan():
            #     optimizer.step()
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

        if verbose:
            print('Epoch: %d | Train Loss: %.5f | Time: %.5f' % (epoch + start_epoch, average_loss, time() - epoch_start))
        
        scheduler.step()

        if world_size > 1:
            dist.barrier()

        # validation
        model.eval() # evaluation mode --> ie. disable dropout
        eval_start = time()
        eval_losses = []
        with torch.no_grad(): # disables the calculation of gradient graphs and possibility of backprop --> also saves memory
            for batch in valloader:
                batch = batch.to(rank)  
                ### Energy Prediction
                # similar to denoise_prednoise.py
                node2graph = batch.batch
                # edge2graph = node2graph[batch.edge_index[0]]
                pos = batch.pos
                edge_attr = gen_edge_onehot(config, batch.edge_type)
                num_nodes = batch.num_nodes

                nrg_pred = model(batch.node_feature, pos, batch.edge_index, edge_attr, num_nodes, node2graph)
                label = batch[config.train.property].to(device, dtype) # label --> training target
                loss = L1_loss(nrg_pred, label)
                ### 
                eval_losses.append(loss.item())
        average_loss = sum(eval_losses) / len(eval_losses)

        if rank == 0:
            print('Evaluate val Loss: %.5f | Time: %.5f' % (average_loss, time() - eval_start))
            
            val_losses.append(average_loss)

            if epoch % 10 == 0: 
                fname = os.path.join(config.train.save_train_val, config.train.save_val_file)
                with open(fname, 'wb') as f:
                    np.save(f, np.array(val_losses))

            # if val_losses[-1] < best_loss: # best_loss is 100 initially
            #     best_loss = val_losses[-1]
            # save at every 10 epochs in case it crashes
            if epoch % 10 == 0 or epoch == 0:
                if config.train.save:
                    state = {
                        "model": model.state_dict(),
                        "config": config,
                        'cur_epoch': epoch + start_epoch,
                        # 'best_loss': best_loss,
                        'best_loss': val_losses[-1],
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "avg_train_loss": train_losses[-1]
                    }
                    epoch = str(epoch) if epoch is not None else ''
                    # save checkpoints of model for each epoch
                    checkpoint = os.path.join(config.train.save_path ,'checkpoint%s' % epoch)
                    # if len(ckpt_list) >= max_ckpt_maintain:
                    #     try:
                    #         os.remove(ckpt_list[0])
                    #     except:
                    #         print('Remove checkpoint failed for', ckpt_list[0])
                    #     ckpt_list = ckpt_list[1:]
                    #     ckpt_list.append(checkpoint)
                    # else:
                    #     ckpt_list.append(checkpoint)

                    torch.save(state, checkpoint)

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        best_loss = best_loss
        start_epoch = start_epoch + num_epochs               
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))
    if world_size > 1:
        dist.destroy_process_group()


def main():
    # This is a modified version of the main function from pretrain_3dmgp.py

    torch.set_printoptions(profile="full")

    parser = argparse.ArgumentParser(description='Drugs Finetune')
    parser.add_argument('--config_path', type=str, default='.', metavar='N',
                        help='Path of config yaml.')
    parser.add_argument('--property', type=str, default='', metavar='N',
                        help='Property to predict.')
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

    if args.property != '':
        config.train.property = args.property

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

    print(dist.is_available())
    if world_size > 1:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group('nccl', rank=args.local_rank, world_size=world_size)
    # print(f'Was dist initialized: {torch.distributed.is_initialized()}')

    # Finetune!
    train(args.local_rank, config, world_size)

if __name__ == '__main__':
    main()