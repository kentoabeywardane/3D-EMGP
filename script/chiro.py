## All from Adams et al. 2021
# https://github.com/keiradams/ChIRo
# https://github.com/keiradams/ChIRo/blob/2686d3a1801db8fb3dec10b61fb0cb0cec047c7c/model/datasets_samplers.py#L261

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import sys
sys.path.append('.')
import math
import pandas as pd
import numpy as np

from copy import deepcopy
from itertools import chain
import torch.distributed as dist
import os
import random

class Sample_Map_To_Positives:
    def __init__(self, dataframe, isSorted=True, include_anchor = False): #isSorted vastly speeds up processing, but requires that the dataframe is sorted by SMILES_nostereo
        self.mapping = {}
        self.include_anchor = include_anchor
        
        for row_index, row in dataframe.iterrows():
            if isSorted:
                subset_df = dataframe.iloc[max(row_index-50, 0): row_index+50, :]
                
                if self.include_anchor == False:
                    positives = set(subset_df[(subset_df.ID == row.ID) & (subset_df.index.values != row_index)].index)
                else:
                    positives = set(subset_df[(subset_df.ID == row.ID)].index)
                
                self.mapping[row_index] = positives
                
    def sample(self, i, N=1, withoutReplacement=True): #sample positives
        if withoutReplacement:
            samples = random.sample(self.mapping[i], min(N, len(self.mapping[i])))
        else:
            samples = [random.choice(list(self.mapping[i])) for _ in range(N)]
        
        return samples

class Sample_Map_To_Negatives:
    def __init__(self, dataframe, isSorted=True): #isSorted vastly speeds up processing, but requires that the dataframe is sorted by SMILES_nostereo
        self.mapping = {}
        for row_index, row in dataframe.iterrows():
            if isSorted:
                negative_classes = []
                subset_df = dataframe.iloc[max(row_index-200, 0) : row_index+200, :]
                grouped_negatives = subset_df[(subset_df.SMILES_nostereo == row.SMILES_nostereo) & (subset_df.ID != row.ID)].groupby(by='ID', sort = False).groups.values()
                negative_classes = [set(list(group)) for group in grouped_negatives]
                self.mapping[row_index] = negative_classes
        
    def sample(self, i, N=1, withoutReplacement=True, stratified=True): #sample negatives
        if withoutReplacement:
            if stratified:
                samples = [random.sample(self.mapping[i][j], min(len(self.mapping[i][j]), N)) for j in range(len(self.mapping[i]))]
                samples = list(chain(*samples))
            else:
                population = list(chain(*[list(self.mapping[i][j]) for j in range(len(self.mapping[i]))]))
                samples = random.sample(population, min(len(population), N))
                
        else:
            if stratified:
                samples = [[random.choice(list(population)) for _ in range(N)] for population in self.mapping[i]]
                samples = list(chain(*samples))

            else:
                population = list(chain(*[list(self.mapping[i][j]) for j in range(len(self.mapping[i]))]))
                samples = [random.choice(population) for _ in range(N)]
            
        return samples

class SingleConformerBatchSampler(torch.utils.data.sampler.Sampler):
    # must be used with Sample_Map_To_Positives with include_anchor == True
    # Samples positives and negatives for each anchor, where the positives include the anchor
    
    # single_conformer_data_source is a dataframe consisting of just 1 conformer per stereoisomer
    # full_data_source is a dataframe consisting of all conformers for each stereoisomer
    # Importantly, single_conformer_data_source must be a subset of full_data_source, with the original indices
    
    def __init__(self, single_conformer_data_source, full_data_source, batch_size, N_pos = 0, N_neg = 1, withoutReplacement = True, stratified = True):
        self.single_conformer_data_source = single_conformer_data_source
        self.full_data_source = full_data_source
        
        self.positive_sampler = Sample_Map_To_Positives(full_data_source, include_anchor = True)
        self.negative_sampler = Sample_Map_To_Negatives(full_data_source)
        
        self.batch_size = batch_size
        self.withoutReplacement = withoutReplacement
        self.stratified = stratified
        self.N_pos = N_pos
        self.N_neg = N_neg
                
    def __iter__(self):
        groups = [[*self.positive_sampler.sample(i, N = 1 + self.N_pos, withoutReplacement = self.withoutReplacement), *self.negative_sampler.sample(i, N = self.N_neg, withoutReplacement = self.withoutReplacement, stratified = self.stratified)] for i in self.single_conformer_data_source.index.values]
        
        np.random.shuffle(groups)
        batches = [list(chain(*groups[self.batch_size*i:self.batch_size*i+self.batch_size])) for i in range(math.floor(len(groups)/self.batch_size))]
        return iter(batches)

    def __len__(self): # number of batches
        return math.floor(len(self.full_data_source) / self.batch_size) #drops the last batch if it doesn't contain batch_size anchors


def gen_edge_onehot(config, edge_types):
    # from denoise_prednoise.py --> data processing
    config_edge_types = 0 if config.model.no_edge_types else config.model.order + 1
    if not config_edge_types:
        return None
    return F.one_hot(edge_types.long(), config_edge_types)

def MSE(y, y_hat):
    MSE = torch.mean(torch.square(y - y_hat))
    return MSE

def binary_ranking_regression_loop_alpha(rank, config, model, loader, optimizer, training = True, absolute_penalty = 1.0, relative_penalty = 0.0, ranking_margin = 0.3):
    if training:
        model.train()
    else:
        model.eval()

    batch_losses = []
    batch_rel_losses = []
    batch_abs_losses = []
    batch_sizes = []
    
    batch_acc = []
    
    for batch in loader:
        # unpack batched data
        batch_data, y = batch
        batch_data = batch_data.to(rank)
        y = y.type(torch.float32)
        y = y.to(rank)

        # batched data attributes
        node2graph = batch_data.batch
        pos = batch_data.pos
        node_feature = batch_data.node_feature
        edge_index = batch_data.edge_index
        edge_attr = gen_edge_onehot(config, batch_data.edge_type)
        # forward pass
        output = model(node_feature, pos, edge_index, edge_attr, node2graph)

        if training:
            optimizer.zero_grad()

        loss_absolute = MSE(y.squeeze(), output.squeeze()) # plain MSE loss
        
        criterion = nn.MarginRankingLoss(margin=ranking_margin)
        
        #used in conjunction with negative batch sampler, where the negative immediately follows each anchor
        # notice that we treat the less negative score as being ranked "higher"
        loss_relative = criterion(output[0::2].squeeze(), output[1::2].squeeze(), torch.sign((y[0::2].squeeze() - y[1::2].squeeze()) + 1e-8).squeeze())
                
        backprop_loss = (loss_relative*relative_penalty) + (loss_absolute*absolute_penalty)
        
        if training:
            backprop_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
        
        # return  (binary) ranking accuracies, using margin = 0.1 and equivalence = 0.0
        target_ranking = ((torch.round(y[0::2].squeeze() * 100.) / 100.) > (torch.round(y[1::2].squeeze() * 100.) / 100.)).type(torch.float)
        output_ranking = ((torch.round(output[0::2].squeeze() * 100.) / 100.) > (torch.round(output[1::2].squeeze() * 100.) / 100.)).type(torch.float)
        top_1_acc = torch.sum(output_ranking == target_ranking) / float(output_ranking.shape[0])
        
        batch_acc.append(top_1_acc.item())
        
        batch_sizes.append(y.shape[0])
        batch_losses.append(backprop_loss.item())
        
        batch_rel_losses.append(loss_relative.item())
        batch_abs_losses.append(loss_absolute.item())
        
    return batch_losses, batch_sizes, batch_abs_losses, batch_rel_losses, batch_acc

# https://github.com/keiradams/ChIRo/blob/2686d3a1801db8fb3dec10b61fb0cb0cec047c7c/model/train_models.py#L11
def train_binary_ranking_regression_model(rank, config, world_size, model, train_loader, val_loader, train_sampler, val_sampler,  N_epochs, optimizer, absolute_penalty = 1.0, relative_penalty = 0.0, ranking_margin = 0.3, weighted_sum = False, save = True, PATH = ''):
    train_epoch_losses = []
    train_epoch_abs_losses = []
    train_epoch_rel_losses = []
    train_epoch_accuracies = []
    
    val_epoch_losses = []
    val_epoch_abs_losses = []
    val_epoch_rel_losses = []
    val_epoch_accuracies = []

    best_val_acc = 0.0
    best_val_loss = np.inf
    best_epoch = 0

    for epoch in range(1, N_epochs+1):
        if world_size>1:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
    
        train_losses, train_batch_sizes, train_abs_losses, train_rel_losses, train_accuracies = binary_ranking_regression_loop_alpha(rank, config, model, train_loader, optimizer, training = True, absolute_penalty = absolute_penalty, relative_penalty = relative_penalty, ranking_margin = ranking_margin)
        
        if world_size> 1:
            dist.barrier()

        if weighted_sum:
            epoch_loss = torch.sum(torch.tensor(train_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes))) #weighted mean based on the batch sizes
            epoch_loss /= world_size
            epoch_abs_loss = torch.sum(torch.tensor(train_abs_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
            epoch_abs_loss /= world_size
            epoch_rel_loss = torch.sum(torch.tensor(train_rel_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
            epoch_rel_loss /= world_size
            epoch_acc = torch.sum(torch.tensor(train_accuracies) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
            epoch_acc /= world_size
        else:
            epoch_loss = torch.mean(torch.tensor(train_losses))
            epoch_abs_loss = torch.mean(torch.tensor(train_abs_losses))
            epoch_rel_loss = torch.mean(torch.tensor(train_rel_losses))
            epoch_acc = torch.mean(torch.tensor(train_accuracies))

        epoch_loss = epoch_loss.to_dense().to(rank)
        epoch_abs_loss = epoch_abs_loss.to_dense().to(rank)
        epoch_rel_loss = epoch_rel_loss.to_dense().to(rank)
        epoch_acc = epoch_acc.to_dense().to(rank)

        if world_size > 1:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_abs_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_rel_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_acc, op=dist.ReduceOp.SUM)
            
        train_epoch_losses.append(epoch_loss)
        train_epoch_abs_losses.append(epoch_abs_loss)
        train_epoch_rel_losses.append(epoch_rel_loss)
        train_epoch_accuracies.append(epoch_acc)
        
        with torch.no_grad():
            val_losses, val_batch_sizes, val_abs_losses, val_rel_losses, val_accuracies = binary_ranking_regression_loop_alpha(rank, config, model, val_loader, optimizer, training = False, absolute_penalty = absolute_penalty, relative_penalty = relative_penalty, ranking_margin = ranking_margin)
            
            if world_size > 1:
                dist.barrier()
            
            if weighted_sum:
                val_epoch_loss = torch.sum(torch.tensor(val_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes))) #weighted mean based on the batch sizes
                val_epoch_loss /= world_size
                val_epoch_abs_loss = torch.sum(torch.tensor(val_abs_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
                val_epoch_abs_loss /= world_size
                val_epoch_rel_loss = torch.sum(torch.tensor(val_rel_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
                val_epoch_rel_loss /= world_size
                val_epoch_acc = torch.sum(torch.tensor(val_accuracies) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
                val_epoch_acc /= world_size
            else:
                val_epoch_loss = torch.mean(torch.tensor(val_losses))
                val_epoch_abs_loss = torch.mean(torch.tensor(val_abs_losses))
                val_epoch_rel_loss = torch.mean(torch.tensor(val_rel_losses))
                val_epoch_acc = torch.mean(torch.tensor(val_accuracies))
            
            val_epoch_loss = val_epoch_loss.to_dense().to(rank)
            val_epoch_abs_loss = val_epoch_abs_loss.to_dense().to(rank)
            val_epoch_rel_loss = val_epoch_rel_loss.to_dense().to(rank)
            val_epoch_acc = val_epoch_acc.to_dense().to(rank)
            
            if world_size > 1:
                dist.all_reduce(val_epoch_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_epoch_abs_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_epoch_rel_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_epoch_acc, op=dist.ReduceOp.SUM)
                
            val_epoch_losses.append(val_epoch_loss)
            val_epoch_abs_losses.append(val_epoch_abs_loss)
            val_epoch_rel_losses.append(val_epoch_rel_loss)
            val_epoch_accuracies.append(val_epoch_acc)
            
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if rank == 0 and save == True:
                    checkpt_path = os.path.join(config.train.save_path,'best_model')
                    torch.save(model.state_dict(), checkpt_path)
                    print('\n    saving best model:' + str(epoch))
                    print('    Best Epoch:', best_epoch, 'Train Loss:', epoch_loss, 'Train Acc.:', epoch_acc,'Validation Loss:', val_epoch_loss, 'Validation Acc.:', val_epoch_acc)
                    print('        Train Losses (abs., rel.):', (epoch_abs_loss, epoch_rel_loss), 'Validation Losses (abs., rel.):', (val_epoch_abs_loss, val_epoch_rel_loss))

            if rank == 0 and epoch % 5 == 0:
                print('Epoch:', epoch, 'Train Loss:', epoch_loss, 'Train Acc.:', epoch_acc,'Validation Loss:', val_epoch_loss, 'Validation Acc.:', val_epoch_acc)
                print('        Train Losses (abs., rel.):', (epoch_abs_loss, epoch_rel_loss), 'Validation Losses (abs., rel.):', (val_epoch_abs_loss, val_epoch_rel_loss))
                if (save == True) and (epoch % 10 == 0):
                    torch.save(model.state_dict(), os.path.join(config.train.save_path, 'checkpoint' + str(epoch)))
                    torch.save(train_epoch_losses, os.path.join(PATH,'train_epoch_losses.pt'))
                    torch.save(train_epoch_abs_losses, os.path.join(PATH,'train_epoch_abs_losses.pt'))
                    torch.save(train_epoch_rel_losses, os.path.join(PATH,'train_epoch_rel_losses.pt'))
                    
                    torch.save(val_epoch_losses, os.path.join(PATH, 'val_epoch_losses.pt'))
                    torch.save(val_epoch_abs_losses, os.path.join(PATH, 'val_epoch_abs_losses.pt'))
                    torch.save(val_epoch_rel_losses, os.path.join(PATH,'val_epoch_rel_losses.pt'))
                                        
                    torch.save(train_epoch_accuracies, os.path.join(PATH,'train_epoch_accuracies.pt'))
                    torch.save(val_epoch_accuracies, os.path.join(PATH,'val_epoch_accuracies.pt'))
            
    return best_state_dict


# https://github.com/keiradams/ChIRo/blob/2686d3a1801db8fb3dec10b61fb0cb0cec047c7c/model/gnn_3D/train_functions.py
def evaluate_binary_ranking_regression_loop_alpha(rank, config, world_size, model, loader, dataset_size):
    """Evaluate Docking Loop"""
    model.eval()
    
    all_targets = torch.zeros(dataset_size).to(rank)
    all_outputs = torch.zeros(dataset_size).to(rank)
    
    start = 0
    for batch in loader:
        # unpack batched data
        batch_data, y = batch
        batch_data = batch_data.to(rank)
        y = y.type(torch.float32)
        y = y.to(rank)

        # batched data attributes
        node2graph = batch_data.batch
        pos = batch_data.pos
        node_feature = batch_data.node_feature
        edge_index = batch_data.edge_index
        edge_attr = gen_edge_onehot(config, batch_data.edge_type)
        
        with torch.no_grad():
            # forward pass
            output = model(node_feature, pos, edge_index, edge_attr, node2graph)
        
            all_targets[start:start + y.squeeze().shape[0]] = y.squeeze()
            all_outputs[start:start + y.squeeze().shape[0]] = output.squeeze()
            start += y.squeeze().shape[0]
       
    return all_targets.detach().cpu().numpy(), all_outputs.detach().cpu().numpy()


# https://github.com/keiradams/ChIRo/blob/2686d3a1801db8fb3dec10b61fb0cb0cec047c7c/experiment_analysis/analyze_docking_experiments.ipynb
def get_ranking_accuracies(results_df, mode = '<='):
    """Calculate Docking ranking accuracies"""
    stats = results_df.groupby("ID")["outputs"].agg([np.mean, np.std]).merge(results_df, on = 'ID').reset_index(drop = True)
    
    smiles_groups_std = results_df.groupby(['ID', 'SMILES_nostereo'])['targets', 'outputs'].std().reset_index()
    smiles_groups_mean = results_df.groupby(['ID', 'SMILES_nostereo'])['targets', 'outputs'].mean().reset_index()
    smiles_groups_count = results_df.groupby(['ID', 'SMILES_nostereo'])['targets', 'outputs'].count().reset_index()
    
    stereoisomers_df = deepcopy(smiles_groups_mean).rename(columns = {'outputs': 'mean_predicted_score'})
    stereoisomers_df['std_predicted_score'] = smiles_groups_std['outputs']
    stereoisomers_df['count'] = smiles_groups_count.targets # score here simply contains the count
    
    stereoisomers_df_margins = stereoisomers_df.merge(pd.DataFrame(stereoisomers_df.groupby('SMILES_nostereo').apply(lambda x: np.max(x.targets) - np.min(x.targets)), columns = ['difference']), on = 'SMILES_nostereo')
    top_1_margins = []
    margins = np.arange(0.3, 2.1, 0.1)
    random_baseline_means = np.ones(len(margins)) * 0.5
    random_baseline_stds = []
    
    for margin in margins:
        if mode == '<=':
            subset = stereoisomers_df_margins[np.round(stereoisomers_df_margins.difference, 1) <= np.round(margin, 1)] # change to  ==, >=, <=
        elif mode == '>=':
            subset = stereoisomers_df_margins[np.round(stereoisomers_df_margins.difference, 1) >= np.round(margin, 1)] # change to  ==, >=, <=
        elif mode == '==':
            subset = stereoisomers_df_margins[np.round(stereoisomers_df_margins.difference, 1) == np.round(margin, 1)] # change to  ==, >=, <=
        
        top_1 = subset.groupby('SMILES_nostereo').apply(lambda x: np.argmin(np.array(x.targets)) == np.argmin(np.array(x.mean_predicted_score)))
        random_baseline_std = np.sqrt(len(top_1) * 0.5 * 0.5) # sqrt(npq) -- std of number of guesses expected to be right, when guessing randomly
        random_baseline_stds.append(random_baseline_std/len(top_1))
        acc = sum(top_1 / len(top_1))
        top_1_margins.append(acc)
    
    random_baseline_stds = np.array(random_baseline_stds)
    
    return margins, np.array(top_1_margins), random_baseline_means, random_baseline_stds


### Distributed Sampler Wrapper from "Catalyst" software
# https://github.com/catalyst-team/catalyst

from typing import Iterator, List, Optional
from operator import itemgetter

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
# from torch.utils.data import Dataset
from torch_geometric.data import Dataset
from torch.utils.data.sampler import BatchSampler


# https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/dataset.py
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


# https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


# from torchnlp
# https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/samplers/distributed_batch_sampler.html
class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs
        self.epoch = 0

    def __iter__(self):
        for batch in self.batch_sampler:
            dist_sampler = DistributedSampler(batch, **self.kwargs)
            dist_sampler.set_epoch(epoch=self.epoch)
            yield list(dist_sampler)

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch):
        self.epoch = epoch