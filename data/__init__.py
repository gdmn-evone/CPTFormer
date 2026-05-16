import os
import random
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, DistributedSampler
from data.datasets_phase import CMPDataset

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler

def create_dataloader(opt, world_size=1, rank=0):
    dataset = CMPDataset(opt)
    sampler = DistributedSampler(dataset, num_replicas=world_size,
        rank=rank, shuffle=True)
    g = torch.Generator()
    g.manual_seed(0)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt['batch_size'],
                                              shuffle=False,
                                              sampler=sampler,
                                              num_workers=int(opt['num_threads']),
                                              worker_init_fn=seed_worker,
                                              generator=g)
    return data_loader

def collate_fn_skip_broken(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return () 
    return torch.utils.data.default_collate(batch)
