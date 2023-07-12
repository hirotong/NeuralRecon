"""
    source: https://github.com/rosinality/stylegan2-pytorch/blob/master/distributed.py
"""

import torch
from torch import distributed as dist
from torch.utils.data.sampler import Sampler

def get_rank():
    if not dist.is_available():
        return 0
    
    if not dist.is_initialized():
        return 0
    
    return dist.get_rank()


def synchronize(local_rank=0):
    if not dist.is_available():
        return 
    
    if not dist.is_initialized():
        return 
    
    world_size = dist.get_world_size()
    
    if world_size == 1:
        return 
    
    dist.barrier()
    # dist.barrier(device_ids=[local_rank])
    

def get_world_size():
    if not dist.is_available():
        return 1
    
    if not dist.is_initialized():
        return 1
    
    return dist.get_world_size()

def reduce_sum(tensor):
    if not dist.is_available():
        return tensor
    
    if not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return tensor

def gather_grad(params):
    world_size = get_world_size()
    
    if world_size == 1:
        return 
    
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)

def all_gather(data):
    world_size = get_world_size()
    
    if world_size == 1:
        return [data]
    
    buffer

def reduce_loss_dict(loss_dict):
    world_size = 