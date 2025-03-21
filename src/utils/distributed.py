# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
import torch.distributed as dist
from logging import getLogger

logger = getLogger()

def init_distributed(port=40112, rank_and_world_size=(None, None)):
    """Initialize distributed training environment.
    
    Args:
        port: Port to use for distributed communication
        rank_and_world_size: Tuple of (rank, world_size) if manually specified
        
    Returns:
        Tuple of (world_size, rank)
    """
    # If already initialized, return current settings
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    
    # Unpack manual settings if provided
    rank, world_size = rank_and_world_size
    
    # Set default master address
    os.environ['MASTER_ADDR'] = 'localhost'
    
    # Try to get SLURM settings if not manually specified
    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank
    
    # Initialize process group
    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f'distributed training not available {e}')
    
    return world_size, rank

def init_rpc(rank_and_world_size=(None, None), port=40112):
    """Initialize RPC and distributed environment.

    Args:
        rank_and_world_size: Tuple of (rank, world_size) if manually specified
        port: Port to use for distributed communication

    Returns:
        Tuple of (world_size, rank)
    """
    
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    # Initialize basic distributed process group
    world_size, rank = init_distributed(port=port, rank_and_world_size=rank_and_world_size)
    for i in range(torch.cuda.device_count()):
        logger.info(f"Worker {rank} - CUDA device {i}: {torch.cuda.get_device_name(i)}")

    if world_size > 1:
        try:
            # Make a unique RPC port offset from the distributed port
            rpc_port = port + 10
            rpc_master_addr = os.environ.get("MASTER_ADDR", "localhost")

            # Configure TensorPipe RPC backend options
            rpc_backend_options = torch.distributed.rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{rpc_master_addr}:{rpc_port}",
                rpc_timeout=120,
                num_worker_threads=8
            )

            # Explicitly map CUDA devices across workers
            for i in range(world_size):  
                rpc_backend_options.set_device_map(f"worker{i}", {rank: rank})

            # Initialize RPC
            worker_name = f"worker{rank}"
            torch.distributed.rpc.init_rpc(
                name=worker_name,
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options
            )

            getLogger().info(f"Initialized RPC for {worker_name} (rank {rank}/{world_size})")

        except Exception as e:
            getLogger().warning(f"RPC initialization failed: {e}")

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    #@staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            grads = grads.contiguous()
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            # Gather all gradient slices
            grad_outputs = [torch.zeros_like(grads) for _ in range(world_size)]
            dist.all_gather(grad_outputs, grads)  
    
            # Sum gradients across all ranks
            gathered_grads = torch.cat(grad_outputs, dim=0)
            
            # Extract relevant slice for this rank
            s = (gathered_grads.shape[0] // world_size) * rank
            e = (gathered_grads.shape[0] // world_size) * (rank + 1)
            return gathered_grads[s:e]
        
        return grads

    # def backward(ctx, grads):
    #     if (
    #         dist.is_available()
    #         and dist.is_initialized()
    #         and (dist.get_world_size() > 1)
    #     ):
    #         s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
    #         e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
    #         grads = grads.contiguous()
    #         dist.all_reduce(grads)
    #         return grads[s:e]
    #     return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
