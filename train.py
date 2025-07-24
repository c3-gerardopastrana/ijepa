import os
import random
import gc
from collections import defaultdict
from functools import partial

import numpy as np
np.set_printoptions(precision=4, suppress=True)

from PIL import Image
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.utils.data import DataLoader, random_split, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision import transforms, datasets

import wandb

from lda import LDA, lda_loss, sina_loss, SphericalLDA
from models import ResNet, BasicBlock
from utils import compute_wandb_metrics
from eval import run_linear_probe_on_embeddings

def ResNet18(num_classes=1000, lda_args=None, use_checkpoint=True, segments=4):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, lda_args, use_checkpoint, segments)


class Solver:
    def __init__(self, dataloaders, model_path, n_classes, lda_args={}, local_rank=0, world_size=1, lr=1e-3, 
                 gradient_accumulation_steps=1, use_amp=True, use_checkpoint=True):
        self.dataloaders = dataloaders
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        
        # Create model with checkpointing enabled
        self.net = ResNet18(n_classes, lda_args, use_checkpoint=use_checkpoint)
        self.net = self.net.to(self.device)
        
        # Wrap model with DDP
        if world_size > 1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank,
                           find_unused_parameters=False)  # Set to True only if needed
        
        self.use_lda = True if lda_args else False
        if self.use_lda:
            self.criterion = sina_loss 
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        if local_rank == 0:
            print(f"Using criterion: {self.criterion}")
            print(f"Using checkpoint: {use_checkpoint}")
            print(f"Using mixed precision: {use_amp}")
            print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=5e-4)
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.model_path = model_path
        self.n_classes = n_classes

    def get_net(self):
        return self.net.module if isinstance(self.net, DDP) else self.net

    def handle_lda(self, inputs, targets, epoch, batch_idx):
        net = self.get_net()
        xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t = net(inputs, targets, epoch)
     
    
        
        loss = self.criterion(sigma_w_inv_b, sigma_w, sigma_b, xc_mean, sigma_t)
    
        if self.local_rank == 0 and batch_idx % 5==0:
            metrics = compute_wandb_metrics(xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t)
            wandb.log(metrics, commit=False)
            wandb.log({'loss': loss.item(), 'epoch': epoch}, commit=False)
    
        return loss, sigma_w_inv_b

    def iterate(self, epoch, phase):
        get_net = self.get_net()
        get_net.train(phase == 'train')
    
        dataloader = self.dataloaders[phase]
        total_loss = 0
        correct = 0
        total = 0
        entropy_sum = 0.0
        entropy_count = 0
    
        torch.cuda.empty_cache()
        gc.collect()
    
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
    
            if phase == 'train':
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.use_lda:
                        loss, sigma_w_inv_b = self.handle_lda(inputs, targets, epoch, batch_idx)
                    else:
                        outputs = get_net(inputs, targets, epoch)
                        loss = self.criterion(outputs, targets)
    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
    
                if self.local_rank == 0 and batch_idx % 5==0:
                    wandb.log({"grad_norm": grad_norm.item()})
            else:
                with torch.no_grad():
                    if self.use_lda:
                        result = self.handle_lda(inputs, targets, epoch, batch_idx)
                        raise NotImplementedError("handle_lda is not implemented yet")
                    else:
                        outputs = get_net(inputs, targets, epoch)
                        loss = self.criterion(outputs, targets)
    
            total_loss += loss.item()
            total += targets.size(0)
    
            del inputs, targets
            if self.use_lda and phase == 'train':
                del sigma_w_inv_b
            torch.cuda.empty_cache()
    
            
        # Sync metrics across GPUs
        if self.world_size > 1:
            metrics = torch.tensor([total_loss], dtype=torch.float32, device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss = metrics.item()

            
        total_loss /= (batch_idx + 1) * self.world_size
        # Log metrics
        if self.local_rank == 0:
            if entropy_count > 0:
                average_entropy = entropy_sum / entropy_count
                print(f'Average Entropy: {average_entropy:.4f}')
            
            print(f'\nepoch {epoch}: {phase} loss: {total_loss:.3f}')
            wandb.log({
                f"epoch_{phase}": epoch,
                f"loss_{phase}": total_loss,
            }) 
        return total_loss
            

    def save_checkpoint(self, epoch, val_loss, suffix=''):
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'state_dict': self.get_net().state_dict()
        }
        path = self.model_path if not suffix else self.model_path.replace('.pth', f'_{suffix}.pth')
        torch.save(checkpoint, path)

    def train(self, epochs):
        best_loss = -float('inf')
    
        for epoch in range(epochs):
            # Set epoch for distributed samplers
            if self.world_size > 1:
                for phase in self.dataloaders:
                    sampler = getattr(self.dataloaders[phase], 'sampler', None)
                    if hasattr(sampler, 'set_epoch'):
                        sampler.set_epoch(epoch)
    
            # Training phase (we ignore returned values here)
            self.iterate(epoch, 'train')
            if epoch % 5 == 0:
                import time
                start_time = time.time()
                lda_accuracy = run_linear_probe_on_embeddings(
                    self.dataloaders['complete_train'],
                    self.dataloaders['val'],
                    self.get_net(),
                    use_amp=self.use_amp
                )
                
                # Only rank 0 gets accuracy; others get None
                if self.local_rank == 0 and lda_accuracy is not None:
                    wandb.log({'lda_accuracy': lda_accuracy})
                    elapsed_time = (time.time() - start_time) / 60  # convert to minutes
                    print(f"Total time: {elapsed_time:.2f} minutes")


    
            # Save best model
            if self.local_rank == 0:
                if lda_accuracy > best_loss:
                    best_loss = lda_accuracy
                    print('Best val loss found')
                    self.save_checkpoint(epoch, lda_accuracy)
    
                print()
    
        # Final save
        if self.local_rank == 0:
            self.save_checkpoint(epochs - 1, lda_accuracy, suffix='final')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    
def train_worker(rank, world_size, config):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    class ClassBalancedBatchSampler(Sampler):
        def __init__(self, dataset, k_classes: int, n_samples: int,
                     world_size: int = 1, rank: int = 0, seed: int = 42):
            """
            Class-balanced batch sampler for distributed training.
    
            Args:
                dataset: Dataset to sample from.
                k_classes: Number of different classes in each batch.
                n_samples: Number of samples per class.
                world_size: Total number of distributed workers.
                rank: Rank of the current worker.
                seed: Random seed for reproducibility.
            """
            super().__init__(dataset)
            self.dataset = dataset
            self.k_classes = k_classes
            self.n_samples = n_samples
            self.world_size = world_size
            self.rank = rank
            self.seed = seed
            self.epoch = 0  # Set externally before each epoch
    
            # Get target labels and build class-to-indices mapping
            if isinstance(dataset, torch.utils.data.Subset):
                indices = dataset.indices
                targets = [dataset.dataset.targets[i] for i in indices]
            else:
                indices = range(len(dataset))
                targets = dataset.targets
    
            self.class_to_indices = defaultdict(list)
            for idx, label in zip(indices, targets):
                self.class_to_indices[label].append(idx)
    
            # Filter out classes with insufficient samples
            self.available_classes = [cls for cls, idxs in self.class_to_indices.items()
                                      if len(idxs) >= n_samples]
            if len(self.available_classes) < k_classes:
                raise ValueError(f"Need at least {k_classes} classes with ≥{n_samples} samples each, "
                                 f"but only {len(self.available_classes)} are available.")
    
            # Estimate batches per epoch
            total_samples = sum(len(self.class_to_indices[cls]) for cls in self.available_classes)
            batch_size = k_classes * n_samples
            print("total samples", total_samples)
            print("batches per epoch", total_samples // batch_size)
            self.batches_per_epoch = total_samples // batch_size
    
        def set_epoch(self, epoch: int):
            self.epoch = epoch
    
        def __iter__(self):
            rng = random.Random(self.seed + self.epoch + self.rank)
            num_batches = 0
            batch_size = self.k_classes * self.n_samples
    
            while num_batches < self.batches_per_epoch:
                selected_classes = rng.sample(self.available_classes, self.k_classes)
    
                batch = np.empty(batch_size, dtype=int)
                offset = 0
                for cls in selected_classes:
                    sampled_indices = rng.sample(self.class_to_indices[cls], self.n_samples)
                    batch[offset:offset + self.n_samples] = sampled_indices
                    offset += self.n_samples
    
                # Shard to the correct worker
                if num_batches % self.world_size == self.rank:
                    yield batch.tolist()
    
                num_batches += 1
    
        def __len__(self):
            return self.batches_per_epoch // self.world_size

            
    # Configure CUDA
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Help with fragmentation
    
    # Setup process group
    setup(rank, world_size)
    
    # Set the device
    torch.cuda.set_device(rank)
    
    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            group=config['wandb_group'],
            config=config,  # Track configuration
        )
    
    # Set seeds for reproducibility
    seed = config['seed'] + rank  # Different seed per process
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Calculate effective batch size and adjust learning rate
    global_batch_size = config['k_classes'] * config['n_samples'] * world_size
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    effective_batch_size = global_batch_size * gradient_accumulation_steps
    base_lr = config.get('base_lr', 1e-3)
    lr = base_lr#get_scaled_lr_sqrt(effective_batch_size, base_batch_size=config.get('base_batch_size', 128), base_lr=base_lr)
    
    if rank == 0:
        print(f"Global batch size: {global_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Learning rate: {lr}")
    
    # Data loading code (same as original)
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    
    trainset = datasets.ImageFolder(config['train_dir'], transform=transform_train)
    valset = datasets.ImageFolder(config['val_dir'], transform=transform_test)
    testset = datasets.ImageFolder(config['test_dir'], transform=transform_test)



    # # Load the full datasets
    # trainset_full = datasets.ImageFolder(config['train_dir'], transform=transform_train)
    # valset_full = datasets.ImageFolder(config['val_dir'], transform=transform_test)
    # testset_full = datasets.ImageFolder(config['test_dir'], transform=transform_test)
    
    # # Select 10 class indices (e.g., 10 random or specific ones)
    # selected_classes = list(range(100))  # or any 10 specific indices you want
    
    # # Map class name to index
    # class_to_idx = trainset_full.class_to_idx
    # idx_to_class = {v: k for k, v in class_to_idx.items()}

    
    # # Create a filter function
    # def filter_by_class(dataset, allowed_classes):
    #     indices = [i for i, (_, label) in enumerate(dataset.samples) if label in allowed_classes]
    #     return Subset(dataset, indices)
    
    # # Create filtered datasets
    # trainset = filter_by_class(trainset_full, selected_classes)
    # valset = filter_by_class(valset_full, selected_classes)
    # testset = filter_by_class(testset_full, selected_classes)

    # Create subset
    transit_size = int(0.5 * len(trainset))
    indices = random.sample(range(len(trainset)), transit_size)
    transit_subset = Subset(trainset, indices)

    # Create distributed samplers
    train_sampler = ClassBalancedBatchSampler(
        dataset=trainset,
        k_classes=config['k_classes'],
        n_samples=config['n_samples'],
        world_size=world_size,
        rank=rank,
        seed=config['seed']
    )


    val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False)
    complete_train_sampler = DistributedSampler(transit_subset, num_replicas=world_size, rank=rank, shuffle=False)
    

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        #persistent_workers=False
    )

    
    valloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=config['batch_size'],
        sampler=test_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
        
    complete_train_loader = torch.utils.data.DataLoader(
        transit_subset, 
        batch_size=config['batch_size'],
        sampler=complete_train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    dataloaders = {'train': trainloader, 'val': valloader, 'test': testloader, 'complete_train':complete_train_loader}
    
    if config['loss'] == 'LDA':
        lda_args = {'lamb': config['lamb'], 'n_eig': config['n_eig'], 'margin': config['margin']}
    else:
        lda_args = {}
        
    # Create solver with optimized parameters
    solver = Solver(
        dataloaders=dataloaders, 
        model_path=config['model_path'],
        n_classes=config['n_classes'],
        lda_args=lda_args if config['loss'] == 'LDA' else {},
        local_rank=rank,
        world_size=world_size,
        lr=lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=config.get('use_amp', True),
        use_checkpoint=config.get('use_checkpointing', True)
    )
    
    # Train
    solver.train(config['epochs'])
    
    # Test
    solver.test()
    
    # Clean up
    cleanup()


if __name__ == '__main__':
    # Configuration with memory optimizations
    config = {
        'wandb_project': "DELETEME_medium",
        'wandb_entity': "gerardo-pastrana-c3-ai",
        'wandb_group': "gapLoss",
        'seed': 42,
        'n_classes': 1000,
        'train_val_split': 0.1,
        'batch_size': 4096,  # Global batch size
        'num_workers': 1,  # Adjust based on CPU cores
        'train_dir': '/data/datasets/imagenet_full_size/061417/train',
        'val_dir': '/data/datasets/imagenet_full_size/061417/val',
        'test_dir': '/data/datasets/imagenet_full_size/061417/test',
        'model_path': 'models/deeplda_best.pth',
        'loss': 'LDA',
        'lamb': 0.0002,
        'n_eig': 4,
        'margin': None,
        'epochs': 100,
        'k_classes': 160,
        'n_samples': 64, #32
        # Memory optimization parameters
        'gradient_accumulation_steps': 1,  # Accumulate gradients to save memory
        'use_amp': True,                   # Use automatic mixed precision
        'use_checkpointing': True,         # Use gradient checkpointing
        'base_lr': 1e-3,                   # Base learning rate
        'base_batch_size': 128,            # Reference batch size for LR scaling
        'cuda_visible_devices': '',        # Optional GPU restrictions
    }
    
    # Number of available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")
    
    # Launch processes
    mp.spawn(
        train_worker,
        args=(n_gpus, config),
        nprocs=n_gpus,
        join=True
    )