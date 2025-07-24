import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from src.datasets.imagenet1k import make_imagenet1k
import random
import numpy as np
import os

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

class Evaluation:
    def __init__(self, transform, mask_collator, pin_mem, num_workers, world_size, rank, root_path, image_folder, batch_size, seed=0):
        self.set_seed(seed)
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create train and val loaders
        _, self.unsupervised_loader, _ = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=True,
            copy_data=False,
            drop_last=False,
            subset_file="src/datasets/imagenet_1_percent.txt"
        )

        _, self.validation_loader, _ = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=False,
            copy_data=False,
            drop_last=False,
        )

    @staticmethod
    def set_seed(seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @staticmethod
    def extract_embeddings(encoder, dataloader, device, use_amp=True):
        encoder.eval()
        features, labels = [], []

        with torch.no_grad():
            for udata, _, _ in dataloader:
                imgs = udata[0].to(device, non_blocking=True)
                lbls = udata[1].to(device, non_blocking=True)
                # inherit same fot training TODO
                with torch.cuda.amp.autocast(enabled=use_amp):
                    emb = encoder(imgs)

                features.append(emb)
                labels.append(lbls)

        embeddings = torch.cat(features)
        labels = torch.cat(labels)

        if dist.is_initialized():
            embeddings = gather_tensor(embeddings)
            labels = gather_tensor(labels)

        return embeddings, labels

    def run_linear_probe(self, encoder, epochs=10, lr=0.1, batch_size=4096, use_amp=True):
        # Extract embeddings for train and val
        X_train, y_train = self.extract_embeddings(encoder, self.unsupervised_loader, self.device, use_amp)
        X_val, y_val = self.extract_embeddings(encoder, self.validation_loader, self.device, use_amp)

        if self.rank == 0:
            print(f"Extracted embeddings shapes: train {X_train.shape}, val {X_val.shape}")

            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

            classifier = LinearClassifier(X_train.shape[1], int(y_train.max()) + 1).to(self.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                classifier.train()
                total_correct = 0
                total_samples = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = classifier(xb)
                    loss = criterion(out, yb)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    preds = out.argmax(dim=1)
                    total_correct += (preds == yb).sum().item()
                    total_samples += yb.size(0)

                train_acc = total_correct / total_samples * 100
                print(f"Epoch {epoch+1}: Train Accuracy = {train_acc:.2f}%")

            classifier.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = classifier(xb).argmax(dim=1)
                    total_correct += (preds == yb).sum().item()
                    total_samples += yb.size(0)

            val_acc = total_correct / total_samples * 100
            print(f"Validation Accuracy = {val_acc:.2f}%")
            return val_acc
        else:
            return None


