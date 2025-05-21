import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def gather_tensor(tensor):
    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

def run_linear_probe_on_embeddings(train_loader, val_loader, model, device=None, use_amp=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def extract_embeddings(loader):
        embeddings, labels = [], []
        rank = dist.get_rank() if dist.is_initialized() else 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    feats = model._forward_impl(x)
                    feats = F.normalize(feats, p=2, dim=1)
                embeddings.append(feats)
                labels.append(y)
        
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)
    
        if dist.is_initialized():
            embeddings = gather_tensor(embeddings)
            labels = gather_tensor(labels)
        
        return embeddings, labels  # Keep everything on the GPU

    X_train, y_train = extract_embeddings(train_loader)
    X_val, y_val = extract_embeddings(val_loader)
    print("rank done", dist.get_rank())
    
    if dist.get_rank() == 0:
        print("Linear probing on embeddings with shape", X_train.shape)

        # Prepare data loaders for linear probing (already on CUDA)
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4096)

        # Define linear classifier
        classifier = LinearClassifier(X_train.shape[1], int(y_train.max()) + 1).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-1)
        criterion = nn.CrossEntropyLoss()

        # --- Training ---
        epochs = 10
        for epoch in range(epochs):
            classifier.train()
            correct, total = 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = classifier(xb)
                loss = criterion(out, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            acc = correct / total * 100
            print(f"Epoch {epoch+1}: Train Accuracy = {acc:.2f}%")
            print('total samples', total)

        # --- Evaluation ---
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = classifier(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        test_acc = correct / total * 100
        print(f"Test Accuracy = {test_acc:.2f}%")
        return test_acc
    else:
        return None  # only rank 0 computes linear probe



