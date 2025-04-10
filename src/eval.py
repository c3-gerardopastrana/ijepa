import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import random
import os

from src.datasets.imagenet1k import make_imagenet1k

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="'multi_class' was deprecated", category=FutureWarning, module="sklearn.linear_model._logistic")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Evaluation:
    def __init__(self, transform, mask_collator, pin_mem, num_workers, world_size, rank, root_path, image_folder, batch_size, seed=0):
        self.set_seed(seed)
        self.unsupervised_loader = self.create_loader(batch_size, transform, mask_collator, pin_mem, num_workers, world_size, rank, root_path, image_folder)
        self.clf = LogisticRegression(
            penalty='l2',
            solver='saga',
            multi_class='multinomial',
            max_iter=3, # It takes around 20 to fully converge, early stopping
            n_jobs=-1,
            warm_start=True  # Enable warm start for faster re-fitting
        )

    @staticmethod
    def set_seed(seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def create_loader(self, batch_size, transform, mask_collator, pin_mem, num_workers, world_size, rank, root_path, image_folder):
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=True,
            drop_last=False,
            subset_file="src/datasets/imagenet_1_percent.txt"
        )
        return unsupervised_loader

    @staticmethod
    def extract_embeddings(encoder, dataloader, device):
        encoder.eval()
        features, labels = [], []

        with torch.no_grad():
            for udata, _, _ in dataloader:
                imgs = udata[0].to(device)
                lbls = udata[1]

                emb = encoder(imgs).cpu()
                features.append(emb)
                labels.append(lbls)

        X = torch.cat(features).numpy()
        y = torch.cat(labels).numpy()
        
        return X, y

    def train_and_evaluate_logistic_regression(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
        self.clf.fit(X_train, y_train)
        y_probs = self.clf.predict_proba(X_val)
        top5 = top_k_accuracy_score(y_val, y_probs, k=5)

        return top5

    def evaluate_top5_performance(self, encoder, device):
        X, y = self.extract_embeddings(encoder, self.unsupervised_loader, device)
        top5 = self.train_and_evaluate_logistic_regression(X, y)

        return top5

