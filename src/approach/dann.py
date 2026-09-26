import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser

from datasets.exemplars_dataset import ExemplarsDataset
from .incremental_learning import Inc_Learning_Appr

class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for Domain-Adversarial Neural Networks.
    Forward pass: Identity mapping.
    Backward pass: Multiplies incoming gradients by -alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class FocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss to address severe class imbalance in IoT traffic.
    Downweights easy negative examples and focuses training on hard/rare attacks.
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class Appr(Inc_Learning_Appr):
    """
    Domain-Adversarial Neural Network (DANN) with Class-Balanced Focal Loss.
    
    Key Features:
    1. Gradient Reversal Layer (GRL): Learns domain-invariant feature representations
       across heterogeneous networks (Edge-IIoT, ToN-IoT, IoT-NIDD), eliminating the
       drastic cross-network generalization drop.
    2. Focal Loss: Prevents majority classes (e.g. benign, inject-http) from drowning
       out rare, high-severity attacks (ransomware, backdoor, arp-spoofing).
    3. Replay Memory Alignment: Preserves source-domain knowledge while adapting to target domains.
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000, momentum=0.9, wd=0.0001, logger=None, exemplars_dataset=None,
                 gamma=2.0, domain_weight=0.2, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience,
                                   clipgrad, momentum, wd, logger, exemplars_dataset, **kwargs)
        self.__dict__.update(kwargs)
        
        self.gamma = gamma
        self.domain_weight = domain_weight
        self.focal_loss = FocalLoss(gamma=self.gamma)
        
        # Domain Discriminator Head: 200 -> 64 -> 2 domains (Source vs Target)
        feature_dim = 200
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        ).to(self.device)

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument('--gamma', default=2.0, type=float, help='Focal loss gamma focusing parameter')
        parser.add_argument('--domain-weight', default=0.2, type=float, help='Weight of domain adversarial loss')
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _get_optimizer(self, t=None):
        """Optimizes both model parameters and domain discriminator parameters"""
        params = list(self.model.parameters()) + list(self.domain_classifier.parameters())
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop, concatenating memory buffer when t > 0"""
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory
            )
        return super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        """Executes one epoch with dynamic GRL alpha and adversarial domain alignment"""
        self._model.train()
        self.domain_classifier.train()

        total_steps = len(trn_loader) * self.nepochs
        
        for step, (images, targets) in enumerate(trn_loader):
            images, targets = self.format_inputs(images, targets)
            
            # Forward pass to extract task outputs and latent features
            outputs, features = self.model(images, return_features=True)
            cat_outputs = torch.cat(outputs, dim=1)
            
            # 1. Classification Loss (Focal Loss)
            loss_cls = self.focal_loss(cat_outputs, targets)
            
            # 2. Domain Adversarial Loss
            if t > 0 and len(self.exemplars_dataset) > 0:
                # Dynamic alpha schedule: gradually increase adversarial pressure
                p = float(step) / total_steps
                alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
                
                # Apply Gradient Reversal Layer
                rev_features = GradReverse.apply(features, alpha)
                domain_preds = self.domain_classifier(rev_features)
                
                # Targets <= t-1 are from source (domain 0), new targets are target (domain 1)
                domain_targets = (targets >= self.model.task_offset[t]).long().to(self.device)
                loss_domain = F.cross_entropy(domain_preds, domain_targets)
                
                total_loss = loss_cls + self.domain_weight * loss_domain
            else:
                total_loss = loss_cls
                
            # Backward and optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets, features=None, epoch=1):
        """Returns the validation loss value (Focal Loss)"""
        cat_outputs = torch.cat(outputs, dim=1)
        return self.focal_loss(cat_outputs, targets)

    def post_train_process(self, t, trn_loader, val_loader, correction=False):
        """Collects representative exemplars into replay memory for subsequent tasks"""
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        return super().post_train_process(t, trn_loader, val_loader)
