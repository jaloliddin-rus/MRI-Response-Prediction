"""Loss functions and signal-similarity helpers for MRI signal prediction.

Extracted from the original `training.py`. `custom_l1_loss` is a weighted
combination of L1, Pearson-correlation, and cosine-similarity terms; the
defaults (0.20 / 0.40 / 0.40) are the weights used throughout the paper.
"""

import torch
import torch.nn.functional as F
from torch.nn import L1Loss


def pearson_correlation_loss(outputs, labels):
    outputs_mean = outputs.mean(dim=-1, keepdim=True)
    labels_mean = labels.mean(dim=-1, keepdim=True)
    corr = ((outputs - outputs_mean) * (labels - labels_mean)).sum(dim=-1) / (
        torch.sqrt(
            ((outputs - outputs_mean) ** 2).sum(dim=-1)
            * ((labels - labels_mean) ** 2).sum(dim=-1)
        )
    )
    return 1 - corr.mean()


def cosine_similarity_loss(outputs, labels):
    return 1 - F.cosine_similarity(outputs, labels, dim=-1).mean()


def custom_l1_loss(outputs, labels, alpha=0.20, beta=0.40, gamma=0.40):
    loss_L1 = L1Loss(reduction='none')(outputs, labels).mean(dim=-1)
    loss_pearson = pearson_correlation_loss(outputs, labels)
    loss_cosine = cosine_similarity_loss(outputs, labels)
    return alpha * loss_L1.mean() + beta * loss_pearson + gamma * loss_cosine


def signal_correlation(y_true, y_pred):
    """Mean Pearson correlation between true and predicted signals (metric)."""
    return torch.mean(
        torch.sum(
            (y_true - y_true.mean(dim=-1, keepdim=True))
            * (y_pred - y_pred.mean(dim=-1, keepdim=True)),
            dim=-1,
        )
        / (torch.std(y_true, dim=-1) * torch.std(y_pred, dim=-1))
    )
