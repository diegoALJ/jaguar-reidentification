from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = cosine * (1 - one_hot) + target * one_hot
        return logits * self.s


class JaguarReIDModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        embedding_dim: int = 512,
        head_type: str = "linear",
        dropout: float = 0.1,
        arcface_s: float = 30.0,
        arcface_m: float = 0.35,
    ):
        super().__init__()
        self.head_type = head_type

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        backbone_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(backbone_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

        if head_type == "linear":
            self.head = nn.Linear(embedding_dim, num_classes)
        elif head_type == "arcface":
            self.head = ArcMarginProduct(
                embedding_dim,
                num_classes,
                s=arcface_s,
                m=arcface_m,
            )
        else:
            raise ValueError("head_type must be 'linear' or 'arcface'")

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embedding = self.embedding(self.dropout(features))
        embedding = F.normalize(self.bn(embedding), p=2, dim=1)

        if self.head_type == "linear":
            logits = self.head(embedding)
        else:
            logits = self.head(embedding, labels) if labels is not None else None

        return embedding, logits
