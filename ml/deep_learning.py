from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch_geometric.nn import SAGEConv

from ml.config import RANDOM_STATE


class TabularResBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(self, in_dim: int = 31, hidden_dim: int = 128, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            TabularResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h)
        return torch.sigmoid(logits).squeeze(-1)


class TabularDeepLearningClassifier:

    def __init__(
        self,
        in_dim: int = 31,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        batch_size: int = 256,
        lr: float = 1e-3,
        epochs: int = 25,
        device: str | None = None,
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.scaler = RobustScaler()
        self.model: TabularResNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabularDeepLearningClassifier":
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TabularResNet(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
        ).to(self.device)

        pos_weight = torch.tensor([1.0], device=self.device)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        self.model.eval()
        X_scaled = self.scaler.transform(X)
        probs = []
        batch_sz = 2048

        with torch.no_grad():
            for start in range(0, len(X_scaled), batch_sz):
                end = min(start + batch_sz, len(X_scaled))
                bx = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self.device)
                p = self.model(bx).cpu().numpy()
                probs.append(p)

        prob_1 = np.concatenate(probs)
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])


class GraphSAGETxnModel(nn.Module):
    def __init__(self, node_dim: int = 16, edge_dim: int = 31, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(node_dim, hidden_dim, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_node: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        eval_edges: torch.Tensor,
    ) -> torch.Tensor:
        h1 = self.act(self.bn1(self.conv1(x_node, edge_index)))
        h2 = self.act(self.bn2(self.conv2(h1, edge_index)))
        orig_nodes = eval_edges[0]
        dest_nodes = eval_edges[1]
        h_orig = h2[orig_nodes]
        h_dest = h2[dest_nodes]
        edge_feat = torch.cat([h_orig, h_dest, edge_attr], dim=-1)
        logits = self.edge_mlp(edge_feat)
        return torch.sigmoid(logits).squeeze(-1)

