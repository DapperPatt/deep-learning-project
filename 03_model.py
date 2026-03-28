"""
03_model.py
===========
GRU model definition, dataset wrapper, and training utilities.

Architecture
------------
Input  → GRU (2 layers, hidden=64, dropout=0.3)
       → LayerNorm
       → FC(64 → 32) + ReLU + Dropout
       → FC(32 → 1)   [predict next-season mean_delta for one team]

The model is designed to be simple and regularised — we have only ~10 teams
× ~12 seasons = ~120 samples total, so overfitting is the main risk.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from config import (
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, RANDOM_SEED,
)

torch.manual_seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class F1SequenceDataset(Dataset):
    """
    Wraps numpy arrays (X, y) as a PyTorch Dataset.

    Parameters
    ----------
    X : np.ndarray  shape (N, SEQ_LEN, num_features)
    y : np.ndarray  shape (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X, y, shuffle=True) -> DataLoader:
    ds = F1SequenceDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# GRU Model
# ─────────────────────────────────────────────────────────────────────────────

class F1GRU(nn.Module):
    """
    Multi-layer GRU for time-series regression.

    Input:  (batch, seq_len, input_size)
    Output: (batch,)   — scalar prediction per sequence
    """

    def __init__(
        self,
        input_size: int  = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int  = NUM_LAYERS,
        dropout: float   = DROPOUT,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        """
        out, _ = self.gru(x)         # (batch, seq_len, hidden)
        last    = out[:, -1, :]      # take final timestep
        last    = self.norm(last)
        pred    = self.head(last)    # (batch, 1)
        return pred.squeeze(-1)      # (batch,)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 40, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds.append(pred.cpu().numpy())
        targets.append(y_batch.cpu().numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae     = float(np.mean(np.abs(preds - targets)))
    return total_loss / len(loader.dataset), mae, preds, targets


def build_model_and_optim(lr: float, weight_decay: float, device: torch.device):
    model     = F1GRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=False
    )
    criterion = nn.MSELoss()
    return model, optimizer, scheduler, criterion
