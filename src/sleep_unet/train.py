"""Group-K-fold training loop from ``unet/unet.ipynb``."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from sleep_unet.data import FILTERS, MyDataset, PreparedWindows
from sleep_unet.model import build_unet


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class CustomCosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, eta_min=0, last_epoch=-1, lr_decay_factor=0.9):
        self.T_0 = T_0
        self.eta_min = eta_min
        self.lr_decay_factor = lr_decay_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle, position = divmod(self.last_epoch, self.T_0)
        amplitude = (1 + math.cos(math.pi * position / self.T_0)) / 2
        return [
            self.eta_min + (base_lr * self.lr_decay_factor**cycle - self.eta_min) * amplitude
            for base_lr in self.base_lrs
        ]


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    n = 0

    for batch in data_loader:
        X = batch[0].to(device)
        Y = batch[1].to(device)
        mask = batch[2].to(device).bool()
        if not mask.any():
            continue

        loss = criterion(model(X)[mask], Y[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n += int(mask.sum())

    if n == 0:
        raise RuntimeError("training loader has no valid minutes")


@torch.no_grad()
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()

    n = 0
    total_loss = 0.0
    for batch in data_loader:
        X = batch[0].to(device)
        Y = batch[1].to(device)
        mask = batch[2].to(device).bool()
        count = int(mask.sum())
        if count == 0:
            continue

        loss = criterion(model(X)[mask], Y[mask])
        total_loss += loss.item() * count
        n += count

    if n == 0:
        raise RuntimeError("validation loader has no valid minutes")
    return total_loss / n


@torch.no_grad()
def predict(model, data_loader, device):
    model.eval()

    preds_all = []
    for batch in data_loader:
        X = batch[0].to(device)
        mask = batch[2].to(device)

        preds = model(X) * mask
        preds = preds.cpu().numpy()
        preds_all.append(preds)

    return np.concatenate(preds_all)


def rmse(y_true, y_pred, mask=None) -> float:
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != y_true.shape or valid.shape != y_pred.shape or not valid.any():
            raise ValueError("RMSE mask must match the targets and contain valid minutes")
        y_true, y_pred = y_true[valid], y_pred[valid]
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def prepared_fingerprint(prepared: PreparedWindows) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([prepared.index.tolist(), prepared.features]).encode())
    digest.update(pd.util.hash_pandas_object(prepared.df_1min, index=False).to_numpy())
    for array in (prepared.X, prepared.Y, prepared.mask, prepared.norm_min, prepared.norm_max):
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array))
    return digest.hexdigest()


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResult:
    oof: np.ndarray
    best_rmse: float
    initial_channels: int


def train_model(
    prepared: PreparedWindows,
    out_dir: str | Path,
    *,
    epochs: int = 20,
    n_splits: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 1,
    filters: int = FILTERS,
    initial_channels: int | None = None,
    patience: int = 20,
    res_depth: int = 3,
    se_ratio: int = 4,
) -> TrainResult:
    if epochs < 1 or batch_size < 1 or seed < 1:
        raise ValueError("epochs, batch_size and seed count must be positive")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"training output directory must be empty: {out_dir}")
    X = prepared.X
    Y = prepared.Y
    mask = prepared.mask
    groups = prepared.index.get_level_values("series_id").to_numpy()
    n_groups = len(np.unique(groups))
    if n_splits > n_groups:
        raise ValueError(f"GroupKFold n_splits={n_splits} but only {n_groups} series are present")
    if initial_channels is None:
        initial_channels = int(X.shape[1] * filters)

    device = get_device()
    pin = device.type == "cuda"
    oof = np.zeros_like(Y, dtype=np.float64)
    best_model_rmse = np.inf
    gkf = GroupKFold(n_splits=n_splits)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(seed):
        for fold, (idx_train, idx_valid) in enumerate(gkf.split(X, Y, groups=groups)):
            print(f"Seed: {k} | Fold: {fold}")
            ds_train = MyDataset(X[idx_train], Y[idx_train], mask[idx_train])
            ds_valid = MyDataset(X[idx_valid], Y[idx_valid], mask[idx_valid])
            use_batch = min(batch_size, len(ds_train))
            dl_train = DataLoader(
                ds_train,
                batch_size=use_batch,
                shuffle=True,
                num_workers=0,
                pin_memory=pin,
                drop_last=True,
            )
            dl_valid = DataLoader(
                ds_valid,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin,
                drop_last=False,
            )

            torch.manual_seed(k)
            model = build_unet(
                input_channels=X.shape[1],
                initial_channels=initial_channels,
                res_depth=res_depth,
                se_ratio=se_ratio,
            )
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = CustomCosineAnnealingWarmRestarts(
                optimizer, T_0=10, eta_min=1e-5, lr_decay_factor=0.9
            )
            criterion = nn.MSELoss()
            best_loss = np.inf
            early_stopping = EarlyStopping(patience=patience)
            checkpoint = out_dir / f"model_{k}_{fold}.pth"

            for epoch in range(epochs):
                train_one_epoch(model, dl_train, optimizer, criterion, device)
                loss = evaluate_loss(model, dl_valid, criterion, device)
                scheduler.step()
                early_stopping(loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
                current_lr = optimizer.param_groups[0]["lr"]
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), checkpoint)
                    print(
                        f"epoch: {epoch}\tvalid-loss: {loss}\t"
                        f"Current Learning Rate: {current_lr}\tbest!"
                    )
                else:
                    print(
                        f"epoch: {epoch}\tvalid-loss: {loss}\tCurrent Learning Rate: {current_lr}"
                    )

            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            preds = predict(model, dl_valid, device)
            oof[idx_valid] += preds
            fold_rmse = rmse(Y[idx_valid], preds, mask[idx_valid])
            best_model_rmse = min(best_model_rmse, fold_rmse)
            print(f"seed: {k} fold: {fold} best_loss: {best_loss} RMSE: {fold_rmse}")

    oof /= seed
    meta = {
        "input_channels": int(X.shape[1]),
        "initial_channels": int(initial_channels),
        "res_depth": int(res_depth),
        "se_ratio": int(se_ratio),
        "features": list(prepared.features),
        "norm_min": prepared.norm_min.tolist(),
        "norm_max": prepared.norm_max.tolist(),
        "seed": int(seed),
        "n_splits": int(n_splits),
        "batch_size": int(batch_size),
        "prepared_hash": prepared_fingerprint(prepared),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    np.save(out_dir / "oof.npy", oof)
    return TrainResult(oof=oof, best_rmse=best_model_rmse, initial_channels=initial_channels)


def load_meta(checkpoint_dir: str | Path) -> dict:
    return json.loads((Path(checkpoint_dir) / "meta.json").read_text())


def fold_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    directory = Path(checkpoint_dir)
    meta = load_meta(directory)
    if meta["seed"] < 1 or meta["n_splits"] < 2:
        raise ValueError("checkpoint metadata must declare positive seeds and at least two folds")
    files = [
        directory / f"model_{seed}_{fold}.pth"
        for seed in range(meta["seed"])
        for fold in range(meta["n_splits"])
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"missing fold checkpoint: {path}")
    return files


def load_oof(prepared: PreparedWindows, checkpoint_dir: str | Path) -> np.ndarray:
    preds = np.load(Path(checkpoint_dir) / "oof.npy")
    if load_meta(checkpoint_dir).get("prepared_hash") != prepared_fingerprint(prepared):
        raise ValueError(
            "OOF data does not match this prepared dataset; retrain on the matching data"
        )
    if preds.shape != prepared.Y.shape or not np.isfinite(preds).all():
        raise ValueError("OOF predictions must be finite and match the target shape")
    return preds


def predict_windows(
    X: np.ndarray,
    mask: np.ndarray,
    checkpoint_dir: str | Path,
    batch_size: int = 16,
) -> np.ndarray:
    """Average the fold checkpoints. Training saved ``model_{seed}_{fold}.pth``."""
    meta = load_meta(checkpoint_dir)
    device = get_device()
    pin = device.type == "cuda"
    loader = DataLoader(
        MyDataset(X, None, mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )
    checkpoints = fold_checkpoints(checkpoint_dir)
    model = build_unet(
        input_channels=int(meta["input_channels"]),
        initial_channels=int(meta["initial_channels"]),
        res_depth=int(meta["res_depth"]),
        se_ratio=int(meta["se_ratio"]),
    ).to(device)
    total = None
    for path in checkpoints:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        preds = predict(model, loader, device)
        total = preds if total is None else total + preds
    return total / len(checkpoints)
