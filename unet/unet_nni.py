import gc
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.signal import find_peaks

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary
import os
import pickle
import math
from torch.optim.lr_scheduler import _LRScheduler
import nni
from IPython.display import display

INPUT_DIR = ''

df_events = pl.read_csv(INPUT_DIR + "train_events.csv")
df_events = df_events.with_columns(
    pl.col("event").map_dict({"wakeup": 1.0, "onset": -1.0}, return_dtype=pl.Float32),
)

seed = 2
n_splits = 10
n_epochs = 50
params = nni.get_next_parameter()
batch_size = params["batch_size"]
initial_channels_num = params["initial_channels_num"]
lr = params["lr"]

df_1min = pd.read_csv('df_1min.csv')
df_y = df_1min.pivot(index=["series_id", "date"], columns="time", values="target").fillna(0)
df_mask = df_1min.pivot(index=["series_id", "date"], columns="time", values="valid_flag").fillna(0)
X = np.load('X_array.npy')
with open('dict_valid_ratio.pkl', 'rb') as pickle_file:
    dict_valid_ratio = pickle.load(pickle_file)


class MyDataset(Dataset):
    def __init__(self, X, Y, flag):
        self.X = torch.FloatTensor(X)
        if Y is not None:
            self.Y = torch.FloatTensor(Y)
        self.flag = torch.FloatTensor(flag)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if "Y" in dir(self):
            return (self.X[idx], self.Y[idx], self.flag[idx])
        else:
            return (self.X[idx], torch.Tensor(), self.flag[idx])
        
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
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

class CustomCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, lr_decay_factor=0.9):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.lr_decay_factor = lr_decay_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(CustomCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch % self.T_0
        if self.last_epoch != 0 and T_cur == 0:
            self.base_lrs = [lr * self.lr_decay_factor for lr in self.base_lrs]

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * T_cur / self.T_0)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        
        if stride == 1:
            padding = "same"
        else:
            padding = (kernel_size - stride) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x_out = self.layers(x)
        return x_out

class SEBlock(nn.Module):
    def __init__(self, n_channels, se_ratio):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),  #  Global Average Pooling
            nn.Conv1d(n_channels, n_channels//se_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_channels//se_ratio, n_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_out = torch.mul(x, self.layers(x))
        return x_out

class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, se_ratio):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
            ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
            SEBlock(n_channels, se_ratio)
        )
    
    def forward(self, x):
        x_re = self.layers(x)
        x_out = x + x_re
        return x_out
    
class UNet1d(nn.Module):
    def __init__(self, input_channels, initial_channels, initial_kernel_size,
                 down_channels, down_kernel_size, down_stride, res_depth, res_kernel_size, se_ratio, out_kernel_size):
        super().__init__()
        self.down_kernel_size = down_kernel_size
        self.down_stride = down_stride
        self.initial_layers = ConvBNReLU(input_channels, initial_channels, initial_kernel_size, stride=1, groups=input_channels)
        
        self.down_layers = nn.ModuleList()
        for i in range(len(down_channels)):
            if i == 0:
                in_channels = initial_channels
            else:
                in_channels = down_channels[i-1] + input_channels
            out_channels = down_channels[i]
            kernel_size = down_kernel_size[i]
            stride = down_stride[i]
            
            block = []
            block.append(ConvBNReLU(in_channels, out_channels, kernel_size, stride))
            for j in range(res_depth):
                block.append(ResBlock(out_channels, res_kernel_size, se_ratio))
            self.down_layers.append(nn.Sequential(*block))
        
        self.up_layers = nn.ModuleList()
        for i in range(len(down_channels)-1, 0, -1):
            in_channels = out_channels + down_channels[i]
            out_channels = down_channels[i]
            kernel_size = down_kernel_size[i]
            self.up_layers.append(ConvBNReLU(in_channels, out_channels, kernel_size, stride=1))
        
        self.out_layers = nn.Conv1d(down_channels[1], 1, out_kernel_size, padding="same")
    
    def forward(self, x):
        outs = []
        x_avg = x
        x = self.initial_layers(x)
        
        for i in range(len(self.down_layers)):
            x_out = self.down_layers[i](x)
            if i == len(self.down_layers) - 1:
                x = x_out
            else:
                outs.append(x_out)
                kernel_size = self.down_kernel_size[i]
                stride = self.down_stride[i]
                padding = (kernel_size - stride) // 2
                x_avg = F.avg_pool1d(x_avg, kernel_size, stride, padding)
                x = torch.cat([x_out, x_avg], dim=1)
        
        for i in range(len(self.up_layers)):
            scale_factor = self.down_stride[-i-1]
            x = F.interpolate(x, scale_factor=scale_factor, mode="linear")
            x = torch.cat([x, outs[-i-1]], dim=1)
            x = self.up_layers[i](x)
        
        x_out = self.out_layers(x)
        x_out = x_out[:, 0, 180:-180]
        
        return 
    
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    
    for batch in data_loader:
        X = batch[0].to(device)
        Y = batch[1].to(device)
        mask = batch[2].to(device)
        
        preds = model(X) * mask
        loss = criterion(preds, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, data_loader, criterion, device):
    model.eval()
    
    n = 0
    total_loss = 0.0
    for batch in data_loader:
        X = batch[0].to(device)
        Y = batch[1].to(device)
        mask = batch[2].to(device)
        
        with torch.no_grad():
            preds = model(X) * mask
        
        loss = criterion(preds, Y)
        total_loss += loss.item() * X.shape[0]
        n += X.shape[0]
    
    avg_loss = total_loss / n
    
    return avg_loss


def predict(model, data_loader, device):
    model.eval()
    
    preds_all = []
    for batch in data_loader:
        X = batch[0].to(device)
        mask = batch[2].to(device)
        
        with torch.no_grad():
            preds = model(X) * mask
        preds = preds.cpu().numpy()
        preds_all.append(preds)
    
    preds_all = np.concatenate(preds_all)
        
    return preds_all

best_model_rmse = np.inf
best_model_path = "best_model.pth" 
    
Y = df_y.values
mask = df_mask.values
groups = df_y.index.get_level_values("series_id").tolist()

preds1 = np.zeros_like(Y)
preds2 = np.zeros_like(Y)
preds_valid = np.zeros_like(Y)
gkf = GroupKFold(n_splits=n_splits)
for k in range(1, seed):
    for fold, (idx_train, idx_valid) in enumerate(gkf.split(X, Y, groups=groups)):   
        print(f"Seed: {k} | Fold: {fold}")
        X_train = X[idx_train]
        Y_train = Y[idx_train]
        mask_train = mask[idx_train]

        X_valid = X[idx_valid]
        Y_valid = Y[idx_valid]
        mask_valid = mask[idx_valid]

        # dataset
        ds_train = MyDataset(X_train, Y_train, mask_train)
        ds_valid = MyDataset(X_valid, Y_valid, mask_valid)

        # dataloader
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True, drop_last=True)
        dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, 
                              num_workers=0, pin_memory=True, drop_last=False)

        # build model
        torch.manual_seed(k)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet1d(
            input_channels=X.shape[1],
            initial_channels=initial_channels_num, #should be mutiply of feature channels
            initial_kernel_size=15,
            down_channels=(initial_channels_num, initial_channels_num, initial_channels_num),
            down_kernel_size=(12, 15, 15),
            down_stride=(12, 9, 5),  # first element must be 12
            res_depth=3,
            res_kernel_size=15,
            se_ratio=8,
            out_kernel_size=21,
        )
        if fold == 0 and k==0:
            print(summary(
                model=model,
                input_size=(batch_size, X.shape[1], X.shape[2]),
                col_names=["input_size", "output_size", "num_params"],
                col_width=20
            ))
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        # scheduler = CustomCosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5, lr_decay_factor=0.9)
        criterion = nn.MSELoss()

        best_loss = np.inf
        early_stopping = EarlyStopping(patience=20)

        for epoch in range(n_epochs):
            train(model, dl_train, optimizer, criterion, device)
            loss = evaluate(model, dl_valid, criterion, device)

            # Pass the validation loss to the scheduler
            scheduler.step()
            # scheduler.step(loss)

            early_stopping(loss)
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            current_lr = optimizer.param_groups[0]['lr']
            if loss < best_loss:
                best_loss = loss
                model_path = f"model_{k}_{fold}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"epoch: {epoch}\tvalid-loss: {loss}\tCurrent Learning Rate: {current_lr}\tbest!")
            else:
                print(f"epoch: {epoch}\tvalid-loss: {loss}\tCurrent Learning Rate: {current_lr}")

        with torch.no_grad():
            model.load_state_dict(torch.load(f"model_{k}_{fold}.pth", map_location=device))
            preds = predict(model, dl_valid, device)
            if k == 0:
                preds1[idx_valid] = preds
            else:
                preds2[idx_valid] = preds
            preds_valid[idx_valid] = predict(model, dl_valid, device)

        fold_rmse = mean_squared_error(Y[idx_valid], preds_valid[idx_valid], squared=False)
        if fold_rmse < best_model_rmse:
            best_model_rmse = fold_rmse

        print(f"fold: {fold} best_loss: {best_loss} RMSE: {fold_rmse}")
        print()

# rmse = mean_squared_error(Y, preds_valid, squared=False)
# print(f"RMSE: {rmse:.5f}")
preds_valid = (preds1 + preds2)/2  

df_pred = pd.DataFrame(preds_valid, index=df_y.index, columns=df_y.columns)
df_pred = df_pred.stack().reset_index(name="score")
df_pred = pd.merge(
    df_1min[["series_id", "date", "time", "step", "event"]],
    df_pred,
    on=["series_id", "date", "time"],
    how="inner"
)

def sub_with_pp(df_pred_test):
    # pp setup
    df_events = pd.read_csv(INPUT_DIR + "train_events.csv").dropna()
    df_events["timestamp"] = pd.to_datetime(df_events["timestamp"], utc=True).dt.tz_localize(None)
    df_events["time"] = df_events["timestamp"].dt.time.astype(str)
    df_events["minute_mod15"] = df_events["timestamp"].dt.minute % 15

    df_agg = df_events.groupby(["time", "event"], as_index=False).size()
    df_agg["rate"] = df_agg["size"] / df_agg.groupby("event")["size"].transform("sum") * (60*24)
    df_time = df_agg.pivot(index="time", columns="event", values="rate").fillna(0).reset_index()
    df_time = df_time.merge(df_pred_test[["time"]].drop_duplicates(), how="right").fillna(0)
    df_time = pd.concat([df_time]*3, ignore_index=True)
    df_time["onset"] = df_time["onset"].rolling(60, center=True).mean()
    df_time["wakeup"] = df_time["wakeup"].rolling(60, center=True).mean()
    df_time = df_time.iloc[60*24:-60*24].reset_index(drop=True)

    df_agg = df_events.groupby(["minute_mod15", "event"], as_index=False).size()
    df_agg["rate"] = df_agg["size"] / df_agg.groupby("event")["size"].transform("sum") * 15
    df_minute = df_agg.pivot(index="minute_mod15", columns="event", values="rate").reset_index()

    df_agg = df_events.groupby(["minute_mod15", "event"], as_index=False).size()
    df_agg["rate"] = df_agg["size"] / df_agg.groupby("event")["size"].transform("sum") * 15
    df_minute = df_agg.pivot(index="minute_mod15", columns="event", values="rate").reset_index()

    df_time[["onset", "wakeup"]] = df_time[["onset", "wakeup"]].clip(0.1, 1.1) ** 0.13
    df_minute[["onset", "wakeup"]] = df_minute[["onset", "wakeup"]].clip(0.5, 1.3) ** 0.06
    
    df_pred_test["minute_mod15"] = df_pred_test["time"].str[3:5].astype(int) % 15

    list_df = []
    for series_id, df in tqdm(df_pred_test.groupby("series_id")):
        df = df.merge(df_time, how="left", on="time")
        df = df.merge(df_minute, how="left", on="minute_mod15")

        df_tmp = df.copy()
        df_tmp["score"] = df_tmp["score"].replace(0.0, np.nan)
        df_tmp = df_tmp.groupby("time")["score"].mean()
        df_tmp = pd.concat([df_tmp]*3).rolling(90, center=True, min_periods=1).mean()
        df_tmp = df_tmp.iloc[60*24:-60*24].reset_index().rename({"score": "score_mean"}, axis=1)
        df = df.merge(df_tmp, on="time", how="left")

        df["score"] = 0.9*df["score"] + 0.1*df["score_mean"]
        df["score"] *= np.where(df["score"]>0, df["wakeup_x"], df["onset_x"])
        df["score"] *= np.where(df["score"]>0, df["wakeup_y"], df["onset_y"])
        valid_ratio = dict_valid_ratio[series_id]

        for event in ["onset", "wakeup"]:
            values_step = df["step"].values
            if event == "onset":
                values_score = -df["score"].values
            else:
                values_score = df["score"].values

            # measure peaks
            peak_idx = find_peaks(values_score, height=0.04, distance=60*16)[0]  # at least 16 hours interval
            df_measure_peak = pd.DataFrame(values_step[peak_idx], columns=["step"])
            df_measure_peak["series_id"] = series_id
            df_measure_peak["event"] = event
            df_measure_peak["score"] = values_score[peak_idx] * 4 * valid_ratio**0.15

            # minor peaks
            peak_idx = find_peaks(values_score, height=0.0, distance=6)[0]
            df_minor_peak = pd.DataFrame(values_step[peak_idx], columns=["step"])
            df_minor_peak["series_id"] = series_id
            df_minor_peak["event"] = event
            df_minor_peak["score"] = values_score[peak_idx]

            df_peak = pd.concat([df_measure_peak, df_minor_peak]).drop_duplicates(subset=["step"])
            list_df.append(df_peak)

    df_sub = pd.concat(list_df)
    df_sub = df_sub.sort_values("score", ascending=False).groupby("event").head(100000)  # avoid Submission Scoring Error
    df_sub = df_sub.sort_values(["series_id", "step"]).reset_index(drop=True)
    df_sub = df_sub[["series_id", "step", "event", "score"]].reset_index(names="row_id")
    
    return df_sub

df_sub = sub_with_pp(df_pred)

##########################################

from bisect import bisect_left
from typing import Dict, List, Tuple


class ParticipantVisibleError(Exception):
    pass


# Set some placeholders for global parameters
series_id_column_name = "series_id"
time_column_name = "step"
event_column_name = "event"
score_column_name = "score"
use_scoring_intervals = False
tolerances = {
    "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]],
    series_id_column_name: str,
    time_column_name: str,
    event_column_name: str,
    score_column_name: str,
    use_scoring_intervals: bool = False,
    verbose: bool = False,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    # Validate metric parameters
    assert len(tolerances) > 0, "Events must have defined tolerances."
    assert set(tolerances.keys()) == set(solution[event_column_name]).difference(
        {"start", "end"}
    ), (
        f"Solution column {event_column_name} must contain the same events "
        "as defined in tolerances."
    )
    assert pd.api.types.is_numeric_dtype(
        solution[time_column_name]
    ), f"Solution column {time_column_name} must be of numeric type."

    # Validate submission format
    for column_name in [
        series_id_column_name,
        time_column_name,
        event_column_name,
        score_column_name,
    ]:
        if column_name not in submission.columns:
            raise ParticipantVisibleError(
                f"Submission must have column '{column_name}'."
            )

    if not pd.api.types.is_numeric_dtype(submission[time_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{time_column_name}' must be of numeric type."
        )
    if not pd.api.types.is_numeric_dtype(submission[score_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{score_column_name}' must be of numeric type."
        )

    # Set these globally to avoid passing around a bunch of arguments
    globals()["series_id_column_name"] = series_id_column_name
    globals()["time_column_name"] = time_column_name
    globals()["event_column_name"] = event_column_name
    globals()["score_column_name"] = score_column_name
    globals()["use_scoring_intervals"] = use_scoring_intervals

    return event_detection_ap(solution, submission, tolerances, verbose=verbose)


def find_nearest(xs: np.ndarray, value):
    """
    Find the index of the closest value to x in the array xs.
    """
    idx = np.searchsorted(xs, value, side="left")
    best_idx = None
    best_error = float("inf")
    best_diff = float("inf")

    range_min = max(0, idx - 1)
    range_max = min(len(xs), idx + 2)
    for check_idx in range(
        range_min, range_max
    ):  # Check the exact, one before, and one after
        error = abs(xs[check_idx] - value)
        if error < best_error:
            best_error = error
            best_idx = check_idx
            best_diff = xs[check_idx] - value

    return best_idx, best_error, best_diff


def find_nearest_time_idx(sorted_gt_times, det_time, excluded_indices: set):
    """
    search index of gt_times closest to det_time.

    assumes gt_times is sorted in ascending order.
    """
    # e.g. if gt_times = [0, 1, 2, 3, 4, 5] and det_time = 2.5, then idx = 3
    sorted_gt_times = np.asarray(sorted_gt_times)
    available_indices = np.asarray(
        sorted(set(range(len(sorted_gt_times))) - excluded_indices), dtype=int
    )
    sorted_gt_times = sorted_gt_times[available_indices]
    idx, error, diff = find_nearest(sorted_gt_times, det_time)
    best_idx = available_indices[idx] if idx is not None else None

    return best_idx, error, diff


def match_detections(
    tolerance: float, ground_truths: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    detections_sorted = detections.sort_values(
        score_column_name, ascending=False
    ).dropna()
    is_matched = np.full_like(detections_sorted[event_column_name], False, dtype=bool)
    diffs = np.full_like(
        detections_sorted[event_column_name], float("inf"), dtype=float
    )
    ground_truths_times = ground_truths.sort_values(time_column_name)[
        time_column_name
    ].to_list()
    matched_gt_indices: set[int] = set()

    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        det_time = getattr(det, time_column_name)

        best_idx, best_error, best_diff = find_nearest_time_idx(
            ground_truths_times, det_time, matched_gt_indices
        )

        if (best_idx is not None) and (best_error < tolerance):
            is_matched[i] = True
            diffs[i] = best_diff
            matched_gt_indices.add(best_idx)

    detections_sorted["matched"] = is_matched
    detections_sorted["diff"] = diffs
    return detections_sorted


def precision_recall_curve(
    matches: np.ndarray, scores: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind="stable")[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = (
        tps / p
    )  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def event_detection_ap(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]] = tolerances,
    progress_bar: bool = True,
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    # Ensure solution and submission are sorted properly
    solution = solution.sort_values([series_id_column_name, time_column_name])
    submission = submission.sort_values([series_id_column_name, time_column_name])

    # Extract scoring intervals.
    if use_scoring_intervals:
        raise NotImplementedError("Scoring intervals not implemented.")

    # Extract ground-truth events.
    ground_truths = solution.query("event not in ['start', 'end']").reset_index(
        drop=True
    )

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts(event_column_name).to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched=False)

    # Remove detections outside of scoring intervals
    if use_scoring_intervals:
        raise NotImplementedError("Scoring intervals not implemented.")
    else:
        detections_filtered = detections

    # Create table of event-class x tolerance x series_id values
    aggregation_keys = pd.DataFrame(
        [
            (ev, tol, vid)
            for ev in tolerances.keys()
            for tol in tolerances[ev]
            for vid in ground_truths[series_id_column_name].unique()
        ],
        columns=[event_column_name, "tolerance", series_id_column_name],
    )

    # Create match evaluation groups: event-class x tolerance x series_id
    detections_grouped = aggregation_keys.merge(
        detections_filtered, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])
    ground_truths_grouped = aggregation_keys.merge(
        ground_truths, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])

    # Match detections to ground truth events by evaluation group
    pbars = aggregation_keys.itertuples(index=False)
    if progress_bar:
        pbars = tqdm(pbars, total=len(aggregation_keys), desc="Matching detections")
    detections_matched = []
    for key in pbars:
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(
            match_detections(dets["tolerance"].iloc[0], gts, dets)
        )
    detections_matched = pd.concat(detections_matched)

    # Compute AP per event x tolerance group
    event_classes = ground_truths[event_column_name].unique()
    ap_table = (
        detections_matched.query("event in @event_classes")
        .groupby([event_column_name, "tolerance"])
        .apply(
            lambda group: average_precision_score(
                group["matched"].to_numpy(),
                group[score_column_name].to_numpy(),
                class_counts[group[event_column_name].iat[0]],
            )
        )
        .reset_index()
        .pivot(index="tolerance", columns="event", values=0)
    )
    if verbose:
        display(ap_table)
    # Average over tolerances, then over event classes
    mean_ap = ap_table.mean().mean()

    return mean_ap, ap_table, detections_matched

#######################################

df_solution = pd.read_csv(INPUT_DIR + "train_events.csv").dropna()
df_solution = df_solution[df_solution['series_id'] != '05e1944c3818']
score_all, df_score, df_result = score(
    solution=df_solution,
    submission=df_sub,
    tolerances=tolerances,
    series_id_column_name=series_id_column_name,
    time_column_name=time_column_name,
    event_column_name=event_column_name,
    score_column_name=score_column_name
)
# print(score_all)
nni.report_final_result(score_all)