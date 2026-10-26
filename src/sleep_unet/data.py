"""Window features from ``unet/unet.ipynb`` (log anglez, log ENMO, valid flag)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset

from sleep_unet.model import CONTEXT_STEPS, DAY_STEPS, MINUTE_STEPS

FEATURES = ["log_anglez_std", "log_enmo", "valid_flag"]
# Series whose event labels are excluded in the training notebook and the scorer.
DROP_SERIES = ["05e1944c3818"]
FILTERS = 24

FIVE_SEC_TIMES = [
    f"{hour:02d}:{minute:02d}:{second:02d}"
    for hour in range(24)
    for minute in range(60)
    for second in range(0, 60, 5)
]
MINUTE_TIMES = [f"{hour:02d}:{minute:02d}:00" for hour in range(24) for minute in range(60)]

if len(FIVE_SEC_TIMES) != DAY_STEPS or len(MINUTE_TIMES) != MINUTE_STEPS:
    raise RuntimeError("canonical day grids do not match the U-Net strides")


class MyDataset(Dataset):
    def __init__(self, X, Y, flag):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        if Y is not None:
            self.Y = torch.as_tensor(Y, dtype=torch.float32)
        self.flag = torch.as_tensor(flag, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if hasattr(self, "Y"):
            return (self.X[idx], self.Y[idx], self.flag[idx])
        return (self.X[idx], torch.Tensor(), self.flag[idx])


@dataclass
class PreparedWindows:
    X: np.ndarray
    Y: np.ndarray
    mask: np.ndarray
    df_1min: pd.DataFrame
    index: pd.MultiIndex
    valid_ratio: dict[str, float]
    norm_min: np.ndarray
    norm_max: np.ndarray
    features: list[str]

    @property
    def df_y(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.Y, index=self.index, columns=MINUTE_TIMES)
        frame.columns.name = "time"
        return frame

    @property
    def df_mask(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.mask, index=self.index, columns=MINUTE_TIMES)
        frame.columns.name = "time"
        return frame


def spread_event_targets(values_event: np.ndarray) -> np.ndarray:
    """Two-sided exponential target used by the training notebook."""
    values_event = np.asarray(values_event, dtype=np.float64)
    values_target = values_event.copy()
    for j in range(30):
        weight = np.exp(-j / 2.8)
        values_target[: -(j + 1)] += weight * values_event[(j + 1) :]
        if j > 0:
            values_target[j:] += weight * values_event[:-j]
    return values_target


def encode_events(events: pd.DataFrame) -> pd.DataFrame:
    """Map onset/wakeup labels to the signed targets the notebook regresses."""
    if events.empty:
        return events.copy()
    mapped = events["event"].map({"wakeup": 1.0, "onset": -1.0})
    numeric = pd.to_numeric(events["event"], errors="coerce")
    out = events.copy()
    out["event"] = mapped.fillna(numeric)
    out = out.dropna(subset=["event", "timestamp"]).copy()
    out["event"] = out["event"].astype("float32")
    out["timestamp"] = _naive_timestamp(out["timestamp"])
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
    else:
        frame = pl.read_csv(path)
    return frame.to_pandas()


def _naive_timestamp(values) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_localize(None)


def _one_series(df: pd.DataFrame, events: pd.DataFrame, features: list[str]):
    series_id = str(df["series_id"].iloc[0])
    df = df.copy()
    df["timestamp"] = _naive_timestamp(df["timestamp"])
    df = df.sort_values(["timestamp", "step"])
    if not events.empty:
        ev = events.loc[events["series_id"].astype(str) == series_id, ["timestamp", "event"]]
        df = df.merge(ev, on="timestamp", how="left")
    if "event" not in df.columns:
        df["event"] = 0.0
    df["event"] = df["event"].fillna(0.0).astype("float64")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")

    dup_count = df.groupby(["anglez", "enmo", "time"])["step"].transform("count")
    df["valid_flag"] = (dup_count == 1).astype("float32")
    valid_ratio = float(df["valid_flag"].mean())

    df["log_anglez_std"] = np.log(
        df.rolling("60s", on="timestamp", min_periods=1, center=True, closed="left")["anglez"].std()
        + 1
    ).astype("float32")
    df["log_enmo"] = np.log(df["enmo"] + 0.01).astype("float32")

    blocks = []
    feat_index = None
    for feature in features:
        pivoted = df.pivot(index="date", columns="time", values=feature)
        pivoted.index = pd.to_datetime(pivoted.index)
        pivoted = pivoted.reindex(columns=FIVE_SEC_TIMES).sort_index()
        values = pivoted.fillna(0).to_numpy(dtype=np.float32)
        before = (
            pivoted.shift(1, freq="D").reindex(pivoted.index).fillna(0).to_numpy(dtype=np.float32)
        )
        after = (
            pivoted.shift(-1, freq="D").reindex(pivoted.index).fillna(0).to_numpy(dtype=np.float32)
        )
        window = np.concatenate(
            [before[:, -CONTEXT_STEPS:], values, after[:, :CONTEXT_STEPS]],
            axis=1,
        )
        blocks.append(window)
        feat_index = pd.MultiIndex.from_product(
            [[series_id], pivoted.index.strftime("%Y-%m-%d")], names=["series_id", "date"]
        )
    stacked = np.stack(blocks, axis=1)

    indexed = df.set_index("timestamp").sort_index()
    df_1min = (
        indexed.resample("1min")
        .agg(
            {
                "series_id": "first",
                "step": "mean",
                "event": "sum",
                "valid_flag": "max",
            }
        )
        .reset_index()
    )
    df_1min["series_id"] = series_id
    df_1min["date"] = df_1min["timestamp"].dt.strftime("%Y-%m-%d")
    df_1min["time"] = df_1min["timestamp"].dt.strftime("%H:%M:%S")
    df_1min["step"] = df_1min["step"].fillna(0).astype("int32")
    df_1min["event"] = df_1min["event"].fillna(0.0)
    df_1min["valid_flag"] = df_1min["valid_flag"].fillna(0).astype("float32")
    df_1min["target"] = spread_event_targets(df_1min["event"].to_numpy())

    y_pivot = df_1min.pivot(index=["series_id", "date"], columns="time", values="target")
    m_pivot = df_1min.pivot(index=["series_id", "date"], columns="time", values="valid_flag")
    y_pivot = y_pivot.reindex(index=feat_index, columns=MINUTE_TIMES).fillna(0)
    m_pivot = m_pivot.reindex(index=feat_index, columns=MINUTE_TIMES).fillna(0)
    keep = ["timestamp", "series_id", "date", "time", "step", "event", "valid_flag", "target"]
    return (
        stacked,
        y_pivot.to_numpy(dtype=np.float64),
        m_pivot.to_numpy(dtype=np.float32),
        df_1min[keep],
        feat_index,
        valid_ratio,
    )


def normalize_windows(
    X: np.ndarray,
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if normalization is None:
        x_min = X.min(axis=(0, 2), keepdims=True)
        x_max = X.max(axis=(0, 2), keepdims=True)
    else:
        x_min, x_max = (np.asarray(bound, dtype=np.float32) for bound in normalization)
    shape = (1, X.shape[1], 1)
    if x_min.shape != shape or x_max.shape != shape:
        raise ValueError("normalization bounds must match the feature channels")
    if not np.isfinite(x_min).all() or not np.isfinite(x_max).all() or (x_max < x_min).any():
        raise ValueError("normalization bounds must be finite and ordered")
    denom = x_max - x_min
    denom = np.where(denom == 0, 1.0, denom)
    scaled = ((X - x_min) / denom).astype(np.float32)
    return scaled, x_min.astype(np.float32), x_max.astype(np.float32)


def prepare_windows(
    series: pd.DataFrame,
    events: pd.DataFrame | None = None,
    features: list[str] | None = None,
    drop_series: list[str] | tuple[str, ...] | None = DROP_SERIES,
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> PreparedWindows:
    features = list(features or FEATURES)
    series = series.copy()
    series["series_id"] = series["series_id"].astype(str)
    if drop_series:
        series = series.loc[~series["series_id"].isin(drop_series)].copy()
    if series.empty:
        raise ValueError("no series left after filtering")

    if events is None or events.empty:
        encoded = pd.DataFrame(columns=["series_id", "timestamp", "event"])
    else:
        encoded = encode_events(events)
        encoded["series_id"] = encoded["series_id"].astype(str)
        if drop_series:
            encoded = encoded.loc[~encoded["series_id"].isin(drop_series)].copy()
        if not encoded.empty:
            encoded = encoded.groupby(["series_id", "timestamp"], as_index=False)["event"].sum()

    blocks_x = []
    blocks_y = []
    blocks_m = []
    frames = []
    indices = []
    valid_ratio: dict[str, float] = {}
    for series_id, part in series.groupby("series_id", sort=False, dropna=False):
        stacked, y, mask, df_1min, feat_index, ratio = _one_series(part, encoded, features)
        if stacked.shape[-1] != DAY_STEPS + 2 * CONTEXT_STEPS:
            raise ValueError(
                f"{series_id} produced length {stacked.shape[-1]}, expected a full day plus context"
            )
        blocks_x.append(stacked)
        blocks_y.append(y)
        blocks_m.append(mask)
        frames.append(df_1min)
        indices.append(feat_index)
        valid_ratio[series_id] = ratio

    X = np.concatenate(blocks_x, axis=0)
    Y = np.concatenate(blocks_y, axis=0)
    mask = np.concatenate(blocks_m, axis=0)
    X, norm_min, norm_max = normalize_windows(X, normalization)
    index = indices[0].append(indices[1:])
    if len(index) != len(X) or len(Y) != len(X):
        raise RuntimeError(f"window alignment failed: X {len(X)} Y {len(Y)} index {len(index)}")
    return PreparedWindows(
        X=X,
        Y=Y,
        mask=mask,
        df_1min=pd.concat(frames, ignore_index=True),
        index=index,
        valid_ratio=valid_ratio,
        norm_min=norm_min,
        norm_max=norm_max,
        features=features,
    )


def save_prepared(prepared: PreparedWindows, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(
        path / "windows.npz",
        X=prepared.X,
        Y=prepared.Y,
        mask=prepared.mask,
        norm_min=prepared.norm_min,
        norm_max=prepared.norm_max,
    )
    prepared.df_1min.to_parquet(path / "df_1min.parquet", index=False)
    prepared.index.to_frame(index=False).to_parquet(path / "index.parquet", index=False)
    meta = {
        "features": prepared.features,
        "valid_ratio": prepared.valid_ratio,
    }
    (path / "meta.json").write_text(json.dumps(meta))


def load_prepared(path: str | Path) -> PreparedWindows:
    path = Path(path)
    arrays = np.load(path / "windows.npz")
    meta = json.loads((path / "meta.json").read_text())
    index_frame = pd.read_parquet(path / "index.parquet")
    index = pd.MultiIndex.from_frame(index_frame)
    index = index.set_names(["series_id", "date"])
    return PreparedWindows(
        X=arrays["X"],
        Y=arrays["Y"],
        mask=arrays["mask"],
        df_1min=pd.read_parquet(path / "df_1min.parquet"),
        index=index,
        valid_ratio={str(k): float(v) for k, v in meta["valid_ratio"].items()},
        norm_min=arrays["norm_min"],
        norm_max=arrays["norm_max"],
        features=list(meta["features"]),
    )


def make_synthetic(
    n_series: int = 2, n_days: int = 1, origin: str = "2018-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regular 5-second series with one onset and one wakeup each day. No external files."""
    series_rows = []
    event_rows = []
    origin_ts = pd.Timestamp(origin)
    for i in range(n_series):
        series_id = f"synth{i:02d}"
        start = origin_ts + pd.Timedelta(days=i * n_days)
        times = pd.date_range(start, periods=n_days * DAY_STEPS, freq="5s")
        hour = times.hour + times.minute / 60 + times.second / 3600
        anglez = (30.0 * np.sin(2 * np.pi * hour / 24)).astype(np.float32)
        anglez = anglez + (0.05 * np.sin(np.arange(len(times)) / 12)).astype(np.float32)
        enmo = np.where((hour >= 8) & (hour <= 22), 0.08, 0.001).astype(np.float32)
        series_rows.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "step": np.arange(len(times), dtype=np.int32),
                    "timestamp": times,
                    "anglez": anglez,
                    "enmo": enmo,
                }
            )
        )
        for day in range(n_days):
            day_start = start + pd.Timedelta(days=day)
            for hour_of, label in ((7, "wakeup"), (23, "onset")):
                timestamp = day_start + pd.Timedelta(hours=hour_of)
                step = int((timestamp - start) / pd.Timedelta(seconds=5))
                event_rows.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "timestamp": timestamp,
                        "event": label,
                    }
                )
    return pd.concat(series_rows, ignore_index=True), pd.DataFrame(event_rows)
