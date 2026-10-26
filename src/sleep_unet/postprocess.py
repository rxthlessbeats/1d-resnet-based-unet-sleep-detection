"""Peak submission from ``unet/unet.ipynb``, with and without the time-of-day prior."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sleep_unet.data import MINUTE_TIMES, PreparedWindows


def scores_to_long(prepared: PreparedWindows, preds: np.ndarray) -> pd.DataFrame:
    wide = pd.DataFrame(preds, index=prepared.index, columns=MINUTE_TIMES)
    wide.columns.name = "time"
    long = wide.stack().rename("score").reset_index()
    columns = ["series_id", "date", "time", "step", "event"]
    return pd.merge(
        prepared.df_1min[columns],
        long,
        on=["series_id", "date", "time"],
        how="inner",
    )


def _empty_submission() -> pd.DataFrame:
    return pd.DataFrame(columns=["row_id", "series_id", "step", "event", "score"])


def _finish(list_df: list[pd.DataFrame]) -> pd.DataFrame:
    if not list_df:
        return _empty_submission()
    df_sub = pd.concat(list_df, ignore_index=True)
    df_sub = df_sub.sort_values("score", ascending=False).groupby("event").head(100000)
    df_sub = df_sub.sort_values(["series_id", "step"]).reset_index(drop=True)
    return df_sub[["series_id", "step", "event", "score"]].reset_index(names="row_id")


def _peak_indexes(values_score: np.ndarray, height: float, distance: int) -> np.ndarray:
    peak_idx = find_peaks(values_score, height=height, distance=distance)[0]
    if peak_idx.size == 0 and values_score.size:
        peak_idx = np.array([int(np.nanargmax(np.nan_to_num(values_score, nan=-np.inf)))])
    return peak_idx


def submission_without_priors(df_pred: pd.DataFrame) -> pd.DataFrame:
    list_df = []
    for series_id, df in df_pred.groupby("series_id"):
        for event in ["onset", "wakeup"]:
            values_step = df["step"].to_numpy()
            values_score = -df["score"].to_numpy() if event == "onset" else df["score"].to_numpy()
            values_score = np.nan_to_num(values_score, nan=0.0)
            peak_idx = _peak_indexes(values_score, height=0.0, distance=8)
            df_peak = pd.DataFrame(values_step[peak_idx], columns=["step"])
            df_peak["series_id"] = series_id
            df_peak["event"] = event
            df_peak["score"] = values_score[peak_idx]
            list_df.append(df_peak)
    return _finish(list_df)


def _time_priors(df_pred: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.dropna(subset=["timestamp", "event"]).copy()
    events = events[events["event"].isin(["onset", "wakeup"])].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True).dt.tz_localize(None)
    events["time"] = events["timestamp"].dt.strftime("%H:%M:%S")
    events["minute_mod15"] = events["timestamp"].dt.minute % 15

    df_agg = events.groupby(["time", "event"], as_index=False).size()
    df_agg["rate"] = df_agg["size"] / df_agg.groupby("event")["size"].transform("sum") * (60 * 24)
    df_time = df_agg.pivot(index="time", columns="event", values="rate").fillna(0).reset_index()
    for column in ("onset", "wakeup"):
        if column not in df_time.columns:
            df_time[column] = 0.0
    df_time = df_time.merge(df_pred[["time"]].drop_duplicates(), how="right", on="time").fillna(0)
    n_min = 60 * 24
    if len(df_time) == n_min:
        df_time = pd.concat([df_time] * 3, ignore_index=True)
        df_time["onset"] = df_time["onset"].rolling(60, center=True).mean()
        df_time["wakeup"] = df_time["wakeup"].rolling(60, center=True).mean()
        df_time = df_time.iloc[n_min:-n_min].reset_index(drop=True)
    else:
        df_time["onset"] = df_time["onset"].rolling(60, center=True, min_periods=1).mean()
        df_time["wakeup"] = df_time["wakeup"].rolling(60, center=True, min_periods=1).mean()

    df_agg = events.groupby(["minute_mod15", "event"], as_index=False).size()
    df_agg["rate"] = df_agg["size"] / df_agg.groupby("event")["size"].transform("sum") * 15
    df_minute = (
        df_agg.pivot(index="minute_mod15", columns="event", values="rate")
        .reindex(index=range(15), columns=["onset", "wakeup"])
        .fillna(0)
        .reset_index()
    )

    df_time[["onset", "wakeup"]] = df_time[["onset", "wakeup"]].clip(0.1, 1.1) ** 0.13
    df_minute[["onset", "wakeup"]] = df_minute[["onset", "wakeup"]].clip(0.5, 1.3) ** 0.06
    return df_time, df_minute


def submission_with_priors(
    df_pred: pd.DataFrame,
    events: pd.DataFrame,
    valid_ratio: dict[str, float],
) -> pd.DataFrame:
    if events.empty or not set(events["event"]).intersection({"onset", "wakeup"}):
        return submission_without_priors(df_pred)

    df_time, df_minute = _time_priors(df_pred, events)
    df_pred = df_pred.copy()
    df_pred["minute_mod15"] = df_pred["time"].str[3:5].astype(int) % 15
    list_df = []
    n_min = 60 * 24
    for series_id, df in df_pred.groupby("series_id"):
        df = df.merge(df_time, how="left", on="time")
        df = df.merge(df_minute, how="left", on="minute_mod15")

        df_tmp = df.copy()
        df_tmp["score"] = df_tmp["score"].replace(0.0, np.nan)
        df_tmp = df_tmp.groupby("time")["score"].mean()
        if len(df_tmp) == n_min:
            df_tmp = pd.concat([df_tmp] * 3).rolling(90, center=True, min_periods=1).mean()
            df_tmp = df_tmp.iloc[n_min:-n_min].reset_index().rename({"score": "score_mean"}, axis=1)
        else:
            smoothed = df_tmp.rolling(90, center=True, min_periods=1).mean()
            df_tmp = smoothed.reset_index().rename({"score": "score_mean"}, axis=1)
        df = df.merge(df_tmp, on="time", how="left")

        df["score"] = 0.9 * df["score"] + 0.1 * df["score_mean"]
        df["score"] *= np.where(df["score"] > 0, df["wakeup_x"], df["onset_x"])
        df["score"] *= np.where(df["score"] > 0, df["wakeup_y"], df["onset_y"])
        ratio = float(valid_ratio.get(series_id, 1.0))

        for event in ["onset", "wakeup"]:
            values_step = df["step"].to_numpy()
            values_score = -df["score"].to_numpy() if event == "onset" else df["score"].to_numpy()
            values_score = np.nan_to_num(values_score, nan=0.0)

            peak_idx = find_peaks(values_score, height=0.04, distance=60 * 16)[0]
            df_measure_peak = pd.DataFrame(values_step[peak_idx], columns=["step"])
            df_measure_peak["series_id"] = series_id
            df_measure_peak["event"] = event
            df_measure_peak["score"] = values_score[peak_idx] * 4 * ratio**0.15

            minor_idx = _peak_indexes(values_score, height=0.0, distance=6)
            df_minor_peak = pd.DataFrame(values_step[minor_idx], columns=["step"])
            df_minor_peak["series_id"] = series_id
            df_minor_peak["event"] = event
            df_minor_peak["score"] = values_score[minor_idx]

            df_peak = pd.concat([df_measure_peak, df_minor_peak]).drop_duplicates(subset=["step"])
            list_df.append(df_peak)
    return _finish(list_df)
