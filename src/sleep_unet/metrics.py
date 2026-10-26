"""Event detection average precision from the training notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class ParticipantVisibleError(Exception):
    pass


series_id_column_name = "series_id"
time_column_name = "step"
event_column_name = "event"
score_column_name = "score"
tolerances = {
    "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: dict[str, list[float]],
    series_id_column_name: str,
    time_column_name: str,
    event_column_name: str,
    score_column_name: str,
    use_scoring_intervals: bool = False,
    verbose: bool = True,
    progress_bar: bool = True,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    if use_scoring_intervals:
        raise NotImplementedError("Scoring intervals not implemented.")
    assert len(tolerances) > 0, "Events must have defined tolerances."
    assert set(tolerances.keys()) == set(solution[event_column_name]).difference(
        {"start", "end"}
    ), f"Solution column {event_column_name} must contain the same events as defined in tolerances."
    assert pd.api.types.is_numeric_dtype(solution[time_column_name]), (
        f"Solution column {time_column_name} must be of numeric type."
    )

    for column_name in [
        series_id_column_name,
        time_column_name,
        event_column_name,
        score_column_name,
    ]:
        if column_name not in submission.columns:
            raise ParticipantVisibleError(f"Submission must have column '{column_name}'.")

    if not pd.api.types.is_numeric_dtype(submission[time_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{time_column_name}' must be of numeric type."
        )
    if not pd.api.types.is_numeric_dtype(submission[score_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{score_column_name}' must be of numeric type."
        )

    columns = {
        series_id_column_name: "series_id",
        time_column_name: "step",
        event_column_name: "event",
        score_column_name: "score",
    }
    ap, table, matched = event_detection_ap(
        solution.rename(columns=columns),
        submission.rename(columns=columns),
        tolerances,
        progress_bar=progress_bar,
        verbose=verbose,
    )
    table.columns.name = event_column_name
    return ap, table, matched.rename(columns={value: key for key, value in columns.items()})


def find_nearest(xs: np.ndarray, value):
    """Find the index of the closest value to x in the array xs."""
    idx = np.searchsorted(xs, value, side="left")
    best_idx = None
    best_error = float("inf")
    best_diff = float("inf")

    range_min = max(0, idx - 1)
    range_max = min(len(xs), idx + 2)
    for check_idx in range(range_min, range_max):
        error = abs(xs[check_idx] - value)
        if error < best_error:
            best_error = error
            best_idx = check_idx
            best_diff = xs[check_idx] - value

    return best_idx, best_error, best_diff


def find_nearest_time_idx(sorted_gt_times, det_time, excluded_indices: set):
    """Search index of gt_times closest to det_time. gt_times is sorted ascending."""
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
    detections_sorted = detections.sort_values(score_column_name, ascending=False).dropna()
    is_matched = np.full_like(detections_sorted[event_column_name], False, dtype=bool)
    diffs = np.full_like(detections_sorted[event_column_name], float("inf"), dtype=float)
    ground_truths_times = ground_truths.sort_values(time_column_name)[time_column_name].to_list()
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []

    idxs = np.argsort(scores, kind="stable")[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / p

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    return float(-np.sum(np.diff(recall) * np.array(precision)[:-1]))


def event_detection_ap(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: dict[str, list[float]] = tolerances,
    progress_bar: bool = True,
    verbose: bool = True,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    solution = solution.sort_values([series_id_column_name, time_column_name])
    submission = submission.sort_values([series_id_column_name, time_column_name])

    ground_truths = solution.query("event not in ['start', 'end']").reset_index(drop=True)
    class_counts = ground_truths[event_column_name].value_counts().to_dict()
    detections = submission.assign(matched=False)

    aggregation_keys = pd.DataFrame(
        [
            (ev, tol, vid)
            for ev in tolerances.keys()
            for tol in tolerances[ev]
            for vid in ground_truths[series_id_column_name].unique()
        ],
        columns=[event_column_name, "tolerance", series_id_column_name],
    )

    detections_grouped = aggregation_keys.merge(
        detections, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])
    ground_truths_grouped = aggregation_keys.merge(
        ground_truths, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])

    pbars = aggregation_keys.itertuples(index=False)
    if progress_bar:
        pbars = tqdm(pbars, total=len(aggregation_keys), desc="Matching detections")
    detections_matched = []
    for key in pbars:
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(match_detections(dets["tolerance"].iloc[0], gts, dets))
    detections_matched = pd.concat(detections_matched)

    event_classes = set(ground_truths[event_column_name].unique())
    rows = []
    matched = detections_matched.loc[detections_matched[event_column_name].isin(event_classes)]
    for (event_name, tolerance), group in matched.groupby(
        [event_column_name, "tolerance"], sort=False
    ):
        rows.append(
            {
                "event": event_name,
                "tolerance": tolerance,
                "ap": average_precision_score(
                    group["matched"].to_numpy(),
                    group[score_column_name].to_numpy(),
                    class_counts[event_name],
                ),
            }
        )
    if not rows:
        ap_table = pd.DataFrame(columns=["onset", "wakeup"])
        return 0.0, ap_table, detections_matched
    ap_table = pd.DataFrame(rows).pivot(index="tolerance", columns="event", values="ap")
    if verbose:
        print(ap_table.to_string())
    mean_ap = float(ap_table.mean().mean())
    return mean_ap, ap_table, detections_matched


def score_events(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    *,
    verbose: bool = False,
    progress_bar: bool = False,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Score onset/wakeup steps. Drops the series removed in the notebook."""
    solution = solution.dropna(subset=["series_id", "step", "event"]).copy()
    solution = solution.loc[solution["series_id"].astype(str) != "05e1944c3818"]
    solution = solution.loc[solution["event"].isin(["onset", "wakeup"])]
    return score(
        solution=solution,
        submission=submission,
        tolerances=tolerances,
        series_id_column_name=series_id_column_name,
        time_column_name=time_column_name,
        event_column_name=event_column_name,
        score_column_name=score_column_name,
        use_scoring_intervals=False,
        verbose=verbose,
        progress_bar=progress_bar,
    )
