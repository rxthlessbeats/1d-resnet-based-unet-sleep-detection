"""Contracts for the U-Net notebook path. The synthetic demo is the end-to-end run."""

import numpy as np
import pandas as pd
import torch

from sleep_unet.data import (
    DAY_STEPS,
    DROP_SERIES,
    MINUTE_TIMES,
    spread_event_targets,
    make_synthetic,
    prepare_windows,
    save_prepared,
    load_prepared,
)
from sleep_unet.metrics import score_events
from sleep_unet.model import WINDOW_STEPS, build_unet
from sleep_unet.postprocess import scores_to_long, submission_with_priors
from sleep_unet.train import train_one_epoch


def test_target_kernel_matches_notebook():
    events = np.zeros(10, dtype=np.float64)
    events[5] = 1.0
    target = spread_event_targets(events)
    assert target[5] == 1.0
    assert np.isclose(target[4], 1.0)
    assert np.isclose(target[6], np.exp(-1 / 2.8))


def test_prepare_shapes_roundtrip_and_dropped_series(tmp_path):
    series, events = make_synthetic(n_series=1, n_days=1)
    bad = series.copy()
    bad["series_id"] = DROP_SERIES[0]
    series = pd.concat([series, bad], ignore_index=True)
    prepared = prepare_windows(series, events)
    assert prepared.X.shape == (1, 3, WINDOW_STEPS)
    assert prepared.Y.shape == (1, len(MINUTE_TIMES))
    assert prepared.mask.shape == prepared.Y.shape
    assert WINDOW_STEPS == DAY_STEPS + 2 * 180 * 12
    assert np.isfinite(prepared.X).all()
    assert DROP_SERIES[0] not in set(prepared.index.get_level_values("series_id"))
    # A constant channel (no range) stays finite instead of dividing by zero.
    assert np.isfinite(prepared.norm_max - prepared.norm_min).all()

    save_prepared(prepared, tmp_path / "prepared")
    loaded = load_prepared(tmp_path / "prepared")
    assert loaded.X.shape == prepared.X.shape
    assert np.allclose(loaded.X, prepared.X)
    assert loaded.features == ["log_anglez_std", "log_enmo", "valid_flag"]


def test_unet_returns_minute_vector_and_trains_one_step():
    torch.manual_seed(0)
    model = build_unet(input_channels=3, initial_channels=12, res_depth=1)
    model.train()
    batch = (
        torch.randn(2, 3, WINDOW_STEPS),
        torch.randn(2, len(MINUTE_TIMES)),
        torch.ones(2, len(MINUTE_TIMES)),
    )
    train_one_epoch(
        model,
        [batch],
        torch.optim.Adam(model.parameters(), lr=1e-3),
        torch.nn.MSELoss(),
        torch.device("cpu"),
    )
    model.eval()
    with torch.no_grad():
        out = model(batch[0])
    assert out.shape == (2, len(MINUTE_TIMES))
    assert torch.isfinite(out).all()


def test_score_and_priors_on_a_full_day():
    series, events = make_synthetic(n_series=1, n_days=1)
    prepared = prepare_windows(series, events)
    # Spikes at the labeled minutes so peak picking has something to return.
    preds = np.zeros_like(prepared.Y)
    preds[:, 7 * 60] = 1.0
    preds[:, 23 * 60] = -1.0
    submission = submission_with_priors(
        scores_to_long(prepared, preds),
        events,
        prepared.valid_ratio,
    )
    assert list(submission.columns) == ["row_id", "series_id", "step", "event", "score"]
    assert set(submission["event"]) == {"onset", "wakeup"}
    perfect = events[["series_id", "step", "event"]].copy()
    perfect["score"] = 1.0
    ap, _, _ = score_events(events, perfect, verbose=False, progress_bar=False)
    assert ap > 0.99
