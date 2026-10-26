import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sleep_unet import cli, train
from sleep_unet.data import (
    CONTEXT_STEPS,
    DAY_STEPS,
    _one_series,
    load_prepared,
    make_synthetic,
    prepare_windows,
    save_prepared,
)
from sleep_unet.metrics import score, score_events, tolerances
from sleep_unet.model import WINDOW_STEPS, UNet1d
from sleep_unet.postprocess import _time_priors, scores_to_long, submission_with_priors


@pytest.fixture
def prepared():
    series, events = make_synthetic(n_series=2, n_days=1)
    return prepare_windows(series, events)


def test_training_refuses_to_overwrite_any_existing_output(tmp_path):
    checkpoint = tmp_path / "model_0_0.pth"
    checkpoint.write_bytes(b"previous experiment")
    with pytest.raises(FileExistsError, match="must be empty"):
        train.train_model(None, tmp_path)
    assert checkpoint.read_bytes() == b"previous experiment"
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_inference_requires_exact_declared_folds(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"seed": 1, "n_splits": 2}))
    expected = [tmp_path / f"model_0_{fold}.pth" for fold in range(2)]
    for path in [*expected, tmp_path / "model_99_0.pth", tmp_path / "best_model.pth"]:
        path.touch()
    assert train.fold_checkpoints(tmp_path) == expected
    expected[1].unlink()
    with pytest.raises(FileNotFoundError, match="model_0_1"):
        train.fold_checkpoints(tmp_path)


def test_oof_requires_cache_and_matching_prepared_data(tmp_path, prepared):
    with pytest.raises(FileNotFoundError, match="oof.npy"):
        train.load_oof(prepared, tmp_path)
    np.save(tmp_path / "oof.npy", prepared.Y)
    (tmp_path / "meta.json").write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        train.load_oof(prepared, tmp_path)
    (tmp_path / "meta.json").write_text(
        json.dumps({"prepared_hash": train.prepared_fingerprint(prepared)})
    )
    save_prepared(prepared, tmp_path / "prepared")
    np.testing.assert_array_equal(
        train.load_oof(load_prepared(tmp_path / "prepared"), tmp_path), prepared.Y
    )
    for field in ["X", "Y", "mask", "norm_min", "features", "index", "df_1min"]:
        changed = copy.deepcopy(prepared)
        if field == "features":
            changed.features.reverse()
        elif field == "index":
            changed.index = changed.index[::-1]
        elif field == "df_1min":
            changed.df_1min.loc[0, "step"] += 12
        else:
            getattr(changed, field).flat[0] += 1
        with pytest.raises(ValueError, match="does not match"):
            train.load_oof(changed, tmp_path)
    for bad in [prepared.Y[:1], np.full_like(prepared.Y, np.nan)]:
        np.save(tmp_path / "oof.npy", bad)
        with pytest.raises(ValueError, match="finite.*shape"):
            train.load_oof(prepared, tmp_path)


def test_evaluation_labels_do_not_influence_peak_selection(tmp_path, monkeypatch, prepared):
    _, events = make_synthetic(n_series=2, n_days=1)
    np.save(tmp_path / "oof.npy", prepared.Y)
    (tmp_path / "meta.json").write_text(
        json.dumps({"prepared_hash": train.prepared_fingerprint(prepared)})
    )
    monkeypatch.setattr(cli, "load_prepared", lambda path: prepared)
    monkeypatch.setattr(cli, "read_table", lambda path: events)

    def unexpected(*args, **kwargs):
        pytest.fail("OOF scoring must not call inference or learn priors")

    monkeypatch.setattr(cli, "predict_windows", unexpected)
    monkeypatch.setattr(cli, "submission_with_priors", unexpected)
    args = argparse.Namespace(
        prepared="prepared", checkpoints=tmp_path, events="events", out=tmp_path / "oof.csv"
    )
    assert cli._evaluate(args) == 0
    original = args.out.read_bytes()
    events["timestamp"] += pd.Timedelta(hours=3)
    events["step"] += 3 * 60 * 12
    assert cli._evaluate(args) == 0
    assert args.out.read_bytes() == original
    (tmp_path / "oof.npy").unlink()
    with pytest.raises(FileNotFoundError):
        cli._evaluate(args)


def test_nni_trials_write_to_separate_directories(tmp_path, monkeypatch, prepared):
    _, events = make_synthetic(n_series=2, n_days=1)
    reports, outputs = [], []
    monkeypatch.setitem(
        sys.modules,
        "nni",
        SimpleNamespace(
            get_next_parameter=lambda: {"batch_size": 1, "initial_channels_num": 12, "lr": 0.001},
            report_final_result=reports.append,
        ),
    )
    monkeypatch.setattr(cli, "load_prepared", lambda path: prepared)
    monkeypatch.setattr(cli, "read_table", lambda path: events)

    def fit(data, output, **kwargs):
        assert kwargs["initial_channels"] == 12
        assert kwargs["batch_size"] == 1
        outputs.append(output)
        output.mkdir()
        (output / "model_0_0.pth").touch()
        return SimpleNamespace(oof=prepared.Y, best_rmse=0.0)

    def unexpected(*args, **kwargs):
        pytest.fail("NNI OOF scoring must not learn priors from validation labels")

    monkeypatch.setattr(cli, "train_model", fit)
    monkeypatch.setattr(cli, "submission_with_priors", unexpected)
    args = cli.build_parser().parse_args(
        ["train", "--nni", "--events", "events", "--out", "artifacts"]
    )
    for trial in ["trial-a", "trial-b"]:
        output = tmp_path / trial
        output.mkdir()
        (output / ".nni").mkdir()
        monkeypatch.setenv("NNI_OUTPUT_DIR", str(output))
        assert cli._train(args) == 0
    assert outputs == [tmp_path / trial / "artifacts" for trial in ["trial-a", "trial-b"]]
    assert len(reports) == 2
    assert np.isfinite(reports).all()


def test_masked_loss_and_rmse_ignore_invalid_labels():
    x = torch.ones(1, 1)
    y = torch.tensor([[1.0, float("nan")]])
    mask = torch.tensor([[1.0, 0.0]])
    model = torch.nn.Linear(1, 2, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (x, y, mask)
    train.train_one_epoch(model, [batch], optimizer, torch.nn.MSELoss(), "cpu")
    torch.testing.assert_close(model.weight, torch.tensor([[0.2], [0.0]]))
    assert train.evaluate_loss(model, [batch], torch.nn.MSELoss(), "cpu") == pytest.approx(0.64)
    assert train.rmse(y.numpy(), np.array([[0.2, 999.0]]), mask.numpy()) == pytest.approx(0.8)


def test_validation_loss_weights_valid_minutes():
    batches = [
        (torch.zeros(1, 2), torch.tensor([[1.0, 99.0]]), torch.tensor([[1.0, 0.0]])),
        (torch.zeros(1, 2), torch.tensor([[3.0, 3.0]]), torch.ones(1, 2)),
    ]
    assert train.evaluate_loss(
        torch.nn.Identity(), batches, torch.nn.MSELoss(), "cpu"
    ) == pytest.approx(19 / 3)


def test_empty_masks_do_not_update_training_state():
    model = torch.nn.Linear(1, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1)
    batch = (torch.ones(1, 1), torch.ones(1, 2), torch.zeros(1, 2))
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="no valid minutes"):
        train.train_one_epoch(model, [batch], optimizer, torch.nn.MSELoss(), "cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    with pytest.raises(RuntimeError, match="no valid minutes"):
        train.evaluate_loss(model, [batch], torch.nn.MSELoss(), "cpu")
    with pytest.raises(ValueError, match="valid minutes"):
        train.rmse(batch[1].numpy(), batch[1].numpy(), batch[2].numpy())


def test_context_uses_adjacent_calendar_days_only():
    series, _ = make_synthetic(n_series=1, n_days=2)
    empty = pd.DataFrame()
    adjacent = _one_series(series, empty, ["log_enmo"])[0]
    np.testing.assert_array_equal(
        adjacent[1, 0, :CONTEXT_STEPS], adjacent[0, 0, DAY_STEPS : DAY_STEPS + CONTEXT_STEPS]
    )
    assert (adjacent[1, 0, :CONTEXT_STEPS] != 0).any()
    series.loc[series.index >= DAY_STEPS, "timestamp"] += pd.Timedelta(days=1)
    separated = _one_series(series, empty, ["log_enmo"])[0]
    assert not separated[:, :, :CONTEXT_STEPS].any()
    assert not separated[:, :, -CONTEXT_STEPS:].any()


def test_rolling_features_do_not_cross_sensor_gaps():
    times = pd.date_range("2018-01-01 23:59", periods=12, freq="5s").append(
        pd.date_range("2018-01-03", periods=12, freq="5s")
    )
    series = pd.DataFrame(
        {
            "series_id": "a",
            "step": np.arange(24),
            "timestamp": times,
            "anglez": np.repeat([100.0, 0.0], 12),
            "enmo": 0.01,
        }
    )
    windows = _one_series(series, pd.DataFrame(), ["log_anglez_std"])[0]
    assert not windows.any()


def test_unseen_minute_prior_bins_preserve_a_strong_peak(prepared):
    _, events = make_synthetic(n_series=2, n_days=1)
    preds = np.zeros_like(prepared.Y)
    preds[:, 7 * 60 + 7] = 1.0
    preds[:, 23 * 60 + 8] = -1.0
    long = scores_to_long(prepared, preds)
    # Onset and wakeup also have distinct observed bins.
    events.loc[events["event"] == "onset", "timestamp"] += pd.Timedelta(minutes=1)
    _, minute = _time_priors(long, events)
    assert minute["minute_mod15"].tolist() == list(range(15))
    assert np.isfinite(minute[["onset", "wakeup"]]).all().all()
    result = submission_with_priors(long, events, prepared.valid_ratio)
    wake = result.loc[(result["event"] == "wakeup") & (result["step"] == (7 * 60 + 7) * 12 + 5)]
    assert len(wake) == 2
    assert (wake["score"] > 0.5).all()


def test_custom_metric_columns_do_not_change_subsequent_scores():
    _, events = make_synthetic(n_series=1, n_days=1)
    submission = events[["series_id", "step", "event"]].assign(score=1.0, row_id=[0, 1])
    columns = {"series_id": "subject", "step": "at", "event": "kind", "score": "confidence"}
    ap, table, matched = score(
        events.rename(columns=columns),
        submission.rename(columns=columns),
        tolerances,
        *columns.values(),
        verbose=False,
        progress_bar=False,
    )
    assert ap == pytest.approx(1.0)
    assert table.columns.name == "kind"
    assert set(columns.values()).issubset(matched.columns)
    assert "row_id" in matched
    assert score_events(events, submission)[0] == pytest.approx(ap)
    with pytest.raises(NotImplementedError):
        score(events, submission, tolerances, *columns, use_scoring_intervals=True)
    assert score_events(events, submission)[0] == pytest.approx(ap)


def test_unet_nonuniform_widths_forward_and_backward():
    model = UNet1d(
        input_channels=3,
        initial_channels=12,
        initial_kernel_size=15,
        down_channels=(12, 24, 48),
        down_kernel_size=(12, 15, 15),
        down_stride=(12, 9, 5),
        res_depth=1,
        res_kernel_size=15,
        se_ratio=4,
        out_kernel_size=21,
    )
    output = model(torch.randn(1, 3, WINDOW_STEPS))
    assert output.shape == (1, 1440)
    output.square().mean().backward()
    assert all(
        param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters()
    )


def test_one_training_window_per_fold_is_supported(tmp_path, prepared):
    result = train.train_model(
        prepared, tmp_path, epochs=1, n_splits=2, batch_size=1, filters=4, res_depth=1
    )
    assert np.isfinite(result.oof).all()
    np.testing.assert_array_equal(train.load_oof(prepared, tmp_path), result.oof)
    assert len(train.fold_checkpoints(tmp_path)) == 2


def test_main_notebook_runs_the_shared_pipeline(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/unet/unet.ipynb").read_text())
    series, events = make_synthetic(n_series=2, n_days=1)
    series.to_parquet(tmp_path / "train_series.parquet", index=False)
    series.to_parquet(tmp_path / "test_series.parquet", index=False)
    events.to_csv(tmp_path / "train_events.csv", index=False)
    monkeypatch.chdir(root)
    scope = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), "unet.ipynb", "exec"), scope)
        if "ROOT" in scope and "SERIES_PATH" not in scope:
            scope["ROOT"] = tmp_path
        if "parameters" in cell["metadata"].get("tags", []):
            scope.update(n_splits=2, n_epochs=1, batch_size=1, filters=4)
    checkpoint_dir = scope["CHECKPOINT_DIR"]
    assert scope["oof"].shape == (2, 1440)
    assert np.isfinite(scope["ap"])
    assert (checkpoint_dir / "oof_submission.csv").is_file()
    assert (checkpoint_dir / "submission.csv").is_file()
    assert train.fold_checkpoints(checkpoint_dir) == [
        checkpoint_dir / "model_0_0.pth",
        checkpoint_dir / "model_0_1.pth",
    ]
