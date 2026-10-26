import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sleep_unet import cli, train
from sleep_unet.data import CONTEXT_STEPS, FEATURES, make_synthetic, normalize_windows


def test_scheduler_preserves_decay_and_does_not_change_when_read():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    scheduler = train.CustomCosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-5)
    expected = {0: 0.001, 5: 0.000505, 10: 0.0009, 15: 0.000455, 20: 0.00081, 30: 0.000729}
    for epoch in range(31):
        rate = optimizer.param_groups[0]["lr"]
        assert scheduler.get_last_lr() == [rate]
        assert scheduler.get_lr() == scheduler.get_lr()
        assert optimizer.param_groups[0]["lr"] == rate
        if epoch in expected:
            assert rate == pytest.approx(expected[epoch])
        optimizer.step()
        scheduler.step()


def test_seed_average_and_single_model_inference(tmp_path, monkeypatch):
    built, grad_modes = [], []

    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(1 - torch.rand(()))

        def forward(self, x):
            if not self.training:
                grad_modes.append(torch.is_grad_enabled())
            return self.weight.expand(len(x), 2)

    def build(**kwargs):
        model = ConstantModel()
        built.append(model)
        return model

    monkeypatch.setattr(train, "build_unet", build)
    monkeypatch.setattr(train, "get_device", lambda: torch.device("cpu"))
    prepared = SimpleNamespace(
        X=np.zeros((4, 1, 4), dtype=np.float32),
        Y=np.zeros((4, 2), dtype=np.float32),
        mask=np.ones((4, 2), dtype=np.float32),
        index=pd.MultiIndex.from_product(
            [["a", "b"], ["day1", "day2"]], names=["series_id", "date"]
        ),
        df_1min=pd.DataFrame({"step": [0, 1, 0, 1]}),
        features=["feature"],
        norm_min=np.zeros((1, 1, 1), dtype=np.float32),
        norm_max=np.ones((1, 1, 1), dtype=np.float32),
    )
    result = train.train_model(prepared, tmp_path, epochs=1, n_splits=2, batch_size=2, seed=2)
    weights = [
        float(torch.load(path, weights_only=True)["weight"])
        for path in sorted(tmp_path.glob("model_*_*.pth"))
    ]
    np.testing.assert_allclose(result.oof, np.mean(weights))
    assert result.best_rmse == pytest.approx(min(weights))
    count = len(built)
    predictions = train.predict_windows(prepared.X, prepared.mask, tmp_path, batch_size=2)
    assert len(built) == count + 1
    np.testing.assert_allclose(predictions, result.oof)
    assert grad_modes and not any(grad_modes)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["norm_min"] == prepared.norm_min.tolist()
    assert meta["norm_max"] == prepared.norm_max.tolist()
    np.testing.assert_array_equal(train.load_oof(prepared, tmp_path), result.oof)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"seed": 0}, {"seed": -1}, {"batch_size": 0}])
def test_invalid_counts_do_not_create_artifacts(tmp_path, kwargs):
    out = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="positive"):
        train.train_model(None, out, **kwargs)
    assert not out.exists()


def test_normalization_uses_training_bounds():
    _, low, high = normalize_windows(np.array([[[0.0, 2.0]]], dtype=np.float32))
    scaled, actual_low, actual_high = normalize_windows(
        np.array([[[0.0, 4.0]]], dtype=np.float32), normalization=(low, high)
    )
    np.testing.assert_array_equal(scaled, [[[0.0, 2.0]]])
    np.testing.assert_array_equal(actual_low, low)
    np.testing.assert_array_equal(actual_high, high)


@pytest.mark.parametrize(
    "bounds",
    [
        (np.zeros((1, 2, 1)), np.ones((1, 2, 1))),
        (np.full((1, 1, 1), np.nan), np.ones((1, 1, 1))),
        (np.ones((1, 1, 1)), np.zeros((1, 1, 1))),
    ],
)
def test_normalization_rejects_bad_bounds(bounds):
    with pytest.raises(ValueError, match="normalization"):
        normalize_windows(np.zeros((1, 1, 2)), normalization=bounds)


def test_cli_prediction_uses_checkpoint_normalization(tmp_path, monkeypatch):
    series, events = make_synthetic(n_series=1, n_days=1)
    meta = {
        "features": FEATURES,
        "norm_min": np.zeros((1, 3, 1)).tolist(),
        "norm_max": np.full((1, 3, 1), 2.0).tolist(),
    }
    monkeypatch.setattr(cli, "load_meta", lambda path: meta)

    def read(path):
        assert path == "series"
        return series

    monkeypatch.setattr(cli, "read_table", read)

    def predict(x, mask, *args):
        assert x[0, 1, CONTEXT_STEPS] == pytest.approx(np.log(0.011) / 2, rel=1e-6)
        return np.zeros_like(mask)

    monkeypatch.setattr(cli, "predict_windows", predict)
    out = tmp_path / "submission.csv"
    args = argparse.Namespace(
        series="series",
        events=None,
        checkpoints=tmp_path,
        batch_size=2,
        peaks_only=True,
        out=out,
    )
    assert cli._predict(args) == 0
    assert out.is_file()


def test_cli_requires_events_when_prediction_uses_priors(monkeypatch):
    monkeypatch.setattr(cli, "load_meta", lambda path: {"norm_min": [], "norm_max": []})
    with pytest.raises(SystemExit, match="--events.*--peaks-only"):
        cli._predict(argparse.Namespace(checkpoints="checkpoint", peaks_only=False, events=None))


def test_cli_rejects_checkpoints_without_training_normalization(monkeypatch):
    monkeypatch.setattr(cli, "load_meta", lambda path: {"features": FEATURES})
    with pytest.raises(ValueError, match="normalization"):
        cli._predict(argparse.Namespace(checkpoints="old-checkpoint"))
