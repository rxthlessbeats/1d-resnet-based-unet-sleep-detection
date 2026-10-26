"""Console entry points for the notebook training path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from sleep_unet.data import (
    load_prepared,
    make_synthetic,
    prepare_windows,
    read_table,
    save_prepared,
)
from sleep_unet.metrics import score_events
from sleep_unet.postprocess import scores_to_long, submission_with_priors, submission_without_priors
from sleep_unet.train import load_meta, load_oof, predict_windows, rmse, train_model


def _prepare(args: argparse.Namespace) -> int:
    prepared = prepare_windows(read_table(args.series), read_table(args.events))
    save_prepared(prepared, args.out)
    print(
        f"wrote {args.out} windows={prepared.X.shape[0]} "
        f"channels={prepared.X.shape[1]} length={prepared.X.shape[2]}"
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    prepared = load_prepared(args.prepared)
    out_dir = Path(args.out)
    kwargs = {
        "epochs": args.epochs,
        "n_splits": args.folds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "filters": args.filters,
        "patience": args.patience,
    }
    if args.nni:
        if not args.events:
            raise SystemExit("train --nni needs --events so the trial can report event AP")
        import nni

        out_dir = Path(os.environ["NNI_OUTPUT_DIR"]) / out_dir.name
        params = nni.get_next_parameter()
        kwargs["batch_size"] = int(params["batch_size"])
        kwargs["initial_channels"] = int(params["initial_channels_num"])
        kwargs["lr"] = float(params["lr"])
    result = train_model(prepared, out_dir, **kwargs)
    print(f"best_fold_rmse {result.best_rmse:.5f}")
    if args.nni:
        events = read_table(args.events)
        submission = submission_without_priors(scores_to_long(prepared, result.oof))
        ap, _, _ = score_events(events, submission, verbose=False, progress_bar=False)
        nni.report_final_result(ap)
        print(f"event_ap {ap:.5f}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    prepared = load_prepared(args.prepared)
    preds = load_oof(prepared, args.checkpoints)
    print(f"oof_rmse {rmse(prepared.Y, preds, prepared.mask):.5f}")
    events = read_table(args.events)
    submission = submission_without_priors(scores_to_long(prepared, preds))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    ap, table, _ = score_events(events, submission, verbose=False, progress_bar=False)
    print(f"event_ap {ap:.5f}")
    print(table.to_string())
    print(f"wrote {args.out}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    meta = load_meta(args.checkpoints)
    if "norm_min" not in meta or "norm_max" not in meta:
        raise ValueError(
            "checkpoint metadata lacks training normalization; restore norm_min/norm_max "
            "from the matching prepared dataset or retrain"
        )
    if not args.peaks_only and not args.events:
        raise SystemExit("predict needs --events for time priors, or --peaks-only")
    prepared = prepare_windows(
        read_table(args.series),
        events=None,
        features=meta["features"],
        drop_series=(),
        normalization=(meta["norm_min"], meta["norm_max"]),
    )
    preds = predict_windows(prepared.X, prepared.mask, args.checkpoints, args.batch_size)
    long = scores_to_long(prepared, preds)
    submission = (
        submission_without_priors(long)
        if args.peaks_only
        else submission_with_priors(long, read_table(args.events), prepared.valid_ratio)
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(submission)}")
    return 0


def run_demo() -> dict:
    """Synthetic end-to-end pass. Does not read the competition files."""
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    series, events = make_synthetic(n_series=2, n_days=2)
    prepared = prepare_windows(series, events)
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="sleep-unet-demo-") as tmp:
        out = Path(tmp)
        result = train_model(
            prepared,
            out,
            epochs=1,
            n_splits=2,
            batch_size=2,
            lr=1e-3,
            seed=1,
            filters=4,
            patience=20,
        )
        submission = submission_without_priors(scores_to_long(prepared, result.oof))
        ap, _, _ = score_events(events, submission, verbose=False, progress_bar=False)
        preds = predict_windows(prepared.X, prepared.mask, out, batch_size=2)
        oof_rmse = rmse(prepared.Y, result.oof, prepared.mask)

    if preds.shape != prepared.Y.shape:
        raise SystemExit(f"predict shape {preds.shape} != targets {prepared.Y.shape}")
    if len(submission) == 0:
        raise SystemExit("demo submission was empty")
    if not np.isfinite(oof_rmse) or not np.isfinite(ap):
        raise SystemExit(f"demo metrics were not finite: rmse={oof_rmse} ap={ap}")
    summary = {
        "windows": int(prepared.X.shape[0]),
        "channels": int(prepared.X.shape[1]),
        "length": int(prepared.X.shape[2]),
        "oof_rmse": oof_rmse,
        "best_fold_rmse": result.best_rmse,
        "event_ap": float(ap),
        "submission_rows": int(len(submission)),
        "predict_shape": list(preds.shape),
    }
    print(
        "demo ok "
        f"windows={summary['windows']} channels={summary['channels']} "
        f"length={summary['length']} oof_rmse={summary['oof_rmse']:.5f} "
        f"event_ap={summary['event_ap']:.5f} submission_rows={summary['submission_rows']} "
        f"predict_shape={tuple(summary['predict_shape'])}"
    )
    return summary


def _demo(_args: argparse.Namespace) -> int:
    run_demo()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sleep-unet")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build day windows from series and events")
    prepare.add_argument("--series", required=True, help="train_series.parquet or csv")
    prepare.add_argument("--events", required=True, help="train_events.csv")
    prepare.add_argument("--out", default="data/prepared")
    prepare.set_defaults(func=_prepare)

    train = sub.add_parser("train", help="GroupKFold training from the U-Net notebook")
    train.add_argument("--prepared", default="data/prepared")
    train.add_argument("--out", default="artifacts")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=1, help="number of random seeds to average")
    train.add_argument("--filters", type=int, default=24)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--events", default=None, help="required with --nni")
    train.add_argument(
        "--nni",
        action="store_true",
        help="read batch_size, initial_channels_num, and lr from an NNI trial",
    )
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("evaluate", help="OOF RMSE and event detection AP")
    evaluate.add_argument("--prepared", default="data/prepared")
    evaluate.add_argument("--checkpoints", default="artifacts")
    evaluate.add_argument("--events", required=True)
    evaluate.add_argument("--out", default="artifacts/oof_submission.csv")
    evaluate.set_defaults(func=_evaluate)

    predict = sub.add_parser("predict", help="average fold checkpoints and write a submission")
    predict.add_argument("--series", required=True)
    predict.add_argument(
        "--events", help="train events used for the time-of-day prior; omit with --peaks-only"
    )
    predict.add_argument("--checkpoints", default="artifacts")
    predict.add_argument("--out", default="artifacts/submission.csv")
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--peaks-only", action="store_true")
    predict.set_defaults(func=_predict)

    demo = sub.add_parser("demo", help="synthetic smoke test, no external dataset")
    demo.set_defaults(func=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
