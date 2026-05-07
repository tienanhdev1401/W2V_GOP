"""Fit an isotonic-regression calibration mapping `pred_overall_0_5` -> `gt_total/2`
using the CSV produced by eval_collect.py. Reports baseline + post-calibration
metrics with k-fold cross validation, then refits on the FULL data and saves
knot points to model/iso_calibration.json (no pickle, easy to audit).

Usage:
    python -m scripts.fit_isotonic --csv scripts/out/eval_test.csv \
        --out model/iso_calibration.json --kfold 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.model_selection import KFold  # type: ignore


def label_0_5(score: float, weak_lt: float, good_gte: float) -> str:
    s = float(np.clip(score, 0.0, 5.0))
    if s < weak_lt:
        return "weak"
    if s >= good_gte:
        return "good"
    return "medium"


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    err = pred - gt
    out: Dict[str, float] = {
        "n": int(pred.size),
        "mean_pred": float(pred.mean()),
        "mean_gt": float(gt.mean()),
        "bias": float(err.mean()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pearson_r": float(pearsonr(pred, gt)[0]) if pred.size > 2 else float("nan"),
        "spearman_rho": float(spearmanr(pred, gt).correlation) if pred.size > 2 else float("nan"),
    }
    return out


def label_agreement(pred: np.ndarray, gt: np.ndarray, weak_lt: float, good_gte: float) -> Dict[str, float]:
    plab = np.array([label_0_5(p, weak_lt, good_gte) for p in pred])
    glab = np.array([label_0_5(g, weak_lt, good_gte) for g in gt])
    agree = float((plab == glab).mean())
    return {"label_agreement": agree, "weak_lt": weak_lt, "good_gte": good_gte}


def fit_iso(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=5.0, out_of_bounds="clip", increasing=True)
    iso.fit(x, y)
    return iso


def iso_to_knots(iso: IsotonicRegression) -> Dict[str, List[float]]:
    # IsotonicRegression exposes X_thresholds_ / y_thresholds_ (knots after pooling).
    xs = np.asarray(iso.X_thresholds_, dtype=np.float64).tolist()
    ys = np.asarray(iso.y_thresholds_, dtype=np.float64).tolist()
    return {"x": xs, "y": ys}


def cross_val(df: pd.DataFrame, kfold: int) -> Dict[str, Dict[str, float]]:
    x = df["pred_overall_0_5"].to_numpy(dtype=np.float64)
    y = df["gt_total_0_10"].to_numpy(dtype=np.float64) / 2.0

    base = metrics(x, y)
    base.update(label_agreement(x, y, weak_lt=2.8, good_gte=4.0))

    if kfold < 2 or len(df) < kfold * 2:
        return {"baseline": base, "calibrated_cv": {}}

    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    preds = np.zeros_like(x)
    for tr_idx, te_idx in kf.split(x):
        iso = fit_iso(x[tr_idx], y[tr_idx])
        preds[te_idx] = iso.predict(x[te_idx])

    cal = metrics(preds, y)
    cal.update(label_agreement(preds, y, weak_lt=2.8, good_gte=4.0))
    return {"baseline": base, "calibrated_cv": cal, "kfold": kfold}


def quantile_bands(gt_0_5: np.ndarray) -> Dict[str, float]:
    weak = float(np.quantile(gt_0_5, 0.30))
    good = float(np.quantile(gt_0_5, 0.75))
    return {"band_weak_lt_0_5": weak, "band_good_gte_0_5": good}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="scripts/out/eval_test.csv")
    parser.add_argument("--out", default="model/iso_calibration.json")
    parser.add_argument("--kfold", type=int, default=5)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    df = pd.read_csv(csv_path)
    df = df[(df["error"].isna()) | (df["error"] == "")]
    df = df.dropna(subset=["pred_overall_0_5", "gt_total_0_10"]).copy()
    df["gt_total_0_5"] = df["gt_total_0_10"].astype(float) / 2.0
    df["pred_overall_0_5"] = df["pred_overall_0_5"].astype(float)
    print(f"[data] usable rows = {len(df)}", flush=True)

    rep = cross_val(df, kfold=int(args.kfold))
    print("\n=== Baseline (no isotonic) ===")
    for k, v in rep["baseline"].items():
        print(f"  {k:>20s} = {v}")
    if rep.get("calibrated_cv"):
        print(f"\n=== Cross-validated isotonic (k={rep['kfold']}) ===")
        for k, v in rep["calibrated_cv"].items():
            print(f"  {k:>20s} = {v}")

    # Quantile-based recommended thresholds derived from GT distribution.
    gt = df["gt_total_0_5"].to_numpy(dtype=np.float64)
    bands = quantile_bands(gt)
    print("\n=== Recommended bands (from GT quantiles) ===")
    for k, v in bands.items():
        print(f"  {k:>22s} = {v:.4f}")

    # Final fit on ALL data.
    x_all = df["pred_overall_0_5"].to_numpy(dtype=np.float64)
    y_all = df["gt_total_0_5"].to_numpy(dtype=np.float64)
    iso_full = fit_iso(x_all, y_all)
    knots = iso_to_knots(iso_full)

    payload = {
        "version": "iso_v1",
        "trained_on": str(csv_path.name),
        "n_samples": int(len(df)),
        "input": "overall_score_0_5_pre_iso",
        "output": "overall_score_0_5_calibrated",
        "knots": knots,
        "recommended_bands": bands,
        "metrics": rep,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n[saved] isotonic calibration to {out_path}")


if __name__ == "__main__":
    main()
