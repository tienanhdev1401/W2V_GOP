"""Run the current scoring service on speechocean762 and dump a CSV of
predicted vs ground-truth scores. Used as input for fit_isotonic.py.

Usage (from project root, with .venv activated):
    python -m scripts.eval_collect --split test --limit 0 --out scripts/out/eval_test.csv

Flags:
    --split   {train,test}   default: test
    --limit   N              0 = all samples (default 0)
    --out     path           output CSV path
    --hf-id   string         HF dataset id (default: mispeech/speechocean762)

The script is resumable: rows already present in --out (matched by utt_id)
are skipped on subsequent runs.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np
import soundfile as sf  # bundled with librosa
from datasets import Audio, load_dataset  # type: ignore

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.scoring_service import W2VGOPScoringService  # noqa: E402


CSV_FIELDS = [
    "utt_id",
    "text",
    "duration_sec",
    # ground-truth (0-10 scale from speechocean762)
    "gt_accuracy_0_10",
    "gt_completeness_0_10",
    "gt_fluency_0_10",
    "gt_prosodic_0_10",
    "gt_total_0_10",
    # predicted calibrated 0-5
    "pred_overall_0_5",
    "pred_utt_total_0_5",
    "pred_utt_accuracy_0_5",
    "pred_utt_completeness_0_5",
    "pred_utt_fluency_0_5",
    "pred_utt_prosodic_0_5",
    "pred_word_total_mean_0_5",
    # raw model outputs (pre-sigmoid calibration), 0-2 scale
    "raw_utt_total",
    "raw_utt_accuracy",
    "raw_utt_completeness",
    "raw_utt_fluency",
    "raw_utt_prosodic",
    "raw_word_total_mean",
    "error",
]


def encode_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    arr = np.asarray(audio_array, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, int(sample_rate), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def already_processed(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()
    out: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = row.get("utt_id")
            if uid:
                out.add(uid)
    return out


def get_field(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", default="mispeech/speechocean762")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--out", default="scripts/out/eval_test.csv")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load-dataset] {args.hf_id}:{args.split}", flush=True)
    ds = load_dataset(args.hf_id, split=args.split)
    # Disable internal decoding (avoids torchcodec dependency on Windows).
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
    print(f"[load-dataset] num_rows={len(ds)}", flush=True)

    print("[load-service] initialising W2VGOPScoringService ...", flush=True)
    service = W2VGOPScoringService()
    service.load()
    print("[load-service] ready", flush=True)

    seen = already_processed(out_path)
    if seen:
        print(f"[resume] found {len(seen)} rows already in {out_path}", flush=True)

    write_header = not out_path.exists()
    fh = out_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        fh.flush()

    n_total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    n_ok = 0
    n_err = 0

    for idx in range(n_total):
        sample = ds[idx]
        uid = str(get_field(sample, "utt_id", "id", "audio_id", default=f"{args.split}_{idx}"))
        if uid in seen:
            continue

        text = str(get_field(sample, "text", "transcript", default="")).strip()
        audio = sample.get("audio") or {}
        # With decode=False the dict typically has either 'bytes' or 'path'.
        arr: np.ndarray
        sr = 16000
        try:
            if audio.get("bytes"):
                arr_loaded, sr_loaded = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                arr = np.asarray(arr_loaded, dtype=np.float32)
                sr = int(sr_loaded)
            elif audio.get("path"):
                arr_loaded, sr_loaded = sf.read(audio["path"], dtype="float32")
                arr = np.asarray(arr_loaded, dtype=np.float32)
                sr = int(sr_loaded)
            elif "array" in audio:  # fallback (decode=True path)
                arr = np.asarray(audio.get("array", []), dtype=np.float32)
                sr = int(audio.get("sampling_rate", 16000))
            else:
                arr = np.zeros(0, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            arr = np.zeros(0, dtype=np.float32)

        gt_total = float(get_field(sample, "total", default=float("nan")))
        gt_acc = float(get_field(sample, "accuracy", default=float("nan")))
        gt_flu = float(get_field(sample, "fluency", default=float("nan")))
        gt_pro = float(get_field(sample, "prosodic", default=float("nan")))
        gt_cmp = float(get_field(sample, "completeness", default=float("nan")))

        row: Dict[str, Any] = {f: "" for f in CSV_FIELDS}
        row.update(
            {
                "utt_id": uid,
                "text": text,
                "duration_sec": float(arr.shape[0] / max(sr, 1)) if arr.size else 0.0,
                "gt_accuracy_0_10": gt_acc,
                "gt_completeness_0_10": gt_cmp,
                "gt_fluency_0_10": gt_flu,
                "gt_prosodic_0_10": gt_pro,
                "gt_total_0_10": gt_total,
            }
        )

        try:
            if not text:
                raise ValueError("empty text")
            if arr.size == 0:
                raise ValueError("empty audio")
            wav_bytes = encode_wav_bytes(arr, sr)
            payload = service.score(text=text, audio_bytes=wav_bytes, filename=f"{uid}.wav")

            scores_0_5 = payload.get("scores_0_5", {}) or {}
            raw = payload.get("scores_normalized", {}) or {}

            row["pred_overall_0_5"] = float(payload.get("overall_score_0_5", float("nan")))
            row["pred_utt_total_0_5"] = float(scores_0_5.get("utt_total", float("nan")))
            row["pred_utt_accuracy_0_5"] = float(scores_0_5.get("utt_accuracy", float("nan")))
            row["pred_utt_completeness_0_5"] = float(scores_0_5.get("utt_completeness", float("nan")))
            row["pred_utt_fluency_0_5"] = float(scores_0_5.get("utt_fluency", float("nan")))
            row["pred_utt_prosodic_0_5"] = float(scores_0_5.get("utt_prosodic", float("nan")))
            row["pred_word_total_mean_0_5"] = float(scores_0_5.get("word_total_mean", float("nan")))
            row["raw_utt_total"] = float(raw.get("utt_total", float("nan")))
            row["raw_utt_accuracy"] = float(raw.get("utt_accuracy", float("nan")))
            row["raw_utt_completeness"] = float(raw.get("utt_completeness", float("nan")))
            row["raw_utt_fluency"] = float(raw.get("utt_fluency", float("nan")))
            row["raw_utt_prosodic"] = float(raw.get("utt_prosodic", float("nan")))
            row["raw_word_total_mean"] = float(raw.get("word_total_mean", float("nan")))
            row["error"] = ""
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            n_err += 1

        writer.writerow(row)
        fh.flush()

        if (idx + 1) % 25 == 0 or (idx + 1) == n_total:
            print(
                f"[progress] {idx + 1}/{n_total}  ok={n_ok}  err={n_err}  last_uid={uid}",
                flush=True,
            )

    fh.close()
    print(f"[done] wrote {n_ok} ok, {n_err} err to {out_path}", flush=True)


if __name__ == "__main__":
    main()
