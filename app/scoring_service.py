from __future__ import annotations

# pyright: reportMissingImports=false

import json
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC

SERVICE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = SERVICE_ROOT / "model"

from app import pipeline_local as pipeline
from app.model_arch import Wav2VecGOPT


def scale_score_0_5_from_0_2(value: float) -> float:
    return float(np.clip(value, 0.0, 2.0) * 2.5)


def sigmoid_calibrated_score_0_5(value: float, center: float, scale: float) -> float:
    scale_safe = max(scale, 1e-6)
    z = (value - center) / scale_safe
    return float(np.clip(5.0 / (1.0 + np.exp(-z)), 0.0, 5.0))


def temperature_scale_0_2(value: float, temperature: float, center: float = 1.0) -> float:
    clipped = float(np.clip(value, 0.0, 2.0))
    temp = max(temperature, 1e-3)
    scaled = center + (clipped - center) / temp
    return float(np.clip(scaled, 0.0, 2.0))


def bell_curve_score_0_5(value: float, target: float, sigma: float) -> float:
    sigma_safe = max(float(sigma), 1e-3)
    z = (float(value) - float(target)) / sigma_safe
    return float(np.clip(5.0 * np.exp(-0.5 * z * z), 0.0, 5.0))


def apply_isotonic_0_5(value: float, knots_x: np.ndarray, knots_y: np.ndarray) -> float:
    """Piecewise-linear monotonic mapping using isotonic-regression knots.
    Values outside the trained range are clipped to the boundary outputs.
    """
    if knots_x is None or knots_y is None or knots_x.size == 0:
        return float(np.clip(value, 0.0, 5.0))
    out = float(np.interp(float(value), knots_x, knots_y))
    return float(np.clip(out, 0.0, 5.0))


def score_band_0_5(score: float, weak_lt: float, good_gte: float) -> str:
    s = float(np.clip(score, 0.0, 5.0))
    lo = float(min(weak_lt, good_gte))
    hi = float(max(weak_lt, good_gte))
    if s < lo:
        return "weak"
    if s >= hi:
        return "good"
    return "medium"


def overall_label_0_5(score: float, fair_gte: float, good_gte: float) -> str:
    s = float(np.clip(score, 0.0, 5.0))
    fair = float(min(fair_gte, good_gte))
    good = float(max(fair_gte, good_gte))
    if s >= good:
        return "Good"
    if s >= fair:
        return "Fair"
    return "Needs work"


@dataclass
class ScoreConfig:
    checkpoint: Path = field(
        default_factory=lambda: MODEL_ROOT / "best_w2v_gopt_research_main.pth"
    )
    data_dir: Path = field(default_factory=lambda: MODEL_ROOT / "seq_data_w2v_research_pitch")
    confusion_json: Path = field(default_factory=lambda: MODEL_ROOT / "adaptive_confusion.json")
    phone_map: Path = field(default_factory=lambda: MODEL_ROOT / "seq_data_w2v_research_pitch" / "phone_to_idx.json")
    ctc_model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    sample_rate: int = 16000
    max_audio_sec: float = 8.0
    proj_dim: int = 96
    confusion_topk: int = 3
    lambda_entropy: float = 0.05
    prosody_fmin: float = 50.0
    prosody_fmax: float = 350.0
    num_heads: int = 4
    device: str = "auto"
    allow_text_tokenizer_fallback: bool = False
    trim_silence: bool = True
    trim_top_db: float = 30.0
    trim_frame_length: int = 2048
    trim_hop_length: int = 512
    calibration_sigma_floor: float = 0.12
    calibration_center_shift_sigma: float = 1.0
    calibration_temperature: float = 1.3
    calibration_clip_min: float = 0.0
    calibration_clip_max: float = 2.0
    vad_enabled: bool = True
    vad_top_db: float = 30.0
    vad_frame_length: int = 1024
    vad_hop_length: int = 256
    vad_min_segment_sec: float = 0.08
    vad_pad_sec: float = 0.03
    denoise_enabled: bool = True
    denoise_strength: float = 1.0
    denoise_floor_ratio: float = 0.08
    denoise_n_fft: int = 512
    denoise_hop_length: int = 128
    normalize_audio: bool = True
    target_rms: float = 0.05
    max_gain: float = 8.0
    min_duration_sec: float = 1.0
    min_rms_energy: float = 0.003
    min_rms_db: float = -52.0
    feature_clip_value: float = 5.0
    smoothing_utt_weight: float = 0.5
    smoothing_word_weight: float = 0.5
    phone_temperature: float = 0.75
    phone_temperature_center_0_2: float = 1.0
    speech_rate_target_phones_per_sec: float = 3.0
    speech_rate_sigma: float = 0.9
    speech_rate_weight_on_fluency: float = 0.35
    pause_ratio_target: float = 0.10
    pause_ratio_sigma: float = 0.22
    timing_cv_target: float = 0.55
    timing_cv_sigma: float = 0.45
    prosody_pattern_weight_on_prosodic: float = 0.25
    total_adjustment_weight: float = 0.30
    threshold_profile_version: str = "v1.0.0"
    band_weak_lt_0_5: float = 2.8
    band_good_gte_0_5: float = 4.0
    overall_fair_gte_0_5: float = 2.8
    overall_good_gte_0_5: float = 4.0
    isotonic_calibration_path: Path = field(
        default_factory=lambda: MODEL_ROOT / "iso_calibration.json"
    )
    conversation_pronunciation_weight: float = 0.75
    conversation_grammar_weight: float = 0.25
    grammar_profile_version: str = "rule_v1.0.0"


class W2VGOPScoringService:
    def __init__(self, config: Optional[ScoreConfig] = None) -> None:
        self.config = config or ScoreConfig()
        self._lock = threading.Lock()
        self._ready = False

        self.device: Optional[torch.device] = None
        self.model = None
        self.ctc_model = None
        self.processor = None

        self.arch: Dict[str, Any] = {}
        self.variant: str = ""
        self.blank_id: Optional[int] = None
        self.unk_id: Optional[int] = None
        self.confusion_graph: Dict[int, List[int]] = {}
        self.phone_to_idx: Dict[str, int] = {}
        self.unk_phone_idx: int = 38
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.calibration_stats: Dict[str, Dict[str, float]] = {}
        self.iso_knots_x: Optional[np.ndarray] = None
        self.iso_knots_y: Optional[np.ndarray] = None
        self.iso_metadata: Dict[str, Any] = {}
        self.model_version: str = "unknown"
        self.calibration_version: str = "unknown"
        self.preprocessing_profile: Dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        with self._lock:
            if self._ready:
                return

            checkpoint = self.config.checkpoint.expanduser().resolve()
            data_dir = self.config.data_dir.expanduser().resolve()
            confusion_json = self.config.confusion_json.expanduser().resolve()
            phone_map = self.config.phone_map.expanduser().resolve()

            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            if not data_dir.exists():
                raise FileNotFoundError(f"Data dir not found: {data_dir}")
            if not confusion_json.exists():
                raise FileNotFoundError(f"Confusion graph not found: {confusion_json}")
            if not phone_map.exists():
                raise FileNotFoundError(f"Phone map not found: {phone_map}")

            if self.config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.config.device)

            ckpt_stat = checkpoint.stat()
            self.model_version = f"{checkpoint.name}:{int(ckpt_stat.st_mtime)}:{int(ckpt_stat.st_size)}"
            self.calibration_version = (
                "sigmoid_train_stats"
                f":clip={self.config.calibration_clip_min:.2f}-{self.config.calibration_clip_max:.2f}"
                f":temp={self.config.calibration_temperature:.2f}"
                f":shift={self.config.calibration_center_shift_sigma:.2f}"
            )
            self.preprocessing_profile = {
                "sample_rate": int(self.config.sample_rate),
                "max_audio_sec": float(self.config.max_audio_sec),
                "trim_silence": bool(self.config.trim_silence),
                "trim_top_db": float(self.config.trim_top_db),
                "vad_enabled": bool(self.config.vad_enabled),
                "vad_top_db": float(self.config.vad_top_db),
                "denoise_enabled": bool(self.config.denoise_enabled),
                "denoise_strength": float(self.config.denoise_strength),
                "normalize_audio": bool(self.config.normalize_audio),
                "target_rms": float(self.config.target_rms),
            }

            raw_state = torch.load(str(checkpoint), map_location="cpu")
            if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
                raw_state = raw_state["state_dict"]
            if not isinstance(raw_state, dict):
                raise ValueError("Unsupported checkpoint format: expected a state_dict dictionary")

            state = pipeline.strip_module_prefix(raw_state)
            self.arch = pipeline.infer_arch_from_state_dict(state, num_heads=self.config.num_heads)

            if self.arch["model_type"] == "wav2vec_gopt":
                model = Wav2VecGOPT(
                    embed_dim=self.arch["embed_dim"],
                    num_heads=self.arch["num_heads"],
                    depth=self.arch["depth"],
                    input_dim=self.arch["input_dim"],
                    adapter_dim=self.arch["adapter_dim"],
                    adapter_dropout=0.1,
                    max_seq_len=self.arch["max_seq_len"],
                    use_phn_embedding=self.arch["use_phn_embedding"],
                )
            elif self.arch["model_type"] == "notebook_w2vgopt":
                model = pipeline.NotebookW2VGOPT(
                    input_dim=self.arch["input_dim"],
                    embed_dim=self.arch["embed_dim"],
                    depth=self.arch["depth"],
                    heads=self.arch["num_heads"],
                    max_len=self.arch["max_seq_len"],
                    phn_vocab=self.arch["phn_vocab"],
                    adapter_dim=self.arch["adapter_dim"],
                    dropout=0.1,
                )
            else:
                raise ValueError(f"Unsupported model_type: {self.arch['model_type']}")

            incompatible = model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"State dict mismatch. Missing={incompatible.missing_keys}, Unexpected={incompatible.unexpected_keys}"
                )

            model = model.to(self.device)
            model.eval()
            self.model = model

            self.variant = pipeline.infer_variant_from_input_dim(self.arch["input_dim"], proj_dim=self.config.proj_dim)
            self.confusion_graph = pipeline.load_confusion_graph(confusion_json)
            self.phone_to_idx, self.unk_phone_idx = pipeline.load_phone_map(phone_map)
            self.norm_mean, self.norm_std = pipeline.load_feature_norm(data_dir)
            self.calibration_stats = self._load_calibration_stats(data_dir)
            if int(self.norm_mean.shape[0]) != int(self.arch["input_dim"]):
                raise ValueError(
                    f"Feature dim mismatch: norm={self.norm_mean.shape[0]} vs model input_dim={self.arch['input_dim']}"
                )

            self.processor = pipeline.load_ctc_processor(self.config.ctc_model_name)
            self.ctc_model = Wav2Vec2ForCTC.from_pretrained(self.config.ctc_model_name).to(self.device)
            self.ctc_model.eval()

            self.blank_id = self.processor.tokenizer.pad_token_id
            self.unk_id = self.processor.tokenizer.unk_token_id

            self._load_isotonic_calibration()
            self._ready = True

    def _load_isotonic_calibration(self) -> None:
        path = Path(self.config.isotonic_calibration_path).expanduser().resolve()
        if not path.exists():
            self.iso_knots_x = None
            self.iso_knots_y = None
            self.iso_metadata = {}
            return
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            knots = data.get("knots", {}) or {}
            xs = np.asarray(knots.get("x", []), dtype=np.float64)
            ys = np.asarray(knots.get("y", []), dtype=np.float64)
            if xs.size < 2 or ys.size != xs.size:
                raise ValueError("isotonic knots invalid")
            order = np.argsort(xs)
            self.iso_knots_x = xs[order]
            self.iso_knots_y = ys[order]
            self.iso_metadata = {
                "version": data.get("version", "iso"),
                "trained_on": data.get("trained_on", ""),
                "n_samples": int(data.get("n_samples", 0)),
                "path": str(path),
            }
            self.calibration_version = (
                f"isotonic:{self.iso_metadata['version']}:n={self.iso_metadata['n_samples']}"
            )
            bands = (data.get("recommended_bands") or {})
            if "band_weak_lt_0_5" in bands and "band_good_gte_0_5" in bands:
                weak = float(bands["band_weak_lt_0_5"])
                good = float(bands["band_good_gte_0_5"])
                self.config.band_weak_lt_0_5 = weak
                self.config.band_good_gte_0_5 = good
                self.config.overall_fair_gte_0_5 = weak
                self.config.overall_good_gte_0_5 = good
        except Exception:
            self.iso_knots_x = None
            self.iso_knots_y = None
            self.iso_metadata = {}

    def _calibrate_overall_0_5(self, value: float) -> float:
        if self.iso_knots_x is None or self.iso_knots_y is None:
            return float(np.clip(value, 0.0, 5.0))
        return apply_isotonic_0_5(value, self.iso_knots_x, self.iso_knots_y)

    def _fit_stats_1d(self, values: np.ndarray) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return 1.0, 0.5
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        sigma = max(float(self.config.calibration_sigma_floor), sigma)
        return mu, sigma

    def _load_calibration_stats(self, data_dir: Path) -> Dict[str, Dict[str, float]]:
        utt_path = data_dir / "tr_label_utt.npy"
        word_path = data_dir / "tr_label_word.npy"
        phn_path = data_dir / "tr_label_phn.npy"

        if (not utt_path.exists()) or (not word_path.exists()) or (not phn_path.exists()):
            return {}

        utt = np.load(utt_path, mmap_mode="r")
        word = np.load(word_path, mmap_mode="r")
        phn = np.load(phn_path, mmap_mode="r")

        valid = phn[:, :, 1] >= 0
        word_valid = word[valid]

        out: Dict[str, Dict[str, float]] = {}
        utt_keys = ["utt_accuracy", "utt_completeness", "utt_fluency", "utt_prosodic", "utt_total"]
        word_keys = ["word_accuracy_mean", "word_stress_mean", "word_total_mean"]

        for idx, key in enumerate(utt_keys):
            mu, sigma = self._fit_stats_1d(utt[:, idx])
            out[key] = {"mu": mu, "sigma": sigma}

        for idx, key in enumerate(word_keys):
            mu, sigma = self._fit_stats_1d(word_valid[:, idx])
            out[key] = {"mu": mu, "sigma": sigma}

        return out

    def _score_0_5(self, key: str, value: float) -> float:
        clip_min = min(float(self.config.calibration_clip_min), float(self.config.calibration_clip_max))
        clip_max = max(float(self.config.calibration_clip_min), float(self.config.calibration_clip_max))
        value_clipped = float(np.clip(value, clip_min, clip_max))

        stat = self.calibration_stats.get(key)
        if stat is None:
            return scale_score_0_5_from_0_2(value_clipped)

        sigma = max(stat["sigma"], self.config.calibration_sigma_floor)
        scale = sigma * self.config.calibration_temperature
        center = stat["mu"] - self.config.calibration_center_shift_sigma * scale
        return sigmoid_calibrated_score_0_5(value=value_clipped, center=center, scale=scale)

    def _audio_rms(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32)) + 1e-12))

    def _audio_rms_db(self, audio: np.ndarray) -> float:
        return float(20.0 * np.log10(max(self._audio_rms(audio), 1e-8)))

    def _spectral_denoise(self, audio: np.ndarray) -> np.ndarray:
        if (not self.config.denoise_enabled) or (audio.size < max(32, int(self.config.denoise_n_fft))):
            return audio

        try:
            n_fft = int(self.config.denoise_n_fft)
            hop = int(self.config.denoise_hop_length)
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft)
            mag = np.abs(stft)
            phase = np.exp(1j * np.angle(stft))

            noise_profile = np.percentile(mag, 20.0, axis=1, keepdims=True)
            cleaned_mag = mag - float(self.config.denoise_strength) * noise_profile
            floor = float(self.config.denoise_floor_ratio) * mag
            cleaned_mag = np.maximum(cleaned_mag, floor)

            recon = librosa.istft(cleaned_mag * phase, hop_length=hop, win_length=n_fft, length=audio.shape[0])
            return np.asarray(recon, dtype=np.float32)
        except Exception:
            return audio

    def _apply_vad(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, float]]:
        original_samples = int(audio.shape[0])
        empty_meta = {
            "segments": 0.0,
            "pause_samples": 0.0,
            "pause_ratio": 0.0,
            "gap_samples": 0.0,
        }

        if not self.config.vad_enabled:
            return audio, empty_meta

        intervals = librosa.effects.split(
            audio,
            top_db=float(self.config.vad_top_db),
            frame_length=int(self.config.vad_frame_length),
            hop_length=int(self.config.vad_hop_length),
        )
        if intervals.size == 0:
            return audio, empty_meta

        min_len = int(float(self.config.vad_min_segment_sec) * sample_rate)
        pad = int(float(self.config.vad_pad_sec) * sample_rate)

        merged: List[List[int]] = []
        for st_raw, en_raw in intervals.tolist():
            st = int(st_raw)
            en = int(en_raw)
            if (en - st) < min_len:
                continue

            st = max(0, st - pad)
            en = min(audio.shape[0], en + pad)
            if not merged or st > merged[-1][1]:
                merged.append([st, en])
            else:
                merged[-1][1] = max(merged[-1][1], en)

        if not merged:
            return audio, empty_meta

        chunks = [audio[st:en] for st, en in merged if en > st]
        if not chunks:
            return audio, empty_meta

        out = np.concatenate(chunks).astype(np.float32)

        speech_samples = int(sum(max(0, en - st) for st, en in merged))
        pause_samples = max(0, original_samples - speech_samples)
        gap_samples = 0
        for idx in range(1, len(merged)):
            gap_samples += max(0, int(merged[idx][0]) - int(merged[idx - 1][1]))

        vad_meta = {
            "segments": float(len(chunks)),
            "pause_samples": float(pause_samples),
            "pause_ratio": float(pause_samples / max(original_samples, 1)),
            "gap_samples": float(gap_samples),
        }
        return out, vad_meta

    def _normalize_waveform(self, audio: np.ndarray) -> np.ndarray:
        out = np.asarray(audio, dtype=np.float32)
        if out.size == 0:
            return out

        out = out - float(np.mean(out))
        if self.config.normalize_audio:
            rms = self._audio_rms(out)
            if rms > 0.0:
                gain = min(float(self.config.max_gain), float(self.config.target_rms) / max(rms, 1e-8))
                out = out * gain

        peak = float(np.max(np.abs(out))) if out.size > 0 else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)

        return np.asarray(out, dtype=np.float32)

    def _decode_and_standardize_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        if not audio_bytes:
            raise ValueError("Audio upload is empty")

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                tmp_path = Path(tmp.name)

            audio, source_sr = librosa.load(str(tmp_path), sr=None, mono=True)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        if source_sr is None:
            raise ValueError("Cannot decode sample rate from uploaded audio")

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("Decoded audio is empty")

        source_sr = int(source_sr)
        target_sr = int(self.config.sample_rate)
        if source_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr).astype(np.float32)

        duration_before = float(audio.shape[0] / target_sr)
        if duration_before < self.config.min_duration_sec:
            raise ValueError(
                f"Audio rejected: duration {duration_before:.2f}s is shorter than {self.config.min_duration_sec:.2f}s"
            )

        audio = self._spectral_denoise(audio)

        if self.config.trim_silence:
            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=float(self.config.trim_top_db),
                frame_length=int(self.config.trim_frame_length),
                hop_length=int(self.config.trim_hop_length),
            )
            if trimmed.size > 0:
                audio = np.asarray(trimmed, dtype=np.float32)

        audio, vad_meta = self._apply_vad(audio, sample_rate=target_sr)
        audio = np.asarray(audio, dtype=np.float32)

        audio = pipeline.trim_audio(audio, sample_rate=target_sr, max_sec=float(self.config.max_audio_sec))
        if audio.size == 0:
            raise ValueError("Audio is empty after standardization")

        duration_after = float(audio.shape[0] / target_sr)
        rms_before_norm = self._audio_rms(audio)
        rms_db_before_norm = self._audio_rms_db(audio)

        if duration_after < float(self.config.min_duration_sec):
            raise ValueError(
                f"Audio rejected after silence filtering: duration {duration_after:.2f}s is shorter than {self.config.min_duration_sec:.2f}s"
            )
        if (rms_before_norm < float(self.config.min_rms_energy)) or (rms_db_before_norm < float(self.config.min_rms_db)):
            raise ValueError(
                f"Audio rejected: low energy (rms={rms_before_norm:.6f}, rms_db={rms_db_before_norm:.2f} dBFS)"
            )

        audio = self._normalize_waveform(audio)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        processed_samples = int(audio.shape[0])
        audio_meta = {
            "duration_before_sec": duration_before,
            "duration_after_sec": duration_after,
            "rms_before_norm": float(rms_before_norm),
            "rms_db_before_norm": float(rms_db_before_norm),
            "rms_after_norm": float(self._audio_rms(audio)),
            "rms_db_after_norm": float(self._audio_rms_db(audio)),
            "vad_segments": int(vad_meta["segments"]),
            "pause_duration_sec": float(vad_meta["pause_samples"] / target_sr),
            "pause_ratio": float(vad_meta["pause_ratio"]),
            "gap_pause_duration_sec": float(vad_meta["gap_samples"] / target_sr),
            "denoise_enabled": bool(self.config.denoise_enabled),
            "vad_enabled": bool(self.config.vad_enabled),
            "trim_silence": bool(self.config.trim_silence),
        }
        return audio, source_sr, audio_meta

    def _phonemize_target_words(self, text: str, backend: Any) -> Tuple[List[str], List[List[str]]]:
        input_words = re.findall(r"[a-z']+", self._clean_transcript(text))
        if len(input_words) == 0:
            return [], []

        # Phonemize each word independently to keep 1-to-1 alignment with input words.
        try:
            from phonemizer.separator import Separator

            out = backend.phonemize(
                input_words,
                separator=Separator(phone=" ", word=" | "),
                strip=True,
            )
            if isinstance(out, list) and len(out) == len(input_words):
                aligned: List[List[str]] = []
                for item in out:
                    toks = [str(p).strip() for p in str(item).replace("|", " ").split() if str(p).strip()]
                    aligned.append(toks)
                return input_words, aligned
        except Exception:
            pass

        # Fallback to sentence-level phonemization only when word count still matches.
        try:
            fallback = pipeline.phonemize_words(text, backend)
            if len(fallback) == len(input_words):
                aligned = [[str(p).strip() for p in row if str(p).strip()] for row in fallback]
                return input_words, aligned
        except Exception:
            pass

        return input_words, [[] for _ in input_words]

    def _build_target_ipa_payload(
        self,
        input_words: List[str],
        transcript_encoding_mode: str,
        word_phones: List[List[str]],
        phone_symbols: List[str],
    ) -> Dict[str, Any]:
        tokens = [str(p).strip() for p in phone_symbols if str(p).strip()]
        if (len(tokens) == 0) and (len(word_phones) > 0):
            tokens = [str(p).strip() for row in word_phones for p in row if str(p).strip()]

        payload: Dict[str, Any] = {
            "mode": str(transcript_encoding_mode),
            "text": " ".join(tokens),
            "tokens": tokens,
            "token_count": int(len(tokens)),
            "words": [],
        }

        if (transcript_encoding_mode != "phonemizer_espeak") or (len(word_phones) == 0):
            return payload

        by_word: List[Dict[str, Any]] = []
        ipa_chunks: List[str] = []

        for idx in range(len(input_words)):
            phones = word_phones[idx] if idx < len(word_phones) else []
            phones_clean = [str(p).strip() for p in phones if str(p).strip()]
            ipa_word = " ".join(phones_clean)
            if ipa_word:
                ipa_chunks.append(ipa_word)

            by_word.append(
                {
                    "index": int(idx + 1),
                    "word": str(input_words[idx]),
                    "ipa": ipa_word,
                    "phones": phones_clean,
                }
            )

        payload["words"] = by_word
        payload["text"] = " | ".join(ipa_chunks)
        return payload

    def _encode_transcript(self, text: str) -> Tuple[List[int], List[str], str, Dict[str, Any]]:
        if not text or not text.strip():
            raise ValueError("Text is empty")

        transcript_encoding_mode = "phonemizer_espeak"
        word_phones: List[List[str]] = []
        ipa_input_words: List[str] = []
        ipa_word_phones: List[List[str]] = []
        try:
            backend = pipeline.build_phonemizer_backend()
            word_phones = pipeline.phonemize_words(text, backend)
            ipa_input_words, ipa_word_phones = self._phonemize_target_words(text, backend)
            token_ids, phone_symbols = pipeline.encode_word_phones(word_phones, self.processor.tokenizer, self.unk_id)
        except Exception as e:
            if not self.config.allow_text_tokenizer_fallback:
                raise RuntimeError(f"Phonemizer backend failed: {e}") from e
            token_ids, phone_symbols = pipeline.encode_text_fallback(text, self.processor.tokenizer, self.unk_id, self.blank_id)
            transcript_encoding_mode = "tokenizer_text_fallback"
            ipa_input_words = re.findall(r"[a-z']+", self._clean_transcript(text))
            ipa_word_phones = []

        if len(token_ids) == 0:
            raise ValueError("No valid phone tokens extracted from transcript")

        ipa_target = self._build_target_ipa_payload(
            input_words=ipa_input_words,
            transcript_encoding_mode=transcript_encoding_mode,
            word_phones=ipa_word_phones,
            phone_symbols=phone_symbols,
        )

        return token_ids, phone_symbols, transcript_encoding_mode, ipa_target

    def _clean_transcript(self, text: str) -> str:
        return " ".join(str(text).replace("|", " ").replace("\n", " ").strip().lower().split())

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "upload") -> Dict[str, Any]:
        if not self._ready:
            self.load()

        audio, source_sr, audio_meta = self._decode_and_standardize_audio(audio_bytes)

        with self._lock:
            x = self.processor(audio, sampling_rate=int(self.config.sample_rate), return_tensors="pt")
            input_values = x.input_values.to(self.device)
            attention_mask = x.attention_mask.to(self.device) if "attention_mask" in x else None
            with torch.no_grad():
                out = self.ctc_model(input_values, attention_mask=attention_mask)

        logits = out.logits[0].detach().float().cpu()
        probs = torch.softmax(logits, dim=-1)
        frame_conf = probs.max(dim=-1).values
        asr_conf = float(frame_conf.mean().item()) if frame_conf.numel() > 0 else 0.0

        pred_ids = torch.argmax(logits, dim=-1).unsqueeze(0)
        try:
            decoded = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        except TypeError:
            decoded = self.processor.batch_decode(pred_ids)[0]
        transcript = self._clean_transcript(decoded)
        if not transcript:
            raise RuntimeError("ASR transcript is empty after decoding; cannot score conversation turn")

        return {
            "filename": filename,
            "sample_rate_input": int(source_sr),
            "sample_rate_model": int(self.config.sample_rate),
            "transcript": transcript,
            "asr_confidence": asr_conf,
            "audio_quality": audio_meta,
            "meta": {
                "asr_model": str(self.config.ctc_model_name),
                "reject_reason": None,
            },
        }

    def _grammar_signal(self, transcript: str) -> Dict[str, Any]:
        words = re.findall(r"[a-z']+", str(transcript).lower())
        word_count = int(len(words))

        issues: Dict[str, Dict[str, Any]] = {}

        def add_issue(key: str, message: str, weight: float, example: str) -> None:
            slot = issues.get(key)
            if slot is None:
                slot = {
                    "type": key,
                    "message": message,
                    "count": 0,
                    "weight": float(weight),
                    "examples": [],
                }
                issues[key] = slot
            slot["count"] = int(slot["count"]) + 1
            if example and (len(slot["examples"]) < 5):
                slot["examples"].append(example)

        for i in range(1, word_count):
            if words[i] == words[i - 1]:
                add_issue(
                    key="repeated_word",
                    message="Repeated consecutive words detected",
                    weight=0.7,
                    example=f"{words[i - 1]} {words[i]}",
                )

        subject_set = {"he", "she", "it"}
        base_verbs = {
            "go",
            "work",
            "play",
            "study",
            "speak",
            "live",
            "need",
            "want",
            "like",
            "look",
            "say",
            "talk",
            "use",
            "make",
            "have",
            "do",
            "try",
            "watch",
        }
        irregular_ok = {"has", "does", "goes"}
        for i in range(word_count - 1):
            if words[i] in subject_set:
                nxt = words[i + 1]
                if (nxt in base_verbs) and (nxt not in irregular_ok) and (not nxt.endswith("s")):
                    add_issue(
                        key="missing_third_person_s",
                        message="Possible missing third-person singular '-s'",
                        weight=1.6,
                        example=f"{words[i]} {nxt}",
                    )

        quantifiers = {
            "many",
            "several",
            "few",
            "these",
            "those",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        }
        for i in range(word_count - 1):
            if words[i] in quantifiers:
                nxt = words[i + 1]
                if re.fullmatch(r"[a-z']+", nxt) and (len(nxt) > 2) and (not nxt.endswith("s")):
                    add_issue(
                        key="missing_plural_s",
                        message="Possible missing plural '-s' after quantifier",
                        weight=1.4,
                        example=f"{words[i]} {nxt}",
                    )

        weighted_errors = 0.0
        for row in issues.values():
            weighted_errors += float(row["weight"]) * float(row["count"])

        error_rate = float(weighted_errors / max(word_count, 1))
        score_0_5 = float(np.clip(5.0 * np.exp(-1.25 * error_rate), 0.0, 5.0))
        band = score_band_0_5(
            score=score_0_5,
            weak_lt=float(self.config.band_weak_lt_0_5),
            good_gte=float(self.config.band_good_gte_0_5),
        )

        issue_list = sorted(
            issues.values(),
            key=lambda x: (float(x["weight"]) * int(x["count"]), int(x["count"])),
            reverse=True,
        )
        missing_s_count = int(
            issues.get("missing_third_person_s", {}).get("count", 0)
            + issues.get("missing_plural_s", {}).get("count", 0)
        )

        return {
            "profile_version": str(self.config.grammar_profile_version),
            "word_count": word_count,
            "weighted_error_count": float(weighted_errors),
            "error_rate": error_rate,
            "score_0_5": score_0_5,
            "score_0_100": float(np.clip(score_0_5 * 20.0, 0.0, 100.0)),
            "band": band,
            "missing_s_count": missing_s_count,
            "issues": issue_list,
        }

    def score_conversation(
        self,
        audio_bytes: bytes,
        filename: str = "upload",
        hr_prompt: str = "",
        include_phone_details: bool = False,
    ) -> Dict[str, Any]:
        asr_payload = self.transcribe_audio(audio_bytes=audio_bytes, filename=filename)
        transcript = str(asr_payload["transcript"])

        sentence_payload = self.score(text=transcript, audio_bytes=audio_bytes, filename=filename)
        grammar_signal = self._grammar_signal(transcript)

        pron_score_0_5 = float(sentence_payload.get("overall_score_0_5", sentence_payload["scores_0_5"]["final_smoothed"]))
        grammar_score_0_5 = float(grammar_signal["score_0_5"])
        pw = max(float(self.config.conversation_pronunciation_weight), 0.0)
        gw = max(float(self.config.conversation_grammar_weight), 0.0)
        ws = max(pw + gw, 1e-8)
        overall_score_0_5 = float(np.clip((pw * pron_score_0_5 + gw * grammar_score_0_5) / ws, 0.0, 5.0))
        overall_score_0_100 = float(np.clip(overall_score_0_5 * 20.0, 0.0, 100.0))
        overall_label = overall_label_0_5(
            score=overall_score_0_5,
            fair_gte=float(self.config.overall_fair_gte_0_5),
            good_gte=float(self.config.overall_good_gte_0_5),
        )
        sentence_band = score_band_0_5(
            score=overall_score_0_5,
            weak_lt=float(self.config.band_weak_lt_0_5),
            good_gte=float(self.config.band_good_gte_0_5),
        )

        phone_rows = sentence_payload.get("phones", [])
        th_scores = [
            float(r.get("pred_phn_0_5", 0.0))
            for r in phone_rows
            if ("θ" in str(r.get("phone", ""))) or ("ð" in str(r.get("phone", "")))
        ]
        ending_scores = [
            float(r.get("pred_phn_0_5", 0.0))
            for r in phone_rows
            if str(r.get("phone", "")) in {"s", "z", "ts", "dz", "ɪz", "əz"}
        ]
        weak_th_count = int(
            sum(
                1
                for r in phone_rows
                if ((("θ" in str(r.get("phone", ""))) or ("ð" in str(r.get("phone", "")))) and (str(r.get("phone_band", "")) == "weak"))
            )
        )
        weak_ending_count = int(
            sum(
                1
                for r in phone_rows
                if ((str(r.get("phone", "")) in {"s", "z", "ts", "dz", "ɪz", "əz"}) and (str(r.get("phone_band", "")) == "weak"))
            )
        )

        th_module_score = float(np.mean(th_scores)) if th_scores else float(sentence_payload["scores_0_5"]["word_accuracy_mean"])
        ending_penalty = float(min(grammar_signal["missing_s_count"], 5) * 0.25)
        ending_module_base = float(np.mean(ending_scores)) if ending_scores else float(sentence_payload["scores_0_5"]["word_accuracy_mean"])
        ending_module_score = float(np.clip(ending_module_base - ending_penalty, 0.0, 5.0))

        transcript_words = re.findall(r"[a-z']+", transcript.lower())
        cluster_words = [w for w in transcript_words if re.search(r"[bcdfghjklmnpqrstvwxyz]{2,}", w)]
        cluster_module_score = float(
            np.clip(float(sentence_payload["scores_0_5"]["word_accuracy_mean"]) - 0.12 * max(len(cluster_words) - 2, 0), 0.0, 5.0)
        )

        modules = [
            {
                "key": "pronunciation",
                "title": "Pronunciation",
                "score_0_5": pron_score_0_5,
                "score_0_100": float(np.clip(pron_score_0_5 * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(pron_score_0_5, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
            {
                "key": "grammar",
                "title": "Grammar",
                "score_0_5": grammar_score_0_5,
                "score_0_100": float(np.clip(grammar_score_0_5 * 20.0, 0.0, 100.0)),
                "band": str(grammar_signal["band"]),
            },
            {
                "key": "th_sounds",
                "title": "TH sounds /θ/ /ð/",
                "score_0_5": th_module_score,
                "score_0_100": float(np.clip(th_module_score * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(th_module_score, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
            {
                "key": "ending_sounds",
                "title": "Ending sounds /s/ /z/",
                "score_0_5": ending_module_score,
                "score_0_100": float(np.clip(ending_module_score * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(ending_module_score, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
            {
                "key": "consonant_clusters",
                "title": "Consonant clusters",
                "score_0_5": cluster_module_score,
                "score_0_100": float(np.clip(cluster_module_score * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(cluster_module_score, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
        ]

        common_issues: List[Dict[str, Any]] = []
        if weak_th_count > 0:
            common_issues.append(
                {
                    "type": "th_sound_confusion",
                    "count": weak_th_count,
                    "severity": "high" if weak_th_count >= 3 else "medium",
                    "detail": "TH phones show repeated weak pronunciation",
                }
            )
        if grammar_signal["missing_s_count"] > 0:
            common_issues.append(
                {
                    "type": "missing_final_s",
                    "count": int(grammar_signal["missing_s_count"]),
                    "severity": "high" if int(grammar_signal["missing_s_count"]) >= 3 else "medium",
                    "detail": "Potential missing plural/third-person '-s' patterns",
                }
            )
        if weak_ending_count > 0:
            common_issues.append(
                {
                    "type": "weak_ending_sounds",
                    "count": weak_ending_count,
                    "severity": "medium",
                    "detail": "Ending consonant phones are unstable",
                }
            )
        if (len(cluster_words) > 0) and (cluster_module_score < float(self.config.band_good_gte_0_5)):
            common_issues.append(
                {
                    "type": "consonant_cluster_instability",
                    "count": int(len(cluster_words)),
                    "severity": "medium",
                    "detail": "Words with consonant clusters may need clearer articulation",
                }
            )

        for row in grammar_signal["issues"]:
            common_issues.append(
                {
                    "type": str(row["type"]),
                    "count": int(row["count"]),
                    "severity": "high" if int(row["count"]) >= 3 else "medium",
                    "detail": str(row["message"]),
                    "examples": row.get("examples", []),
                }
            )

        result = {
            "mode": "conversation_turn",
            "filename": filename,
            "hr_prompt": str(hr_prompt).strip() or None,
            "transcript_asr": transcript,
            "asr_confidence": float(asr_payload["asr_confidence"]),
            "audio_duration_sec": float(asr_payload["audio_quality"]["duration_after_sec"]),
            "threshold_profile_version": str(self.config.threshold_profile_version),
            "overall_score_0_5": overall_score_0_5,
            "overall_score_0_100": overall_score_0_100,
            "overall_label": overall_label,
            "sentence_band": sentence_band,
            "fusion": {
                "pronunciation_weight": pw,
                "grammar_weight": gw,
            },
            "signals": {
                "pronunciation_score_0_5": pron_score_0_5,
                "grammar_score_0_5": grammar_score_0_5,
            },
            "pronunciation_summary": {
                "overall_score_0_5": float(sentence_payload.get("overall_score_0_5", pron_score_0_5)),
                "overall_score_0_100": float(sentence_payload.get("overall_score_0_100", np.clip(pron_score_0_5 * 20.0, 0.0, 100.0))),
                "overall_label": str(sentence_payload.get("overall_label", "")),
                "scores_0_5": sentence_payload.get("scores_0_5", {}),
            },
            "grammar_signal": grammar_signal,
            "modules": modules,
            "common_issues": common_issues,
            "speech_rate": sentence_payload.get("speech_rate", {}),
            "prosody_pattern": sentence_payload.get("prosody_pattern", {}),
            "meta": {
                "model_version": self.model_version,
                "calibration_version": self.calibration_version,
                "preprocessing_profile": self.preprocessing_profile,
                "threshold_profile_version": str(self.config.threshold_profile_version),
                "grammar_profile_version": str(self.config.grammar_profile_version),
                "asr_model": str(self.config.ctc_model_name),
                "reject_reason": None,
            },
        }

        if include_phone_details:
            result["phones"] = phone_rows

        return result

    def score_conversation_summary(
        self,
        turns: List[Dict[str, Any]],
        include_turn_details: bool = True,
    ) -> Dict[str, Any]:
        if not turns:
            raise ValueError("Conversation summary requires at least one user turn")

        pw = max(float(self.config.conversation_pronunciation_weight), 0.0)
        gw = max(float(self.config.conversation_grammar_weight), 0.0)
        ws = max(pw + gw, 1e-8)

        pron_numer = 0.0
        pron_denom = 0.0
        grammar_numer = 0.0
        grammar_denom = 0.0
        total_audio_duration_sec = 0.0

        th_scores: List[float] = []
        ending_scores: List[float] = []
        weak_th_count = 0
        weak_ending_count = 0
        cluster_word_count = 0
        missing_s_count_total = 0

        grammar_issue_aggr: Dict[str, Dict[str, Any]] = {}
        turn_summaries: List[Dict[str, Any]] = []

        for i, turn in enumerate(turns):
            turn_index = i + 1
            text = str(turn.get("text", "")).strip()
            audio_bytes = turn.get("audio_bytes")
            filename = str(turn.get("filename", "")).strip() or f"turn_{turn_index}.wav"
            hr_prompt = str(turn.get("hr_prompt", "")).strip()

            if not text:
                raise ValueError(f"Turn {turn_index} rejected: text is empty")
            if not isinstance(audio_bytes, (bytes, bytearray)) or (len(audio_bytes) == 0):
                raise ValueError(f"Turn {turn_index} rejected: audio is empty")

            sentence_payload = self.score(text=text, audio_bytes=bytes(audio_bytes), filename=filename)
            grammar_signal = self._grammar_signal(text)

            pron_score_0_5 = float(sentence_payload.get("overall_score_0_5", sentence_payload["scores_0_5"]["final_smoothed"]))
            grammar_score_0_5 = float(grammar_signal["score_0_5"])
            turn_overall_0_5 = float(np.clip((pw * pron_score_0_5 + gw * grammar_score_0_5) / ws, 0.0, 5.0))

            token_weight = float(max(int(sentence_payload.get("token_count", 0)), 1))
            word_weight = float(max(int(grammar_signal.get("word_count", 0)), 1))
            pron_numer += pron_score_0_5 * token_weight
            pron_denom += token_weight
            grammar_numer += grammar_score_0_5 * word_weight
            grammar_denom += word_weight

            duration_sec = float(sentence_payload.get("audio_duration_sec", 0.0))
            total_audio_duration_sec += duration_sec

            phone_rows = sentence_payload.get("phones", [])
            for row in phone_rows:
                phone = str(row.get("phone", ""))
                score_0_5 = float(row.get("pred_phn_0_5", 0.0))
                if ("θ" in phone) or ("ð" in phone):
                    th_scores.append(score_0_5)
                    if str(row.get("phone_band", "")) == "weak":
                        weak_th_count += 1
                if phone in {"s", "z", "ts", "dz", "ɪz", "əz"}:
                    ending_scores.append(score_0_5)
                    if str(row.get("phone_band", "")) == "weak":
                        weak_ending_count += 1

            words = re.findall(r"[a-z']+", text.lower())
            cluster_words = [w for w in words if re.search(r"[bcdfghjklmnpqrstvwxyz]{2,}", w)]
            cluster_word_count += int(len(cluster_words))

            missing_s_count_total += int(grammar_signal.get("missing_s_count", 0))
            for issue in grammar_signal.get("issues", []):
                key = str(issue.get("type", "unknown"))
                slot = grammar_issue_aggr.get(key)
                if slot is None:
                    slot = {
                        "type": key,
                        "message": str(issue.get("message", "")),
                        "count": 0,
                        "examples": [],
                    }
                    grammar_issue_aggr[key] = slot
                slot["count"] = int(slot["count"]) + int(issue.get("count", 0))
                for ex in issue.get("examples", []):
                    exs = slot["examples"]
                    if (len(exs) < 5) and (ex not in exs):
                        exs.append(ex)

            if include_turn_details:
                turn_summaries.append(
                    {
                        "turn_index": turn_index,
                        "filename": filename,
                        "hr_prompt": hr_prompt or None,
                        "text": text,
                        "audio_duration_sec": duration_sec,
                        "token_count": int(sentence_payload.get("token_count", 0)),
                        "pronunciation_score_0_5": pron_score_0_5,
                        "grammar_score_0_5": grammar_score_0_5,
                        "overall_score_0_5": turn_overall_0_5,
                        "overall_score_0_100": float(np.clip(turn_overall_0_5 * 20.0, 0.0, 100.0)),
                        "sentence_band": score_band_0_5(
                            score=turn_overall_0_5,
                            weak_lt=float(self.config.band_weak_lt_0_5),
                            good_gte=float(self.config.band_good_gte_0_5),
                        ),
                    }
                )

        if pron_denom <= 0.0:
            raise RuntimeError("Conversation summary failed: no valid pronunciation tokens in turns")

        pronunciation_score_0_5 = float(pron_numer / pron_denom)
        grammar_score_0_5 = float(grammar_numer / max(grammar_denom, 1e-8))
        overall_score_0_5 = float(np.clip((pw * pronunciation_score_0_5 + gw * grammar_score_0_5) / ws, 0.0, 5.0))
        overall_score_0_100 = float(np.clip(overall_score_0_5 * 20.0, 0.0, 100.0))

        overall_label = overall_label_0_5(
            score=overall_score_0_5,
            fair_gte=float(self.config.overall_fair_gte_0_5),
            good_gte=float(self.config.overall_good_gte_0_5),
        )
        sentence_band = score_band_0_5(
            score=overall_score_0_5,
            weak_lt=float(self.config.band_weak_lt_0_5),
            good_gte=float(self.config.band_good_gte_0_5),
        )

        th_module_score = float(np.mean(th_scores)) if th_scores else pronunciation_score_0_5
        ending_base = float(np.mean(ending_scores)) if ending_scores else pronunciation_score_0_5
        per_turn_missing_s = float(missing_s_count_total / max(len(turns), 1))
        ending_penalty = float(min(per_turn_missing_s, 5.0) * 0.25)
        ending_module_score = float(np.clip(ending_base - ending_penalty, 0.0, 5.0))

        cluster_penalty = 0.06 * max(cluster_word_count - (2 * len(turns)), 0)
        cluster_module_score = float(np.clip(pronunciation_score_0_5 - cluster_penalty, 0.0, 5.0))

        modules = [
            {
                "key": "pronunciation",
                "title": "Pronunciation",
                "score_0_5": pronunciation_score_0_5,
                "score_0_100": float(np.clip(pronunciation_score_0_5 * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(pronunciation_score_0_5, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
            {
                "key": "grammar",
                "title": "Grammar",
                "score_0_5": grammar_score_0_5,
                "score_0_100": float(np.clip(grammar_score_0_5 * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(grammar_score_0_5, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
            {
                "key": "th_sounds",
                "title": "TH sounds /θ/ /ð/",
                "score_0_5": th_module_score,
                "score_0_100": float(np.clip(th_module_score * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(th_module_score, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
            {
                "key": "ending_sounds",
                "title": "Ending sounds /s/ /z/",
                "score_0_5": ending_module_score,
                "score_0_100": float(np.clip(ending_module_score * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(ending_module_score, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
            {
                "key": "consonant_clusters",
                "title": "Consonant clusters",
                "score_0_5": cluster_module_score,
                "score_0_100": float(np.clip(cluster_module_score * 20.0, 0.0, 100.0)),
                "band": score_band_0_5(cluster_module_score, self.config.band_weak_lt_0_5, self.config.band_good_gte_0_5),
            },
        ]

        common_issues: List[Dict[str, Any]] = []
        if weak_th_count > 0:
            common_issues.append(
                {
                    "type": "th_sound_confusion",
                    "count": weak_th_count,
                    "severity": "high" if weak_th_count >= 3 else "medium",
                    "detail": "TH phones show repeated weak pronunciation",
                }
            )
        if missing_s_count_total > 0:
            common_issues.append(
                {
                    "type": "missing_final_s",
                    "count": int(missing_s_count_total),
                    "severity": "high" if int(missing_s_count_total) >= 3 else "medium",
                    "detail": "Potential missing plural/third-person '-s' patterns",
                }
            )
        if weak_ending_count > 0:
            common_issues.append(
                {
                    "type": "weak_ending_sounds",
                    "count": weak_ending_count,
                    "severity": "medium",
                    "detail": "Ending consonant phones are unstable",
                }
            )

        for row in sorted(grammar_issue_aggr.values(), key=lambda x: int(x["count"]), reverse=True):
            common_issues.append(
                {
                    "type": str(row["type"]),
                    "count": int(row["count"]),
                    "severity": "high" if int(row["count"]) >= 3 else "medium",
                    "detail": str(row["message"]),
                    "examples": row.get("examples", []),
                }
            )

        payload = {
            "mode": "conversation_summary",
            "turn_count": int(len(turns)),
            "processed_turn_count": int(len(turns)),
            "total_audio_duration_sec": float(total_audio_duration_sec),
            "threshold_profile_version": str(self.config.threshold_profile_version),
            "overall_score_0_5": overall_score_0_5,
            "overall_score_0_100": overall_score_0_100,
            "overall_label": overall_label,
            "sentence_band": sentence_band,
            "fusion": {
                "pronunciation_weight": pw,
                "grammar_weight": gw,
            },
            "signals": {
                "pronunciation_score_0_5": pronunciation_score_0_5,
                "grammar_score_0_5": grammar_score_0_5,
            },
            "modules": modules,
            "common_issues": common_issues,
            "meta": {
                "model_version": self.model_version,
                "calibration_version": self.calibration_version,
                "preprocessing_profile": self.preprocessing_profile,
                "threshold_profile_version": str(self.config.threshold_profile_version),
                "grammar_profile_version": str(self.config.grammar_profile_version),
                "asr_model": str(self.config.ctc_model_name),
                "reject_reason": None,
            },
        }

        if include_turn_details:
            payload["turn_summaries"] = turn_summaries

        return payload

    def score(self, text: str, audio_bytes: bytes, filename: str = "upload") -> Dict[str, Any]:
        if not self._ready:
            self.load()

        audio, source_sr, audio_meta = self._decode_and_standardize_audio(audio_bytes)

        with self._lock:
            token_ids, phone_symbols, transcript_encoding_mode, ipa_target = self._encode_transcript(text)

            x = self.processor(audio, sampling_rate=int(self.config.sample_rate), return_tensors="pt")
            input_values = x.input_values.to(self.device)
            attention_mask = x.attention_mask.to(self.device) if "attention_mask" in x else None
            with torch.no_grad():
                out = self.ctc_model(input_values, attention_mask=attention_mask)
            logits = out.logits[0].detach().float().cpu()

            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            confidence = probs.max(dim=-1).values
            hidden = logits[:, : min(128, logits.size(1))]

            change_times = pipeline.get_change_times(log_probs, token_ids, self.blank_id)
            bounds = pipeline.build_bounds(change_times, log_probs.size(0))

            if self.variant in ("energy", "pitch", "energy_pitch"):
                prosody_tracks = pipeline.compute_prosody_tracks(
                    audio,
                    n_frames=log_probs.size(0),
                    sample_rate=self.config.sample_rate,
                    fmin=self.config.prosody_fmin,
                    fmax=self.config.prosody_fmax,
                )
            else:
                prosody_tracks = (None, None, None)

            feats, used_token_ids, used_phone_symbols = pipeline.build_token_features(
                log_probs=log_probs,
                hidden=hidden,
                entropy=entropy,
                confidence=confidence,
                token_ids=token_ids,
                phone_symbols=phone_symbols,
                bounds=bounds,
                confusion_graph=self.confusion_graph,
                blank_id=self.blank_id,
                variant=self.variant,
                proj_dim=self.config.proj_dim,
                confusion_topk=self.config.confusion_topk,
                lambda_entropy=self.config.lambda_entropy,
                prosody_tracks=prosody_tracks,
            )

            # Keep timing of retained tokens to expose start/end ms for UI highlighting.
            used_bounds: List[Tuple[int, int]] = []
            upper = min(len(token_ids), max(0, len(bounds) - 1))
            for k in range(upper):
                st = int(bounds[k])
                en = int(bounds[k + 1])
                if en <= st:
                    continue
                used_bounds.append((st, en))

            max_phones = int(self.arch["max_seq_len"])
            seq_len = min(max_phones, len(feats), len(used_token_ids), len(used_phone_symbols))
            if seq_len <= 0:
                raise RuntimeError("No token-level features built from this wav/text pair")

            feat_arr = np.zeros((max_phones, int(self.arch["input_dim"])), dtype=np.float32)
            phn_ids = np.full((max_phones,), -1.0, dtype=np.float32)

            feat_stack = np.asarray(feats[:seq_len], dtype=np.float32)
            feat_stack = (feat_stack - self.norm_mean) / self.norm_std
            clip_v = max(float(self.config.feature_clip_value), 1.0)
            feat_stack = np.clip(feat_stack, -clip_v, clip_v)
            feat_stack = np.nan_to_num(feat_stack, nan=0.0, posinf=clip_v, neginf=-clip_v)
            feat_arr[:seq_len] = feat_stack

            mapped_phone_symbols: List[str] = []
            max_phone_id = int(self.arch.get("phn_vocab", 39) - 1)
            for i in range(seq_len):
                tok_id = int(used_token_ids[i])
                tok_sym = self.processor.tokenizer.convert_ids_to_tokens(tok_id)
                tok_sym = str(tok_sym) if tok_sym is not None else ""

                pid = int(self.phone_to_idx.get(tok_sym, self.unk_phone_idx))
                pid = max(0, min(max_phone_id, pid))
                phn_ids[i] = float(pid)

                if used_phone_symbols[i]:
                    mapped_phone_symbols.append(str(used_phone_symbols[i]))
                else:
                    mapped_phone_symbols.append(tok_sym)

            feat_tensor = torch.tensor(feat_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
            phn_tensor = torch.tensor(phn_ids, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                u1, u2, u3, u4, u5, p, w1, w2, w3 = self.model(feat_tensor, phn_tensor)

            utt_pred = torch.cat((u1, u2, u3, u4, u5), dim=1).squeeze(0).cpu().numpy()
            phn_pred = p.squeeze(0).squeeze(-1).cpu().numpy()
            word_pred = torch.cat((w1, w2, w3), dim=2).squeeze(0).cpu().numpy()

            valid = phn_ids >= 0
            valid_count = int(np.sum(valid))
            if valid_count > 0:
                word_mean = np.mean(word_pred[valid], axis=0)
            else:
                word_mean = np.zeros((3,), dtype=np.float32)

            duration_after_sec = float(audio_meta["duration_after_sec"])
            speech_rate_phones_per_sec = float(valid_count / max(duration_after_sec, 1e-6))
            speech_rate_score_0_5 = bell_curve_score_0_5(
                value=speech_rate_phones_per_sec,
                target=float(self.config.speech_rate_target_phones_per_sec),
                sigma=float(self.config.speech_rate_sigma),
            )

            frame_sec = duration_after_sec / max(int(log_probs.size(0)), 1)
            token_durations_sec: List[float] = []
            for idx in range(min(seq_len, max(0, len(bounds) - 1))):
                frame_count = max(int(bounds[idx + 1]) - int(bounds[idx]), 1)
                token_durations_sec.append(float(frame_count * frame_sec))

            timing_mean_sec = float(np.mean(token_durations_sec)) if token_durations_sec else 0.0
            timing_std_sec = float(np.std(token_durations_sec)) if token_durations_sec else 0.0
            timing_cv = float(timing_std_sec / max(timing_mean_sec, 1e-6)) if token_durations_sec else 0.0

            pause_ratio = float(audio_meta.get("pause_ratio", 0.0))
            pause_score_0_5 = bell_curve_score_0_5(
                value=pause_ratio,
                target=float(self.config.pause_ratio_target),
                sigma=float(self.config.pause_ratio_sigma),
            )
            timing_score_0_5 = bell_curve_score_0_5(
                value=timing_cv,
                target=float(self.config.timing_cv_target),
                sigma=float(self.config.timing_cv_sigma),
            )
            prosody_pattern_score_0_5 = float(np.clip(0.5 * (pause_score_0_5 + timing_score_0_5), 0.0, 5.0))

            phone_rows: List[Dict[str, Any]] = []
            valid_pos = np.where(valid)[0].tolist()
            frame_ms = float(duration_after_sec * 1000.0 / max(int(log_probs.size(0)), 1))
            for i, pos in enumerate(valid_pos):
                pred_val = float(phn_pred[pos])
                clipped = float(np.clip(pred_val, 0.0, 2.0))
                temp_scaled_0_2 = temperature_scale_0_2(
                    value=pred_val,
                    temperature=self.config.phone_temperature,
                    center=self.config.phone_temperature_center_0_2,
                )
                phn_score_0_5 = float(temp_scaled_0_2 * 2.5)

                if i < len(used_bounds):
                    st_frame, en_frame = used_bounds[i]
                else:
                    st_frame = int(round(i * max(int(log_probs.size(0)), 1) / max(seq_len, 1)))
                    en_frame = int(round((i + 1) * max(int(log_probs.size(0)), 1) / max(seq_len, 1)))

                start_ms = int(round(st_frame * frame_ms))
                end_ms = max(start_ms + 1, int(round(en_frame * frame_ms)))
                phone_band = score_band_0_5(
                    score=phn_score_0_5,
                    weak_lt=float(self.config.band_weak_lt_0_5),
                    good_gte=float(self.config.band_good_gte_0_5),
                )

                phone_rows.append(
                    {
                        "position": int(pos),
                        "phone": str(mapped_phone_symbols[i]) if i < len(mapped_phone_symbols) else "",
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": int(max(1, end_ms - start_ms)),
                        "pred_phn_norm": pred_val,
                        "pred_phn_clipped_0_2": clipped,
                        "pred_phn_0_5_linear": float(clipped * 2.5),
                        "pred_phn_temp_scaled_0_2": temp_scaled_0_2,
                        "pred_phn_0_5": phn_score_0_5,
                        "phone_band": phone_band,
                    }
                )

            phone_vals = [float(r["pred_phn_0_5"]) for r in phone_rows]
            phone_stats = {
                "mean": float(np.mean(phone_vals)) if phone_vals else 0.0,
                "std": float(np.std(phone_vals)) if phone_vals else 0.0,
                "min": float(np.min(phone_vals)) if phone_vals else 0.0,
                "max": float(np.max(phone_vals)) if phone_vals else 0.0,
            }

            scores_0_5 = {
                "utt_accuracy": self._score_0_5("utt_accuracy", float(utt_pred[0])),
                "utt_completeness": self._score_0_5("utt_completeness", float(utt_pred[1])),
                "utt_fluency": self._score_0_5("utt_fluency", float(utt_pred[2])),
                "utt_prosodic": self._score_0_5("utt_prosodic", float(utt_pred[3])),
                "utt_total": self._score_0_5("utt_total", float(utt_pred[4])),
                "word_accuracy_mean": self._score_0_5("word_accuracy_mean", float(word_mean[0])),
                "word_stress_mean": self._score_0_5("word_stress_mean", float(word_mean[1])),
                "word_total_mean": self._score_0_5("word_total_mean", float(word_mean[2])),
            }

            fluency_weight = float(np.clip(self.config.speech_rate_weight_on_fluency, 0.0, 1.0))
            prosodic_weight = float(np.clip(self.config.prosody_pattern_weight_on_prosodic, 0.0, 1.0))
            total_adjust_weight = float(np.clip(self.config.total_adjustment_weight, 0.0, 1.0))

            base_fluency = float(scores_0_5["utt_fluency"])
            base_prosodic = float(scores_0_5["utt_prosodic"])
            base_total = float(scores_0_5["utt_total"])

            scores_0_5["utt_fluency"] = float(
                np.clip((1.0 - fluency_weight) * base_fluency + fluency_weight * speech_rate_score_0_5, 0.0, 5.0)
            )
            scores_0_5["utt_prosodic"] = float(
                np.clip((1.0 - prosodic_weight) * base_prosodic + prosodic_weight * prosody_pattern_score_0_5, 0.0, 5.0)
            )
            adjusted_total_anchor = 0.5 * (scores_0_5["utt_fluency"] + scores_0_5["utt_prosodic"])
            scores_0_5["utt_total"] = float(
                np.clip((1.0 - total_adjust_weight) * base_total + total_adjust_weight * adjusted_total_anchor, 0.0, 5.0)
            )

            score_raw = {
                "utt_accuracy": float(utt_pred[0]),
                "utt_completeness": float(utt_pred[1]),
                "utt_fluency": float(utt_pred[2]),
                "utt_prosodic": float(utt_pred[3]),
                "utt_total": float(utt_pred[4]),
                "word_accuracy_mean": float(word_mean[0]),
                "word_stress_mean": float(word_mean[1]),
                "word_total_mean": float(word_mean[2]),
            }
            clip_min = float(min(self.config.calibration_clip_min, self.config.calibration_clip_max))
            clip_max = float(max(self.config.calibration_clip_min, self.config.calibration_clip_max))
            score_clipped = {k: float(np.clip(v, clip_min, clip_max)) for k, v in score_raw.items()}

            utt_w = float(self.config.smoothing_utt_weight)
            word_w = float(self.config.smoothing_word_weight)
            w_sum = max(utt_w + word_w, 1e-8)
            final_smoothed = (utt_w * scores_0_5["utt_total"] + word_w * scores_0_5["word_total_mean"]) / w_sum
            final_smoothed_pre_iso = float(np.clip(final_smoothed, 0.0, 5.0))
            scores_0_5["final_smoothed_pre_iso"] = final_smoothed_pre_iso
            scores_0_5["final_smoothed"] = float(self._calibrate_overall_0_5(final_smoothed_pre_iso))

            overall_score_0_5 = float(scores_0_5["final_smoothed"])
            overall_score_0_100 = float(np.clip(overall_score_0_5 * 20.0, 0.0, 100.0))
            overall_label = overall_label_0_5(
                score=overall_score_0_5,
                fair_gte=float(self.config.overall_fair_gte_0_5),
                good_gte=float(self.config.overall_good_gte_0_5),
            )

            word_band = score_band_0_5(
                score=float(scores_0_5["word_total_mean"]),
                weak_lt=float(self.config.band_weak_lt_0_5),
                good_gte=float(self.config.band_good_gte_0_5),
            )
            sentence_band = score_band_0_5(
                score=overall_score_0_5,
                weak_lt=float(self.config.band_weak_lt_0_5),
                good_gte=float(self.config.band_good_gte_0_5),
            )

            return {
                "filename": filename,
                "text": text,
                "ipa_target_text": str(ipa_target.get("text", "")),
                "ipa_target_words": ipa_target.get("words", []),
                "ipa_target_tokens": ipa_target.get("tokens", []),
                "ipa_target_mode": str(ipa_target.get("mode", transcript_encoding_mode)),
                "sample_rate_input": source_sr,
                "sample_rate_model": self.config.sample_rate,
                "audio_duration_sec": float(audio_meta["duration_after_sec"]),
                "transcript_encoding_mode": transcript_encoding_mode,
                "token_count": valid_count,
                "variant": self.variant,
                "threshold_profile_version": str(self.config.threshold_profile_version),
                "scores_normalized": score_raw,
                "scores_normalized_clipped_0_2": score_clipped,
                "scores_0_5": scores_0_5,
                "overall_score_0_5": overall_score_0_5,
                "overall_score_0_100": overall_score_0_100,
                "overall_label": overall_label,
                "word_band": word_band,
                "sentence_band": sentence_band,
                "bands": {
                    "threshold_profile_version": str(self.config.threshold_profile_version),
                    "weak_lt_0_5": float(self.config.band_weak_lt_0_5),
                    "good_gte_0_5": float(self.config.band_good_gte_0_5),
                },
                "score_calibration": {
                    "method": "sigmoid_train_stats",
                    "center_shift_sigma": self.config.calibration_center_shift_sigma,
                    "sigma_floor": self.config.calibration_sigma_floor,
                    "temperature": self.config.calibration_temperature,
                    "clip_range_0_2": [
                        float(min(self.config.calibration_clip_min, self.config.calibration_clip_max)),
                        float(max(self.config.calibration_clip_min, self.config.calibration_clip_max)),
                    ],
                },
                "phone_score_mapping": {
                    "method": "temperature_scaled_linear_0_2_to_0_5",
                    "temperature": float(self.config.phone_temperature),
                    "center_0_2": float(self.config.phone_temperature_center_0_2),
                },
                "score_smoothing": {
                    "formula": f"final_smoothed = ({utt_w:.3f} * utt_total + {word_w:.3f} * word_total_mean) / {w_sum:.3f}",
                    "utt_weight": utt_w,
                    "word_weight": word_w,
                },
                "score_adjustment": {
                    "fluency_from_speech_rate_weight": fluency_weight,
                    "prosodic_from_pattern_weight": prosodic_weight,
                    "total_rebalance_weight": total_adjust_weight,
                },
                "speech_rate": {
                    "phones_per_sec": speech_rate_phones_per_sec,
                    "target_phones_per_sec": float(self.config.speech_rate_target_phones_per_sec),
                    "sigma": float(self.config.speech_rate_sigma),
                    "score_0_5": float(speech_rate_score_0_5),
                },
                "prosody_pattern": {
                    "pause_duration_sec": float(audio_meta.get("pause_duration_sec", 0.0)),
                    "pause_ratio": pause_ratio,
                    "pause_score_0_5": float(pause_score_0_5),
                    "syllable_timing_mean_sec": timing_mean_sec,
                    "syllable_timing_std_sec": timing_std_sec,
                    "syllable_timing_cv": timing_cv,
                    "timing_score_0_5": float(timing_score_0_5),
                    "combined_score_0_5": float(prosody_pattern_score_0_5),
                },
                "audio_quality": {
                    "duration_before_sec": float(audio_meta["duration_before_sec"]),
                    "duration_after_sec": float(audio_meta["duration_after_sec"]),
                    "rms_before_norm": float(audio_meta["rms_before_norm"]),
                    "rms_db_before_norm": float(audio_meta["rms_db_before_norm"]),
                    "rms_after_norm": float(audio_meta["rms_after_norm"]),
                    "rms_db_after_norm": float(audio_meta["rms_db_after_norm"]),
                    "vad_segments": int(audio_meta["vad_segments"]),
                    "pause_duration_sec": float(audio_meta.get("pause_duration_sec", 0.0)),
                    "pause_ratio": float(audio_meta.get("pause_ratio", 0.0)),
                    "gap_pause_duration_sec": float(audio_meta.get("gap_pause_duration_sec", 0.0)),
                    "trim_silence": bool(audio_meta["trim_silence"]),
                    "vad_enabled": bool(audio_meta["vad_enabled"]),
                    "denoise_enabled": bool(audio_meta["denoise_enabled"]),
                },
                "reject_policy": {
                    "min_duration_sec": float(self.config.min_duration_sec),
                    "min_rms_energy": float(self.config.min_rms_energy),
                    "min_rms_db": float(self.config.min_rms_db),
                },
                "meta": {
                    "model_version": self.model_version,
                    "calibration_version": self.calibration_version,
                    "preprocessing_profile": self.preprocessing_profile,
                    "reject_reason": None,
                },
                "phone_score_stats_0_5": phone_stats,
                "phones": phone_rows,
            }

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "device": str(self.device) if self.device is not None else None,
            "variant": self.variant or None,
            "checkpoint": str(self.config.checkpoint.expanduser().resolve()),
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "preprocessing_profile": self.preprocessing_profile,
            "threshold_profile_version": str(self.config.threshold_profile_version),
        }
