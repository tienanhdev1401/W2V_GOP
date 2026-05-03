# -*- coding: utf-8 -*-

"""
Score a single WAV file + transcript with the finalized W2V-GOPT main checkpoint.

This script recreates the pitch feature pipeline used during training:
- Wav2Vec2 CTC alignment
- adaptive confusion-based local GOP features
- prosody features (pitch/voiced ratio)
- normalization by train feature stats
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from app.model_arch import Wav2VecGOPT


class NotebookW2VGOPT(nn.Module):
    """
    Architecture used in the research notebook checkpoints (adapter/encoder/cls/pos keys).
    """

    def __init__(self, input_dim, embed_dim, depth, heads, max_len, phn_vocab, adapter_dim=256, dropout=0.1):
        super().__init__()
        self.max_len = int(max_len)
        self.phn_vocab = int(phn_vocab)

        self.adapter = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(adapter_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(adapter_dim), int(embed_dim)),
        )
        self.phn_proj = nn.Linear(self.phn_vocab + 1, int(embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(heads),
            dim_feedforward=int(embed_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=int(depth), enable_nested_tensor=False)

        self.cls = nn.Parameter(torch.zeros(1, 5, int(embed_dim)))
        self.pos = nn.Parameter(torch.zeros(1, self.max_len + 5, int(embed_dim)))

        self.head_utt = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(int(embed_dim)), nn.Linear(int(embed_dim), 1)) for _ in range(5)]
        )
        self.head_phn = nn.Sequential(nn.LayerNorm(int(embed_dim)), nn.Linear(int(embed_dim), 1))
        self.head_word_acc = nn.Sequential(nn.LayerNorm(int(embed_dim)), nn.Linear(int(embed_dim), 1))
        self.head_word_stress = nn.Sequential(nn.LayerNorm(int(embed_dim)), nn.Linear(int(embed_dim), 1))
        self.head_word_total = nn.Sequential(nn.LayerNorm(int(embed_dim)), nn.Linear(int(embed_dim), 1))

    def forward(self, x, phn):
        B, T, _ = x.shape

        h = self.adapter(x)
        p = torch.clamp(phn.long(), min=-1, max=self.phn_vocab - 1)
        one_hot = F.one_hot(p + 1, num_classes=self.phn_vocab + 1).float()
        h = h + self.phn_proj(one_hot)

        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, : h.size(1)]

        mask_t = phn < 0
        mask_cls = torch.zeros((B, 5), dtype=torch.bool, device=mask_t.device)
        mask = torch.cat([mask_cls, mask_t], dim=1)

        h = self.encoder(h, src_key_padding_mask=mask)

        u = [self.head_utt[i](h[:, i]) for i in range(5)]
        ph = self.head_phn(h[:, 5:])
        w_acc = self.head_word_acc(h[:, 5:])
        w_stress = self.head_word_stress(h[:, 5:])
        w_total = self.head_word_total(h[:, 5:])
        return u[0], u[1], u[2], u[3], u[4], ph, w_acc, w_stress, w_total


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    has_module = all(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def infer_arch_from_state_dict(state_dict, num_heads):
    # Format A: src/models/Wav2VecGOPT checkpoint
    if (
        "input_adapter.0.weight" in state_dict
        and "input_adapter.1.weight" in state_dict
        and "input_adapter.4.weight" in state_dict
        and "pos_embed" in state_dict
    ):
        input_dim = int(state_dict["input_adapter.0.weight"].shape[0])
        adapter_dim = int(state_dict["input_adapter.1.weight"].shape[0])
        embed_dim = int(state_dict["input_adapter.4.weight"].shape[0])
        max_seq_len = int(state_dict["pos_embed"].shape[1] - 5)

        block_ids = set()
        for k in state_dict.keys():
            if k.startswith("blocks."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    block_ids.add(int(parts[1]))
        if not block_ids:
            raise KeyError("Cannot infer transformer depth from checkpoint blocks.* keys")

        depth = max(block_ids) + 1
        use_phn_embedding = "phn_proj.weight" in state_dict
        phn_vocab = 39
        if use_phn_embedding:
            phn_vocab = int(state_dict["phn_proj.weight"].shape[1] - 1)

        return {
            "model_type": "wav2vec_gopt",
            "input_dim": input_dim,
            "adapter_dim": adapter_dim,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": int(num_heads),
            "max_seq_len": max_seq_len,
            "use_phn_embedding": use_phn_embedding,
            "phn_vocab": int(phn_vocab),
        }

    # Format B: notebook W2VGOPT checkpoint
    if (
        "adapter.0.weight" in state_dict
        and "adapter.1.weight" in state_dict
        and "adapter.4.weight" in state_dict
        and "pos" in state_dict
    ):
        input_dim = int(state_dict["adapter.0.weight"].shape[0])
        adapter_dim = int(state_dict["adapter.1.weight"].shape[0])
        embed_dim = int(state_dict["adapter.4.weight"].shape[0])
        max_seq_len = int(state_dict["pos"].shape[1] - 5)

        layer_ids = set()
        for k in state_dict.keys():
            if k.startswith("encoder.layers."):
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_ids.add(int(parts[2]))
        if not layer_ids:
            raise KeyError("Cannot infer transformer depth from checkpoint encoder.layers.* keys")

        phn_vocab = 39
        if "phn_proj.weight" in state_dict:
            phn_vocab = int(state_dict["phn_proj.weight"].shape[1] - 1)

        return {
            "model_type": "notebook_w2vgopt",
            "input_dim": input_dim,
            "adapter_dim": adapter_dim,
            "embed_dim": embed_dim,
            "depth": max(layer_ids) + 1,
            "num_heads": int(num_heads),
            "max_seq_len": max_seq_len,
            "use_phn_embedding": True,
            "phn_vocab": int(phn_vocab),
        }

    raise KeyError(
        "Unsupported checkpoint layout. Expected Wav2VecGOPT keys (input_adapter.*) or notebook W2VGOPT keys (adapter.*)."
    )


def resolve_paths(args):
    repo_root = Path(__file__).resolve().parents[2]
    default_final_dir = repo_root / "final" / "w2v_gopt_research"

    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None

    if manifest is None:
        default_manifest = default_final_dir / "final_selected_model.json"
        if default_manifest.exists():
            manifest = default_manifest

    manifest_payload = {}
    if manifest and manifest.exists():
        with open(manifest, "r", encoding="utf-8") as f:
            manifest_payload = json.load(f)

    if checkpoint is None:
        ckpt_from_manifest = manifest_payload.get("main_checkpoint")
        if ckpt_from_manifest:
            candidate_ckpt = Path(ckpt_from_manifest).expanduser().resolve()
            if candidate_ckpt.exists():
                checkpoint = candidate_ckpt

    if checkpoint is None:
        default_ckpt = default_final_dir / "best_w2v_gopt_research_main.pth"
        if default_ckpt.exists():
            checkpoint = default_ckpt.resolve()

    if checkpoint is None:
        raise FileNotFoundError("Checkpoint path is required (arg --checkpoint or manifest main_checkpoint)")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if data_dir is None:
        data_dir_from_manifest = manifest_payload.get("data_dir")
        if data_dir_from_manifest:
            candidate_data_dir = Path(data_dir_from_manifest).expanduser().resolve()
            if candidate_data_dir.exists():
                data_dir = candidate_data_dir

    if data_dir is None:
        candidate = checkpoint.parent / "seq_data_w2v_research_pitch"
        if candidate.exists():
            data_dir = candidate

    if data_dir is None:
        candidate = default_final_dir / "seq_data_w2v_research_pitch"
        if candidate.exists():
            data_dir = candidate.resolve()

    if data_dir is None or (not data_dir.exists()):
        raise FileNotFoundError("Data directory not found. Set --data-dir explicitly.")

    confusion_json = Path(args.confusion_json).expanduser().resolve() if args.confusion_json else None
    if confusion_json is None:
        candidate = checkpoint.parent / "adaptive_confusion.json"
        if candidate.exists():
            confusion_json = candidate
    if confusion_json is None:
        candidate = default_final_dir / "adaptive_confusion.json"
        if candidate.exists():
            confusion_json = candidate
    if confusion_json is None or (not confusion_json.exists()):
        raise FileNotFoundError("adaptive_confusion.json not found. Set --confusion-json explicitly.")

    phone_map = Path(args.phone_map).expanduser().resolve() if args.phone_map else None
    if phone_map is None:
        candidate = data_dir / "phone_to_idx.json"
        if candidate.exists():
            phone_map = candidate
    if phone_map is None or (not phone_map.exists()):
        raise FileNotFoundError("phone_to_idx.json not found. Set --phone-map explicitly.")

    wav_path = Path(args.wav).expanduser().resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else checkpoint.parent / "single_wav_score.json"
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else checkpoint.parent / "single_wav_phone_scores.csv"

    return checkpoint, data_dir, confusion_json, phone_map, wav_path, output_json, output_csv


def load_confusion_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    graph = {}
    for k, v in raw.items():
        key = int(k)
        graph[key] = [int(x) for x in v]
    return graph


def load_phone_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {str(k): int(v) for k, v in data.items()}
    unk_idx = int(out.get("<unk_phone>", 38))
    return out, unk_idx


def load_feature_norm(data_dir):
    tr_feat = np.load(data_dir / "tr_feat.npy", mmap_mode="r")
    tr_phn = np.load(data_dir / "tr_label_phn.npy", mmap_mode="r")

    valid = tr_phn[:, :, 1] >= 0
    if not np.any(valid):
        feat_dim = int(tr_feat.shape[-1])
        return np.zeros((feat_dim,), dtype=np.float32), np.ones((feat_dim,), dtype=np.float32)

    valid_feat = np.asarray(tr_feat[valid], dtype=np.float32)
    mean = np.mean(valid_feat, axis=0).astype(np.float32)
    std = np.clip(np.std(valid_feat, axis=0), 1e-6, None).astype(np.float32)
    return mean, std


def infer_variant_from_input_dim(input_dim, proj_dim=96):
    extra_dim = int(input_dim) - (5 + int(proj_dim))
    mapping = {
        0: "baseline",
        1: "energy",
        2: "pitch",
        3: "energy_pitch",
    }
    if extra_dim not in mapping:
        raise ValueError(
            f"Unsupported input_dim={input_dim}. Expected 5 + {proj_dim} + extra, where extra in [0,1,2,3]."
        )
    return mapping[extra_dim]


def trim_audio(audio, sample_rate, max_sec):
    max_len = int(sample_rate * max_sec)
    if audio.shape[0] > max_len:
        return audio[:max_len]
    return audio


def get_trellis(log_probs, tokens, blank):
    T = log_probs.size(0)
    N = len(tokens)
    tr = torch.full((T + 1, N + 1), -1e9)
    tr[0, 0] = 0.0
    tr[1:, 0] = torch.cumsum(log_probs[:, blank], dim=0)
    tok = torch.tensor(tokens, dtype=torch.long)
    for t in range(T):
        stay = tr[t, 1:] + log_probs[t, blank]
        change = tr[t, :-1] + log_probs[t, tok]
        tr[t + 1, 1:] = torch.maximum(stay, change)
    return tr


def get_change_times(log_probs, tokens, blank):
    if len(tokens) == 0:
        return []
    T = log_probs.size(0)
    tr = get_trellis(log_probs, tokens, blank)
    tok = torch.tensor(tokens, dtype=torch.long)

    t, j = T, len(tokens)
    changes = [None] * len(tokens)

    while t > 0 and j > 0:
        p_stay = tr[t - 1, j] + log_probs[t - 1, blank]
        p_change = tr[t - 1, j - 1] + log_probs[t - 1, tok[j - 1]]
        if p_change >= p_stay:
            changes[j - 1] = t - 1
            j -= 1
        t -= 1

    if j > 0:
        return np.linspace(0, max(T - 1, 0), len(tokens), dtype=int).tolist()

    for i in range(len(changes)):
        if changes[i] is None:
            changes[i] = 0 if i == 0 else changes[i - 1]

    return changes


def build_bounds(change_times, n_frames):
    if len(change_times) == 0:
        return [0, n_frames]
    bounds = [0]
    for i in range(1, len(change_times)):
        bounds.append((change_times[i - 1] + change_times[i]) // 2)
    bounds.append(n_frames)

    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = min(n_frames, bounds[i - 1] + 1)
    bounds[-1] = n_frames
    return bounds


def interp_track(track, target_len, fill_value=0.0):
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)

    x = np.asarray(track, dtype=np.float32)
    if x.size == 0:
        return np.full((target_len,), float(fill_value), dtype=np.float32)
    if x.size == target_len:
        return x.astype(np.float32, copy=False)
    if x.size == 1:
        return np.full((target_len,), float(x[0]), dtype=np.float32)

    src = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def compute_prosody_tracks(audio, n_frames, sample_rate, fmin, fmax):
    n_frames = int(max(0, n_frames))
    if n_frames <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z

    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=-1)

    if y.size == 0:
        z = np.zeros((n_frames,), dtype=np.float32)
        return z, z.copy(), z.copy()

    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / (peak + 1e-8)

    frame_length = 1024
    hop_length = 256

    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    except Exception:
        rms = np.zeros((1,), dtype=np.float32)

    rms = np.asarray(rms, dtype=np.float32)
    log_energy = np.log(np.maximum(rms, 1e-6))
    e_mean = float(np.mean(log_energy))
    e_std = float(np.std(log_energy) + 1e-6)
    norm_energy = (log_energy - e_mean) / e_std

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=float(fmin),
            fmax=float(fmax),
            sr=int(sample_rate),
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        f0 = None
        voiced_flag = None

    if f0 is None:
        f0 = np.full((norm_energy.shape[0],), np.nan, dtype=np.float32)
    else:
        f0 = np.asarray(f0, dtype=np.float32)

    if voiced_flag is None:
        voiced = np.zeros((f0.shape[0],), dtype=np.float32)
    else:
        voiced = np.asarray(voiced_flag, dtype=np.float32)
        if voiced.shape[0] != f0.shape[0]:
            voiced = interp_track(voiced, f0.shape[0], fill_value=0.0)

    pitch_norm = np.zeros((f0.shape[0],), dtype=np.float32)
    voiced_mask = np.isfinite(f0) & (f0 > 0.0) & (voiced > 0.5)
    if np.any(voiced_mask):
        log_f0 = np.log(np.maximum(f0[voiced_mask], 1e-6))
        p_mean = float(np.mean(log_f0))
        p_std = float(np.std(log_f0) + 1e-6)
        pitch_norm[voiced_mask] = (log_f0 - p_mean) / p_std

    energy_ctc = interp_track(norm_energy, n_frames, fill_value=0.0)
    pitch_ctc = interp_track(pitch_norm, n_frames, fill_value=0.0)
    voiced_ctc = np.clip(interp_track(voiced, n_frames, fill_value=0.0), 0.0, 1.0)

    return energy_ctc, pitch_ctc, voiced_ctc


def build_token_features(
    log_probs,
    hidden,
    entropy,
    confidence,
    token_ids,
    phone_symbols,
    bounds,
    confusion_graph,
    blank_id,
    variant,
    proj_dim,
    confusion_topk,
    lambda_entropy,
    prosody_tracks,
):
    hdim = hidden.size(1)
    generator = torch.Generator().manual_seed(1234)
    proj_matrix = torch.randn(hdim, proj_dim, generator=generator) / math.sqrt(max(hdim, 1))

    energy_track, pitch_track, voiced_track = prosody_tracks

    feats = []
    used_token_ids = []
    used_phone_symbols = []

    for k, tok in enumerate(token_ids):
        st, en = int(bounds[k]), int(bounds[k + 1])
        if en <= st:
            continue

        m = log_probs[st:en].mean(dim=0)
        true_s = float(m[tok].item())

        cands = confusion_graph.get(int(tok), [])
        cands = [int(c) for c in cands if int(c) != int(tok) and int(c) != int(blank_id)]
        if len(cands) == 0:
            top = torch.topk(m, k=min(10, m.numel())).indices.tolist()
            cands = [int(c) for c in top if int(c) != int(tok) and int(c) != int(blank_id)][: int(confusion_topk)]

        if len(cands) > 0:
            alt_best = max(float(m[c].item()) for c in cands)
        else:
            alt_best = true_s

        top_static = torch.topk(m, k=min(6, m.numel())).indices.tolist()
        alt_static = true_s
        for c in top_static:
            if int(c) != int(tok) and int(c) != int(blank_id):
                alt_static = float(m[c].item())
                break

        dur = max(1, en - st)
        mean_ent = float(entropy[st:en].mean().item())
        mean_conf = float(confidence[st:en].mean().item())

        delta_adp = true_s - alt_best
        delta_static = true_s - alt_static
        local_gop = (delta_adp / math.sqrt(dur)) - float(lambda_entropy) * mean_ent

        h = hidden[st:en].mean(dim=0)
        hproj = torch.matmul(h, proj_matrix).numpy().astype(np.float32)

        extra = []
        if variant in ("energy", "energy_pitch"):
            extra.append(float(np.mean(energy_track[st:en])) if energy_track is not None else 0.0)

        if variant in ("pitch", "energy_pitch"):
            if (pitch_track is not None) and (voiced_track is not None):
                seg_voiced = np.asarray(voiced_track[st:en], dtype=np.float32)
                seg_pitch = np.asarray(pitch_track[st:en], dtype=np.float32)
                voiced_ratio = float(np.mean(seg_voiced)) if seg_voiced.size > 0 else 0.0
                if np.any(seg_voiced > 0.5):
                    pitch_val = float(np.mean(seg_pitch[seg_voiced > 0.5]))
                else:
                    pitch_val = 0.0
            else:
                pitch_val = 0.0
                voiced_ratio = 0.0
            extra.extend([pitch_val, voiced_ratio])

        vec = np.empty((5 + proj_dim + len(extra),), dtype=np.float32)
        vec[0] = float(local_gop)
        vec[1] = float(delta_static / math.sqrt(dur))
        vec[2] = float(math.log1p(dur))
        vec[3] = float(mean_ent)
        vec[4] = float(mean_conf)
        vec[5 : 5 + proj_dim] = hproj
        if len(extra) > 0:
            vec[5 + proj_dim :] = np.asarray(extra, dtype=np.float32)

        feats.append(vec)
        used_token_ids.append(int(tok))
        used_phone_symbols.append(str(phone_symbols[k]) if k < len(phone_symbols) else "")

    return feats, used_token_ids, used_phone_symbols


def score_to_5(x):
    return float(np.clip(float(x) * 5.0, 0.0, 5.0))


def configure_espeak_library():
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return

    # Common install paths for eSpeak NG on Windows.
    candidates = [
        Path("C:/Program Files/eSpeak NG/libespeak-ng.dll"),
        Path("C:/Program Files (x86)/eSpeak NG/libespeak-ng.dll"),
    ]
    for dll_path in candidates:
        if dll_path.is_file():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(dll_path)
            return


def build_phonemizer_backend():
    configure_espeak_library()

    try:
        from phonemizer.backend import EspeakBackend
    except Exception as e:
        raise RuntimeError(
            "phonemizer is not installed. Install it with: pip install phonemizer"
        ) from e

    try:
        return EspeakBackend(
            "en-us",
            preserve_punctuation=False,
            with_stress=False,
            language_switch="remove-flags",
            words_mismatch="ignore",
        )
    except Exception as e:
        raise RuntimeError(
            "Cannot initialize Espeak backend. Ensure espeak-ng is installed and PHONEMIZER_ESPEAK_LIBRARY is set if needed."
        ) from e


def phonemize_words(text, backend):
    from phonemizer.separator import Separator

    out = backend.phonemize(
        [str(text)],
        separator=Separator(phone=" ", word=" | "),
        strip=True,
    )
    if not isinstance(out, list) or len(out) == 0:
        return []

    words = []
    for w in out[0].split("|"):
        toks = [p.strip() for p in w.strip().split() if p.strip()]
        if toks:
            words.append(toks)
    return words


def encode_word_phones(word_phones, tokenizer, unk_id):
    token_ids = []
    phone_symbols = []
    for phones in word_phones:
        for p in phones:
            tid = tokenizer.convert_tokens_to_ids(p)
            if tid is None:
                continue
            tid = int(tid)
            if tid < 0:
                continue
            if (unk_id is not None) and (tid == int(unk_id)):
                continue
            token_ids.append(tid)
            phone_symbols.append(str(p))
    return token_ids, phone_symbols


def encode_text_fallback(text, tokenizer, unk_id, blank_id):
    token_ids = []
    phone_symbols = []

    for tok in tokenizer.tokenize(str(text).lower()):
        if tok == "|":
            continue
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None:
            continue
        tid = int(tid)
        if tid < 0:
            continue
        if (unk_id is not None) and (tid == int(unk_id)):
            continue
        if (blank_id is not None) and (tid == int(blank_id)):
            continue
        token_ids.append(tid)
        phone_symbols.append(str(tok))

    return token_ids, phone_symbols


def load_ctc_processor(model_name):
    try:
        return Wav2Vec2Processor.from_pretrained(model_name)
    except Exception:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        if isinstance(tokenizer, bool):
            raise RuntimeError(
                "Failed to load a valid tokenizer for the CTC model. "
                "Try transformers==4.38.2 or use a different ctc model checkpoint."
            )
        return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def save_phone_rows_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["position", "phone", "pred_phn_norm", "pred_phn_clipped_0_2", "pred_phn_0_5"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--wav", type=str, required=True, help="Path to WAV file (16kHz mono recommended)")
    parser.add_argument("--text", type=str, required=True, help="Transcript text")

    parser.add_argument("--checkpoint", type=str, default=None, help="Path to main checkpoint")
    parser.add_argument("--manifest", type=str, default=None, help="Path to final selection manifest")
    parser.add_argument("--data-dir", type=str, default=None, help="Feature directory with tr_*.npy")
    parser.add_argument("--confusion-json", type=str, default=None, help="Path to adaptive_confusion.json")
    parser.add_argument("--phone-map", type=str, default=None, help="Path to phone_to_idx.json")

    parser.add_argument("--ctc-model", type=str, default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-audio-sec", type=float, default=8.0)
    parser.add_argument("--proj-dim", type=int, default=96)
    parser.add_argument("--confusion-topk", type=int, default=3)
    parser.add_argument("--lambda-entropy", type=float, default=0.05)
    parser.add_argument("--prosody-fmin", type=float, default=50.0)
    parser.add_argument("--prosody-fmax", type=float, default=350.0)

    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads for architecture reconstruction")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--allow-text-tokenizer-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If espeak phonemizer is unavailable, fall back to tokenizer-based text tokens.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main():
    args = build_argparser().parse_args()

    checkpoint, data_dir, confusion_json, phone_map_path, wav_path, output_json, output_csv = resolve_paths(args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    raw_state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        raw_state = raw_state["state_dict"]
    if not isinstance(raw_state, dict):
        raise ValueError("Unsupported checkpoint format: expected a state_dict dictionary")

    state = strip_module_prefix(raw_state)
    arch = infer_arch_from_state_dict(state, num_heads=args.num_heads)

    if arch["model_type"] == "wav2vec_gopt":
        model = Wav2VecGOPT(
            embed_dim=arch["embed_dim"],
            num_heads=arch["num_heads"],
            depth=arch["depth"],
            input_dim=arch["input_dim"],
            adapter_dim=arch["adapter_dim"],
            adapter_dropout=0.1,
            max_seq_len=arch["max_seq_len"],
            use_phn_embedding=arch["use_phn_embedding"],
        )
    elif arch["model_type"] == "notebook_w2vgopt":
        model = NotebookW2VGOPT(
            input_dim=arch["input_dim"],
            embed_dim=arch["embed_dim"],
            depth=arch["depth"],
            heads=arch["num_heads"],
            max_len=arch["max_seq_len"],
            phn_vocab=arch["phn_vocab"],
            adapter_dim=arch["adapter_dim"],
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {arch['model_type']}")

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing={missing}, Unexpected={unexpected}")

    model = model.to(device)
    model.eval()

    variant = infer_variant_from_input_dim(arch["input_dim"], proj_dim=args.proj_dim)
    confusion_graph = load_confusion_graph(confusion_json)
    phone_to_idx, unk_phone_idx = load_phone_map(phone_map_path)
    norm_mean, norm_std = load_feature_norm(data_dir)
    if int(norm_mean.shape[0]) != int(arch["input_dim"]):
        raise ValueError(
            f"Feature dim mismatch: norm={norm_mean.shape[0]} vs model input_dim={arch['input_dim']}"
        )

    audio, sr = librosa.load(str(wav_path), sr=int(args.sample_rate), mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio = trim_audio(audio, sample_rate=int(args.sample_rate), max_sec=float(args.max_audio_sec))

    processor = load_ctc_processor(args.ctc_model)
    ctc_model = Wav2Vec2ForCTC.from_pretrained(args.ctc_model).to(device)
    ctc_model.eval()

    blank_id = processor.tokenizer.pad_token_id
    unk_id = processor.tokenizer.unk_token_id

    transcript_encoding_mode = "phonemizer_espeak"
    try:
        backend = build_phonemizer_backend()
        word_phones = phonemize_words(args.text, backend)
        token_ids, phone_symbols = encode_word_phones(word_phones, processor.tokenizer, unk_id)
    except Exception as phonemizer_error:
        if not args.allow_text_tokenizer_fallback:
            raise
        print("Warning: phonemizer unavailable, switching to tokenizer-text fallback")
        print("Reason:", repr(phonemizer_error))
        token_ids, phone_symbols = encode_text_fallback(args.text, processor.tokenizer, unk_id, blank_id)
        transcript_encoding_mode = "tokenizer_text_fallback"

    if len(token_ids) == 0:
        raise RuntimeError("No valid phone tokens extracted from transcript. Check transcript language/content.")

    x = processor(audio, sampling_rate=int(args.sample_rate), return_tensors="pt")
    input_values = x.input_values.to(device)
    attention_mask = x.attention_mask.to(device) if "attention_mask" in x else None
    with torch.no_grad():
        out = ctc_model(input_values, attention_mask=attention_mask)
    logits = out.logits[0].detach().float().cpu()

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    confidence = probs.max(dim=-1).values
    hidden = logits[:, : min(128, logits.size(1))]

    change_times = get_change_times(log_probs, token_ids, blank_id)
    bounds = build_bounds(change_times, log_probs.size(0))

    if variant in ("energy", "pitch", "energy_pitch"):
        prosody_tracks = compute_prosody_tracks(
            audio,
            n_frames=log_probs.size(0),
            sample_rate=args.sample_rate,
            fmin=args.prosody_fmin,
            fmax=args.prosody_fmax,
        )
    else:
        prosody_tracks = (None, None, None)

    feats, used_token_ids, used_phone_symbols = build_token_features(
        log_probs=log_probs,
        hidden=hidden,
        entropy=entropy,
        confidence=confidence,
        token_ids=token_ids,
        phone_symbols=phone_symbols,
        bounds=bounds,
        confusion_graph=confusion_graph,
        blank_id=blank_id,
        variant=variant,
        proj_dim=args.proj_dim,
        confusion_topk=args.confusion_topk,
        lambda_entropy=args.lambda_entropy,
        prosody_tracks=prosody_tracks,
    )

    max_phones = int(arch["max_seq_len"])
    L = min(max_phones, len(feats), len(used_token_ids), len(used_phone_symbols))
    if L <= 0:
        raise RuntimeError("No token-level features built from this wav/text pair.")

    feat_arr = np.zeros((max_phones, int(arch["input_dim"])), dtype=np.float32)
    phn_ids = np.full((max_phones,), -1.0, dtype=np.float32)

    feat_stack = np.asarray(feats[:L], dtype=np.float32)
    feat_stack = (feat_stack - norm_mean) / norm_std
    feat_arr[:L] = feat_stack

    mapped_phone_symbols = []
    max_phone_id = int(arch.get("phn_vocab", 39) - 1)
    for i in range(L):
        tok_id = int(used_token_ids[i])
        tok_sym = processor.tokenizer.convert_ids_to_tokens(tok_id)
        tok_sym = str(tok_sym) if tok_sym is not None else ""

        pid = int(phone_to_idx.get(tok_sym, unk_phone_idx))
        pid = max(0, min(max_phone_id, pid))
        phn_ids[i] = float(pid)

        if used_phone_symbols[i]:
            mapped_phone_symbols.append(str(used_phone_symbols[i]))
        else:
            mapped_phone_symbols.append(tok_sym)

    feat_tensor = torch.tensor(feat_arr, dtype=torch.float32, device=device).unsqueeze(0)
    phn_tensor = torch.tensor(phn_ids, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        u1, u2, u3, u4, u5, p, w1, w2, w3 = model(feat_tensor, phn_tensor)

    utt_pred = torch.cat((u1, u2, u3, u4, u5), dim=1).squeeze(0).cpu().numpy()
    phn_pred = p.squeeze(0).squeeze(-1).cpu().numpy()
    word_pred = torch.cat((w1, w2, w3), dim=2).squeeze(0).cpu().numpy()

    valid = phn_ids >= 0
    valid_count = int(np.sum(valid))
    if valid_count > 0:
        word_mean = np.mean(word_pred[valid], axis=0)
    else:
        word_mean = np.zeros((3,), dtype=np.float32)

    phone_rows = []
    valid_pos = np.where(valid)[0].tolist()
    for i, pos in enumerate(valid_pos):
        pred_val = float(phn_pred[pos])
        clipped = float(np.clip(pred_val, 0.0, 2.0))
        phone_rows.append(
            {
                "position": int(pos),
                "phone": str(mapped_phone_symbols[i]) if i < len(mapped_phone_symbols) else "",
                "pred_phn_norm": pred_val,
                "pred_phn_clipped_0_2": clipped,
                "pred_phn_0_5": float(clipped * 2.5),
            }
        )

    report = {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "wav": str(wav_path),
        "sample_rate": int(sr),
        "text": str(args.text),
        "transcript_encoding_mode": transcript_encoding_mode,
        "variant": str(variant),
        "architecture": arch,
        "token_count": valid_count,
        "scores_normalized": {
            "utt_accuracy": float(utt_pred[0]),
            "utt_completeness": float(utt_pred[1]),
            "utt_fluency": float(utt_pred[2]),
            "utt_prosodic": float(utt_pred[3]),
            "utt_total": float(utt_pred[4]),
            "word_accuracy_mean": float(word_mean[0]),
            "word_stress_mean": float(word_mean[1]),
            "word_total_mean": float(word_mean[2]),
        },
        "scores_0_5": {
            "utt_accuracy": score_to_5(utt_pred[0]),
            "utt_completeness": score_to_5(utt_pred[1]),
            "utt_fluency": score_to_5(utt_pred[2]),
            "utt_prosodic": score_to_5(utt_pred[3]),
            "utt_total": score_to_5(utt_pred[4]),
            "word_accuracy_mean": score_to_5(word_mean[0]),
            "word_stress_mean": score_to_5(word_mean[1]),
            "word_total_mean": score_to_5(word_mean[2]),
        },
        "outputs": {
            "json": str(output_json),
            "csv": str(output_csv),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    save_phone_rows_csv(phone_rows, output_csv)

    print("Single-file scoring finished")
    print("checkpoint:", checkpoint)
    print("wav:", wav_path)
    print("text:", args.text)
    print("variant:", variant)
    print("token_count:", valid_count)
    print("utt_total (0-5):", report["scores_0_5"]["utt_total"])
    print("word_stress_mean (0-5):", report["scores_0_5"]["word_stress_mean"])
    print("word_total_mean (0-5):", report["scores_0_5"]["word_total_mean"])
    print("report json:", output_json)
    print("phone csv:", output_csv)


if __name__ == "__main__":
    main()
