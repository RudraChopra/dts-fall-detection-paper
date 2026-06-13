#!/usr/bin/env python3
"""Run a strict twin-free FallVision benchmark for the DTS paper.

This script is intentionally self-contained so the paper can cite one command
that regenerates the key artifacts reviewers will ask for:

* parsed clip manifest
* deduplication/twin audit
* strict unique-representative train/val/test split
* per-model validation and test score vectors
* bootstrap confidence intervals
* matched operating-point analysis

The protocol is stricter than the current manuscript's main table. It groups
clips by a feature fingerprint, drops ambiguous groups whose identical features
carry conflicting labels, keeps one representative per remaining group, and
then stratifies train/validation/test on those representatives. This guarantees
no feature-identical twin can appear in different splits, and no duplicate can
inflate the test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from scipy.stats import ttest_ind
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "work" / "revised_paper" / "data" / "fallvision_extracted"
DEFAULT_OUT = ROOT / "outputs" / "ninefive"

EPS = 1e-8
MAX_T = 150
SEED = 20260610

COCO_NAMES = {
    "nose": 0,
    "left eye": 1,
    "right eye": 2,
    "left ear": 3,
    "right ear": 4,
    "left shoulder": 5,
    "right shoulder": 6,
    "left elbow": 7,
    "right elbow": 8,
    "left wrist": 9,
    "right wrist": 10,
    "left hip": 11,
    "right hip": 12,
    "left knee": 13,
    "right knee": 14,
    "left ankle": 15,
    "right ankle": 16,
}


@dataclass
class Clip:
    clip_id: str
    path: str
    source_folder: str
    filename: str
    label: int
    scenario: str
    session_id: str
    seq: np.ndarray
    conf: np.ndarray
    fixed_seq: np.ndarray
    feature30: np.ndarray
    raw_hash: str
    seq_hash: str
    feature_hash: str
    n_frames: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, (os_cpu_count() or 4) - 1)))


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def scenario_from_folder(name: str) -> str:
    low = name.lower()
    if "_b_" in low:
        return "Bed"
    if "_c_" in low:
        return "Chair"
    if "_s_" in low:
        return "Stand"
    return "Unknown"


def label_from_folder(name: str) -> int:
    low = name.lower()
    return 0 if low.startswith("nf_") or "/nf_" in low else 1


def session_from_folder(name: str) -> str:
    parts = name.lower().split("_")
    for p in reversed(parts):
        if p.isdigit():
            return p
    return "unknown"


def parse_keypoint_value(value: str) -> int | None:
    txt = str(value).strip()
    try:
        k = int(float(txt))
        return k if 0 <= k < 17 else None
    except Exception:
        pass
    key = txt.lower().replace("_", " ")
    return COCO_NAMES.get(key)


def parse_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with path.open("r", newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None
    if not rows:
        return None

    cols = {c.strip().lower(): c for c in rows[0].keys() if c is not None}
    required = ["frame", "keypoint", "x", "y"]
    if not all(k in cols for k in required):
        return None
    conf_col = None
    for k, c in cols.items():
        if "conf" in k:
            conf_col = c
            break
    if conf_col is None:
        return None

    frames: dict[int, dict[str, np.ndarray]] = {}
    for row in rows:
        try:
            frame = int(float(row[cols["frame"]]))
            kp = parse_keypoint_value(row[cols["keypoint"]])
            if kp is None:
                continue
            x = float(row[cols["x"]])
            y = float(row[cols["y"]])
            cf = float(row[conf_col])
        except Exception:
            continue
        slot = frames.setdefault(
            frame,
            {
                "xy": np.zeros((17, 2), dtype=np.float64),
                "cf": np.zeros(17, dtype=np.float64),
                "cnt": np.zeros(17, dtype=np.float64),
            },
        )
        slot["xy"][kp] += (x, y)
        slot["cf"][kp] = max(slot["cf"][kp], cf)
        slot["cnt"][kp] += 1.0

    if not frames:
        return None
    ordered = sorted(frames)
    seq = np.zeros((len(ordered), 17, 2), dtype=np.float32)
    conf = np.zeros((len(ordered), 17), dtype=np.float32)
    for i, fr in enumerate(ordered):
        cnt = np.maximum(frames[fr]["cnt"], 1.0)[:, None]
        seq[i] = frames[fr]["xy"] / cnt
        conf[i] = frames[fr]["cf"]
    return seq, conf


def normalise(seq: np.ndarray) -> tuple[np.ndarray, float]:
    seq = np.asarray(seq, dtype=np.float64)
    hip = (seq[:, 11, :] + seq[:, 12, :]) / 2.0
    torso = np.linalg.norm(seq[:, 5, :] - hip, axis=1)
    scale = float(np.median(torso[torso > 0])) if np.any(torso > 0) else float(np.median(torso))
    scale = scale + EPS
    return (seq - hip[:, None, :]) / scale, scale


def temporal_op16(z: np.ndarray, tau: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if len(z) < 3:
        return np.zeros(16, dtype=np.float64)
    dz = np.zeros_like(z)
    dz[1:] = z[1:] - z[:-1]
    d2z = np.zeros_like(z)
    d2z[1:] = dz[1:] - dz[:-1]
    split = max(1, int(tau * len(z)))
    denom = np.abs(dz[1:]).sum() + EPS
    alpha = float(np.abs(dz[1 : split + 1]).sum() / denom)
    slope = float(np.polyfit(np.arange(len(z), dtype=float), z, 1)[0])
    return np.array(
        [
            z.mean(),
            z.std() + EPS,
            z.min(),
            z.max(),
            np.percentile(z, 25),
            np.percentile(z, 50),
            np.percentile(z, 75),
            z.max() - z.min(),
            z[0],
            z[-1],
            z[-1] - z[0],
            slope,
            np.abs(dz).mean(),
            np.abs(dz).max(),
            np.abs(d2z).max(),
            alpha,
        ],
        dtype=np.float64,
    )


def dts128(seq: np.ndarray, conf: np.ndarray | None = None, tau: float = 0.30) -> np.ndarray:
    n, scale = normalise(seq)
    hip_raw = (seq[:, 11, :] + seq[:, 12, :]) / 2.0
    hip_norm = hip_raw / scale
    bbox_w = n[:, :, 0].max(axis=1) - n[:, :, 0].min(axis=1)
    hip_y = n[:, 11, 1]
    shoulder_mid = (n[:, 5, :] + n[:, 6, :]) / 2.0
    torso_vec = n[:, 0, :] - shoulder_mid
    torso_angle = np.arctan2(torso_vec[:, 0], torso_vec[:, 1] + EPS)
    hip_delta = np.zeros_like(hip_norm)
    hip_delta[1:] = hip_norm[1:] - hip_norm[:-1]
    hip_speed = np.linalg.norm(hip_delta, axis=1)
    centre = n.mean(axis=1)
    centre_delta = np.zeros_like(centre)
    centre_delta[1:] = centre[1:] - centre[:-1]
    centre_speed = np.linalg.norm(centre_delta, axis=1)
    hip_acc = np.zeros(len(seq))
    hip_acc[1:] = hip_speed[1:] - hip_speed[:-1]
    shoulder_y = shoulder_mid[:, 1]
    head_y = n[:, 0, 1]
    primitives = [bbox_w, hip_y, torso_angle, hip_speed, centre_speed, hip_acc, shoulder_y, head_y]
    return np.concatenate([temporal_op16(p, tau) for p in primitives])


def fixed_sequence(seq: np.ndarray, conf: np.ndarray, max_t: int = MAX_T) -> np.ndarray:
    n, _ = normalise(seq)
    n = n.astype(np.float32)
    cf = np.asarray(conf, dtype=np.float32)
    n[cf < 0.10] = 0.0
    flat = n.reshape(len(n), -1)
    if len(flat) >= max_t:
        return flat[:max_t]
    out = np.zeros((max_t, 34), dtype=np.float32)
    out[: len(flat)] = flat
    return out


def sha_array(arr: np.ndarray, decimals: int = 6) -> str:
    a = np.round(np.asarray(arr, dtype=np.float64), decimals=decimals)
    return hashlib.sha256(a.tobytes()).hexdigest()[:20]


def load_clips(data_dir: Path, min_frames: int) -> tuple[list[Clip], list[dict]]:
    clips: list[Clip] = []
    failures: list[dict] = []
    paths = sorted(data_dir.glob("*/*.csv"))
    started = time.time()
    print(f"Parsing {len(paths)} CSV files from {data_dir}", flush=True)
    for file_i, path in enumerate(paths, 1):
        if file_i == 1 or file_i % 250 == 0:
            print(f"  parsed {file_i - 1}/{len(paths)} files, kept {len(clips)}, failed {len(failures)} ({time.time() - started:.1f}s)", flush=True)
        folder = path.parent.name
        scenario = scenario_from_folder(folder)
        label = label_from_folder(folder)
        session = session_from_folder(folder)
        parsed = parse_csv(path)
        if parsed is None:
            failures.append({"path": str(path), "reason": "parse_failed"})
            continue
        seq, conf = parsed
        if seq.shape[0] < min_frames:
            failures.append({"path": str(path), "reason": f"short_{seq.shape[0]}"})
            continue
        fixed = fixed_sequence(seq, conf)
        # raw_hash is a stable provenance id, not used for deduplication. Avoid
        # reading the large CSV a second time; seq_hash/feature_hash below are
        # the deduplication fingerprints.
        raw_hash = hashlib.sha256(f"{path}|{seq.shape}|{conf.shape}".encode("utf-8")).hexdigest()[:20]
        seq_hash = sha_array(fixed, decimals=5)
        feat30 = dts128(seq, conf, tau=0.30)
        feature_hash = sha_array(feat30, decimals=7)
        clips.append(
            Clip(
                clip_id=f"clip_{len(clips):05d}",
                path=str(path),
                source_folder=folder,
                filename=path.name,
                label=int(label),
                scenario=scenario,
                session_id=session,
                seq=seq,
                conf=conf,
                fixed_seq=fixed,
                feature30=feat30,
                raw_hash=raw_hash,
                seq_hash=seq_hash,
                feature_hash=feature_hash,
                n_frames=int(seq.shape[0]),
            )
        )
    print(f"Finished parsing: kept {len(clips)}, failed {len(failures)} in {time.time() - started:.1f}s", flush=True)
    return clips, failures


def save_clip_cache(path: Path, clips: list[Clip], failures: list[dict], data_dir: Path, min_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fixed_seq=np.stack([c.fixed_seq for c in clips]).astype(np.float32),
        feature30=np.stack([c.feature30 for c in clips]).astype(np.float32),
        clip_id=np.array([c.clip_id for c in clips], dtype=object),
        path=np.array([c.path for c in clips], dtype=object),
        source_folder=np.array([c.source_folder for c in clips], dtype=object),
        filename=np.array([c.filename for c in clips], dtype=object),
        label=np.array([c.label for c in clips], dtype=np.int8),
        scenario=np.array([c.scenario for c in clips], dtype=object),
        session_id=np.array([c.session_id for c in clips], dtype=object),
        raw_hash=np.array([c.raw_hash for c in clips], dtype=object),
        seq_hash=np.array([c.seq_hash for c in clips], dtype=object),
        feature_hash=np.array([c.feature_hash for c in clips], dtype=object),
        n_frames=np.array([c.n_frames for c in clips], dtype=np.int32),
        failures_json=np.array([json.dumps(failures)], dtype=object),
        meta_json=np.array([json.dumps({"data_dir": str(data_dir), "min_frames": min_frames, "tau": 0.30})], dtype=object),
    )
    print(f"Saved parsed feature cache to {path}", flush=True)


def load_clip_cache(path: Path) -> tuple[list[Clip], list[dict]]:
    print(f"Loading parsed feature cache from {path}", flush=True)
    z = np.load(path, allow_pickle=True)
    failures = json.loads(str(z["failures_json"][0]))
    fixed_seq = z["fixed_seq"].astype(np.float32)
    feature30 = z["feature30"].astype(np.float64)
    clip_ids = z["clip_id"]
    paths = z["path"]
    source_folders = z["source_folder"]
    filenames = z["filename"]
    labels = z["label"]
    scenarios = z["scenario"]
    session_ids = z["session_id"]
    raw_hashes = z["raw_hash"]
    seq_hashes = z["seq_hash"]
    feature_hashes = z["feature_hash"]
    n_frames = z["n_frames"]
    clips: list[Clip] = []
    empty_seq = np.zeros((0, 17, 2), dtype=np.float32)
    empty_conf = np.zeros((0, 17), dtype=np.float32)
    for i in range(len(labels)):
        clips.append(
            Clip(
                clip_id=str(clip_ids[i]),
                path=str(paths[i]),
                source_folder=str(source_folders[i]),
                filename=str(filenames[i]),
                label=int(labels[i]),
                scenario=str(scenarios[i]),
                session_id=str(session_ids[i]),
                seq=empty_seq,
                conf=empty_conf,
                fixed_seq=fixed_seq[i],
                feature30=feature30[i],
                raw_hash=str(raw_hashes[i]),
                seq_hash=str(seq_hashes[i]),
                feature_hash=str(feature_hashes[i]),
                n_frames=int(n_frames[i]),
            )
        )
    return clips, failures


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_twinfree_representatives(clips: list[Clip]) -> tuple[list[Clip], list[dict], dict]:
    groups: dict[str, list[Clip]] = {}
    for c in clips:
        # Feature hash is intentionally primary: if two samples are identical
        # to the representation being tested, they must not straddle splits.
        groups.setdefault(c.feature_hash, []).append(c)

    reps: list[Clip] = []
    audit: list[dict] = []
    ambiguous = 0
    dropped_clips = 0
    for gid, members in sorted(groups.items()):
        labels = sorted({m.label for m in members})
        scenarios = sorted({m.scenario for m in members})
        sessions = sorted({m.session_id for m in members})
        row = {
            "feature_hash": gid,
            "n_members": len(members),
            "labels": ";".join(map(str, labels)),
            "scenarios": ";".join(scenarios),
            "sessions": ";".join(sessions),
            "representative": members[0].clip_id,
            "member_ids": ";".join(m.clip_id for m in members),
            "member_paths": ";".join(m.path for m in members),
        }
        if len(labels) > 1:
            row["status"] = "drop_ambiguous_conflicting_labels"
            ambiguous += 1
            dropped_clips += len(members)
        else:
            row["status"] = "keep_one_representative"
            reps.append(members[0])
        audit.append(row)
    meta = {
        "n_input_clips": len(clips),
        "n_feature_groups": len(groups),
        "n_kept_representatives": len(reps),
        "n_ambiguous_groups_dropped": ambiguous,
        "n_clips_in_ambiguous_groups_dropped": dropped_clips,
        "n_duplicate_members_removed": len(clips) - len(reps) - dropped_clips,
    }
    return reps, audit, meta


def stratified_split(clips: list[Clip], seed: int) -> dict[str, list[int]]:
    y = np.array([c.label for c in clips])
    idx = np.arange(len(clips))
    train_val, test = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y)
    y_tv = y[train_val]
    train, val = train_test_split(train_val, test_size=0.20, random_state=seed, stratify=y_tv)
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


def find_tau_star(features_by_tau: dict[float, np.ndarray], y_train: np.ndarray, train_idx: list[int]) -> float:
    best_tau = 0.30
    best_score = -np.inf
    yt = y_train
    for tau, feats_all in features_by_tau.items():
        feats = feats_all[train_idx]
        alpha_cols = [i * 16 + 15 for i in range(8)]
        score = 0.0
        for col in alpha_cols:
            a = feats[yt == 1, col]
            b = feats[yt == 0, col]
            if len(a) > 2 and len(b) > 2:
                stat = ttest_ind(a, b, equal_var=False, nan_policy="omit").statistic
                if np.isfinite(stat):
                    score += abs(float(stat))
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau


def metric_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y, scores)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def select_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    best_thr = 0.5
    best = (-1.0, -1.0)
    for thr in np.linspace(0.01, 0.99, 99):
        m = metric_at_threshold(y_val, scores_val, float(thr))
        key = (m["f1"], m["auroc"])
        if key > best:
            best = key
            best_thr = float(thr)
    return best_thr


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, thr: float, n_boot: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = {k: [] for k in ["auroc", "auprc", "f1", "precision", "recall", "accuracy"]}
    for _ in range(n_boot):
        ii = rng.integers(0, n, size=n)
        yy = y[ii]
        ss = scores[ii]
        if len(np.unique(yy)) < 2:
            continue
        m = metric_at_threshold(yy, ss, thr)
        for k in vals:
            vals[k].append(m[k])
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in vals.items() if v}


def evaluate_scores(y_val: np.ndarray, val_scores: np.ndarray, y_test: np.ndarray, test_scores: np.ndarray, n_boot: int, seed: int) -> dict:
    thr = select_threshold(y_val, val_scores)
    out = metric_at_threshold(y_test, test_scores, thr)
    out["threshold_source"] = "validation_f1_grid"
    out["ci95"] = bootstrap_ci(y_test, test_scores, thr, n_boot=n_boot, seed=seed)
    return out


def fit_predict_sklearn(model_name: str, estimator, X_train, y_train, X_val, X_test) -> tuple[np.ndarray, np.ndarray, object]:
    estimator.fit(X_train, y_train)
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_val)[:, 1], estimator.predict_proba(X_test)[:, 1], estimator
    if hasattr(estimator, "decision_function"):
        v = estimator.decision_function(X_val)
        t = estimator.decision_function(X_test)
        return 1 / (1 + np.exp(-v)), 1 / (1 + np.exp(-t)), estimator
    raise ValueError(f"{model_name} has no probability-like output")


def validation_grid(model_name: str, grid: list[dict], factory, X_train, y_train, X_val, y_val) -> tuple[dict, object]:
    print(f"Selecting {model_name} over {len(grid)} validation configs", flush=True)
    best_key = (-np.inf, -np.inf)
    best_params: dict | None = None
    best_model = None
    for i, params in enumerate(grid, 1):
        if i == 1 or i == len(grid) or i % 5 == 0:
            print(f"  {model_name}: config {i}/{len(grid)} {params}", flush=True)
        model = factory(params)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        key = (float(roc_auc_score(y_val, scores)), float(f1_score(y_val, scores >= select_threshold(y_val, scores), zero_division=0)))
        if key > best_key:
            best_key = key
            best_params = params
            best_model = model
    assert best_params is not None and best_model is not None
    return best_params, best_model


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        lengths = []
        seqs = []
        for row in X:
            nz = np.where(np.abs(row).sum(axis=1) > 0)[0]
            length = int(nz[-1] + 1) if len(nz) else 1
            lengths.append(length)
            seqs.append(torch.tensor(row[:length], dtype=torch.float32))
        self.seqs = seqs
        self.y = torch.tensor(y.astype(np.float32))
        self.lengths = lengths

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.seqs[i], self.y[i], self.lengths[i]


def collate_seq(batch):
    seqs, y, lengths = zip(*batch)
    return pad_sequence(seqs, batch_first=True), torch.stack(y), torch.tensor(lengths, dtype=torch.long)


class FallRNN(nn.Module):
    def __init__(self, kind: str, input_dim: int = 34, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.kind = kind
        self.rnn = cls(input_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if self.kind == "lstm":
            _, (h, _) = self.rnn(packed)
        else:
            _, h = self.rnn(packed)
        return self.fc(h[-1]).squeeze(-1)


class FallTransformer(nn.Module):
    def __init__(self, input_dim: int = 34, d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x, lengths):
        b, t, _ = x.shape
        h = self.proj(x)
        pos = torch.arange(t, device=x.device, dtype=torch.float32)[None, :, None] / 100.0
        h = h + pos
        mask = torch.arange(t, device=x.device)[None, :] >= lengths[:, None].to(x.device)
        out = self.enc(h, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (out * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.fc(pooled).squeeze(-1)


class DTSNet(nn.Module):
    def __init__(self, dropout: float = 0.15, lam_aux: float = 0.15):
        super().__init__()
        self.alpha_cols = torch.tensor([i * 16 + 15 for i in range(8)], dtype=torch.long)
        self.attn = nn.Linear(8, 8)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.aux = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.lam_aux = lam_aux

    def forward(self, x):
        av = x[:, self.alpha_cols.to(x.device)]
        att = torch.softmax(self.attn(av), dim=-1)
        xw = (x.view(-1, 8, 16) * att.unsqueeze(-1)).view(-1, 128)
        h1 = self.dropout(F.relu(self.fc1(xw)))
        h2 = F.relu(self.fc2(h1))
        return self.out(h2).squeeze(-1), self.aux(h1).squeeze(-1), av

    def loss(self, x, y):
        logit, aux, av = self.forward(x)
        return F.binary_cross_entropy_with_logits(logit, y) + self.lam_aux * F.mse_loss(aux, av.mean(dim=1))


def coco_adjacency() -> np.ndarray:
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    a = np.eye(17, dtype=np.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    deg = a.sum(axis=1)
    d_inv = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-6)))
    return d_inv @ a @ d_inv


class GraphTemporalBlock(nn.Module):
    def __init__(self, cin: int, cout: int, adj: torch.Tensor, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = nn.Conv2d(cin, cout, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(cout),
            nn.Dropout(dropout),
        )
        if cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(cout))

    def forward(self, x):
        # x: B,C,T,V. Apply graph multiplication over V.
        xg = torch.einsum("bctv,vw->bctw", x, self.adj)
        return F.relu(self.tcn(self.gcn(xg)) + self.res(x))


class STGCN(nn.Module):
    def __init__(self, channels: list[int], dropout: float = 0.25):
        super().__init__()
        adj = torch.tensor(coco_adjacency(), dtype=torch.float32)
        blocks = []
        cin = 2
        for i, cout in enumerate(channels):
            stride = 2 if i in {3, 5} and len(channels) > 4 else 1
            blocks.append(GraphTemporalBlock(cin, cout, adj, stride=stride, dropout=dropout))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, 1)

    def forward(self, x, lengths=None):
        # x: B,T,34 -> B,2,T,17
        b, t, _ = x.shape
        x = x.view(b, t, 17, 2).permute(0, 3, 1, 2).contiguous()
        h = self.blocks(x)
        h = h.mean(dim=(2, 3))
        return self.fc(h).squeeze(-1)


def train_torch_model(
    name: str,
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    is_dts: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    print(f"Training {name}: epochs={epochs}, lr={lr}, batch={batch_size}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and (is_dts or "ST-GCN" in name):
        # Packed RNNs and Transformer masking are more predictable on CPU, but
        # the graph convolution baselines and DTS-Net benefit substantially
        # from Apple Silicon's MPS backend.
        device = torch.device("mps")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc = -np.inf
    best_val = None
    best_test = None

    if is_dts:
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.astype(np.float32)))
        val_x = torch.tensor(X_val, dtype=torch.float32)
        test_x = torch.tensor(X_test, dtype=torch.float32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                if isinstance(model, DTSNet):
                    loss = model.loss(xb, yb)
                else:
                    loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                if isinstance(model, DTSNet):
                    pv = torch.sigmoid(model(val_x.to(device))[0]).cpu().numpy()
                    pt = torch.sigmoid(model(test_x.to(device))[0]).cpu().numpy()
                else:
                    pv = torch.sigmoid(model(val_x.to(device))).cpu().numpy()
                    pt = torch.sigmoid(model(test_x.to(device))).cpu().numpy()
            auc = roc_auc_score(y_val, pv)
            if auc > best_auc:
                best_auc, best_val, best_test = auc, pv.copy(), pt.copy()
            if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  {name} epoch {epoch + 1}/{epochs}: val AUROC={auc:.4f} best={best_auc:.4f}", flush=True)
    else:
        train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True, collate_fn=collate_seq)
        val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=batch_size * 2, shuffle=False, collate_fn=collate_seq)
        test_loader = DataLoader(SeqDataset(X_test, np.zeros(len(X_test))), batch_size=batch_size * 2, shuffle=False, collate_fn=collate_seq)
        loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            model.train()
            for xb, yb, lengths in train_loader:
                xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
                loss = loss_fn(model(xb, lengths), yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = np.concatenate([torch.sigmoid(model(xb.to(device), lengths.to(device))).cpu().numpy() for xb, _, lengths in val_loader])
                pt = np.concatenate([torch.sigmoid(model(xb.to(device), lengths.to(device))).cpu().numpy() for xb, _, lengths in test_loader])
            auc = roc_auc_score(y_val, pv)
            if auc > best_auc:
                best_auc, best_val, best_test = auc, pv.copy(), pt.copy()
            if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  {name} epoch {epoch + 1}/{epochs}: val AUROC={auc:.4f} best={best_auc:.4f}", flush=True)

    assert best_val is not None and best_test is not None
    return best_val, best_test, {"epochs": epochs, "lr": lr, "batch_size": batch_size, "best_val_auroc": float(best_auc)}


def threshold_for_max_fp(y: np.ndarray, scores: np.ndarray, max_fp: int) -> tuple[float, dict]:
    best = None
    # Include thresholds above max(score) so "predict no positives" is always
    # available when a very low FP cap is requested.
    thresholds = [0.0, 1.0, float(np.max(scores)) + 1e-6]
    thresholds.extend(float(x) for x in np.asarray(scores, dtype=np.float64))
    for thr in sorted(set(thresholds)):
        m = metric_at_threshold(y, scores, float(thr))
        if m["fp"] <= max_fp:
            if best is None or m["recall"] > best[1]["recall"] or (m["recall"] == best[1]["recall"] and m["fp"] > best[1]["fp"]):
                best = (float(thr), m)
    assert best is not None
    return best


def threshold_for_min_recall(y: np.ndarray, scores: np.ndarray, target_recall: float) -> tuple[float, dict]:
    best = None
    thresholds = [0.0, 1.0, float(np.max(scores)) + 1e-6]
    thresholds.extend(float(x) for x in np.asarray(scores, dtype=np.float64))
    for thr in sorted(set(thresholds), reverse=True):
        m = metric_at_threshold(y, scores, float(thr))
        if m["recall"] >= target_recall:
            if best is None or m["fp"] < best[1]["fp"] or (m["fp"] == best[1]["fp"] and m["precision"] > best[1]["precision"]):
                best = (float(thr), m)
    assert best is not None
    return best


def matched_operating_analysis(y: np.ndarray, scores: dict[str, np.ndarray], primary: str, comparator: str) -> dict:
    p_default = scores[primary + "__default_metrics"]
    c_default = scores[comparator + "__default_metrics"]
    primary_scores = scores[primary]
    comp_scores = scores[comparator]
    _, p_at_comp_fp = threshold_for_max_fp(y, primary_scores, c_default["fp"])
    _, c_at_primary_fp = threshold_for_max_fp(y, comp_scores, p_default["fp"])
    _, p_at_95 = threshold_for_min_recall(y, primary_scores, 0.95)
    _, c_at_95 = threshold_for_min_recall(y, comp_scores, 0.95)
    _, p_at_comp_rec = threshold_for_min_recall(y, primary_scores, c_default["recall"])
    return {
        "primary": primary,
        "comparator": comparator,
        "default_primary": p_default,
        "default_comparator": c_default,
        "primary_at_comparator_fp": p_at_comp_fp,
        "comparator_at_primary_fp": c_at_primary_fp,
        "primary_at_recall_0.95": p_at_95,
        "comparator_at_recall_0.95": c_at_95,
        "primary_at_comparator_default_recall": p_at_comp_rec,
    }


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_npz)
    if cache_path.exists() and not args.tune_tau and abs(float(args.tau) - 0.30) < 1e-9:
        clips, failures = load_clip_cache(cache_path)
    else:
        clips, failures = load_clips(Path(args.data_dir), min_frames=args.min_frames)
        if not args.tune_tau and abs(float(args.tau) - 0.30) < 1e-9:
            save_clip_cache(cache_path, clips, failures, Path(args.data_dir), args.min_frames)
    reps, dedup_audit, dedup_meta = make_twinfree_representatives(clips)
    split = stratified_split(reps, args.seed)

    manifest_rows = []
    for i, c in enumerate(reps):
        split_name = next((k for k, vals in split.items() if i in set(vals)), "unknown")
        manifest_rows.append(
            {
                "row_index": i,
                "clip_id": c.clip_id,
                "split": split_name,
                "label": c.label,
                "scenario": c.scenario,
                "session_id": c.session_id,
                "n_frames": c.n_frames,
                "feature_hash": c.feature_hash,
                "seq_hash": c.seq_hash,
                "raw_hash": c.raw_hash,
                "source_folder": c.source_folder,
                "filename": c.filename,
                "path": c.path,
            }
        )
    write_csv(out / "split_manifest_twinfree.csv", manifest_rows)
    write_csv(out / "dedup_audit.csv", dedup_audit)
    write_csv(out / "parse_failures.csv", failures)

    y = np.array([c.label for c in reps], dtype=int)
    fixed = np.stack([c.fixed_seq for c in reps])
    if args.tune_tau:
        print("Extracting DTS features over tau grid for train-only tau selection", flush=True)
        tau_grid = np.round(np.linspace(0.20, 0.80, 13), 2)
        features_by_tau = {float(tau): np.stack([dts128(c.seq, c.conf, tau=float(tau)) for c in reps]) for tau in tau_grid}
        tau_star = find_tau_star(features_by_tau, y[split["train"]], split["train"])
        X = features_by_tau[float(tau_star)]
    else:
        tau_star = float(args.tau)
        if abs(tau_star - 0.30) < 1e-9:
            X = np.stack([c.feature30 for c in reps])
        else:
            print(f"Extracting DTS features at fixed tau={tau_star}", flush=True)
            X = np.stack([dts128(c.seq, c.conf, tau=tau_star) for c in reps])

    X_train, X_val, X_test = X[split["train"]], X[split["val"]], X[split["test"]]
    y_train, y_val, y_test = y[split["train"]], y[split["val"]], y[split["test"]]
    S_train, S_val, S_test = fixed[split["train"]], fixed[split["val"]], fixed[split["test"]]

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val = scaler.transform(X_val)
    Xs_test = scaler.transform(X_test)

    results: dict[str, dict] = {}
    score_vectors: dict[str, np.ndarray] = {"y_val": y_val, "y_test": y_test}
    selected: dict[str, dict] = {}

    models_requested = set(args.models.split(","))

    def persist(status: str) -> None:
        np.savez_compressed(out / "score_vectors_twinfree.npz", **score_vectors)
        partial_summary = {
            "protocol": "strict_unique_feature_group_twinfree",
            "status": status,
            "seed": args.seed,
            "data_dir": str(args.data_dir),
            "min_frames": args.min_frames,
            "dedup": dedup_meta,
            "split": {
                k: {
                    "n": len(v),
                    "falls": int(y[v].sum()),
                    "nonfalls": int(len(v) - y[v].sum()),
                }
                for k, v in split.items()
            },
            "tau_star": tau_star,
            "models": results,
            "selected_configs": selected,
            "artifacts": {
                "split_manifest": "split_manifest_twinfree.csv",
                "dedup_audit": "dedup_audit.csv",
                "parse_failures": "parse_failures.csv",
                "score_vectors": "score_vectors_twinfree.npz",
            },
            "command": "python work/ninefive/run_twinfree_benchmark.py --models all",
            "elapsed_s": round(time.time() - t0, 1),
        }
        (out / "twinfree_results.json").write_text(json.dumps(partial_summary, indent=2))

    def add_result(name: str, val_scores: np.ndarray, test_scores: np.ndarray, config: dict):
        metrics = evaluate_scores(y_val, val_scores, y_test, test_scores, n_boot=args.bootstraps, seed=args.seed + len(results))
        results[name] = metrics
        selected[name] = config
        score_vectors[f"{name}_val"] = val_scores.astype(np.float32)
        score_vectors[f"{name}_test"] = test_scores.astype(np.float32)
        persist("partial")
        print(f"{name:18s} AUROC={metrics['auroc']:.4f} F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} FP={metrics['fp']} FN={metrics['fn']}")

    if "dts" in models_requested or "all" in models_requested:
        if args.quick:
            grids = {
                "DTS+LR": list(ParameterGrid({"C": [0.5]})),
                "DTS+ET": list(ParameterGrid({"n_estimators": [200], "max_features": ["sqrt"]})),
                "DTS+RF": list(ParameterGrid({"n_estimators": [200], "max_features": ["sqrt"]})),
                "DTS+HGB": list(ParameterGrid({"max_iter": [300], "learning_rate": [0.1], "l2_regularization": [0.0]})),
            }
        else:
            grids = {
                "DTS+LR": list(ParameterGrid({"C": [0.05, 0.1, 0.5, 1, 2, 10]})),
                "DTS+ET": list(ParameterGrid({"n_estimators": [100, 300, 600], "max_features": ["sqrt", 0.25]})),
                "DTS+RF": list(ParameterGrid({"n_estimators": [100, 300, 600], "max_features": ["sqrt", 0.25]})),
                "DTS+HGB": list(ParameterGrid({"max_iter": [100, 300, 600], "learning_rate": [0.05, 0.1, 0.2], "l2_regularization": [0.0, 0.1]})),
            }
        factories = {
            "DTS+LR": lambda p: make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=3000, random_state=args.seed)),
            "DTS+ET": lambda p: ExtraTreesClassifier(n_estimators=p["n_estimators"], max_features=p["max_features"], n_jobs=-1, random_state=args.seed),
            "DTS+RF": lambda p: RandomForestClassifier(n_estimators=p["n_estimators"], max_features=p["max_features"], n_jobs=-1, random_state=args.seed),
            "DTS+HGB": lambda p: HistGradientBoostingClassifier(max_iter=p["max_iter"], learning_rate=p["learning_rate"], l2_regularization=p["l2_regularization"], random_state=args.seed),
        }
        data_for = {
            "DTS+LR": (X_train, X_val, X_test),
            "DTS+ET": (X_train, X_val, X_test),
            "DTS+RF": (X_train, X_val, X_test),
            "DTS+HGB": (X_train, X_val, X_test),
        }
        for name in ["DTS+LR", "DTS+ET", "DTS+RF", "DTS+HGB"]:
            xt, xv, xte = data_for[name]
            params, model = validation_grid(name, grids[name], factories[name], xt, y_train, xv, y_val)
            pv, pt, _ = fit_predict_sklearn(name, model, xt, y_train, xv, xte)
            add_result(name, pv, pt, {"selected_params": params, "tau_star": tau_star})

    if "mini" in models_requested or "all" in models_requested:
        try:
            from aeon.classification.convolution_based import MiniRocketClassifier
        except Exception:
            from aeon.classification.convolution_based import MiniRocket

            MiniRocketClassifier = MiniRocket
        Xae_train = np.transpose(S_train, (0, 2, 1)).astype(np.float32)
        Xae_val = np.transpose(S_val, (0, 2, 1)).astype(np.float32)
        Xae_test = np.transpose(S_test, (0, 2, 1)).astype(np.float32)
        kernel_grid = [1000] if args.quick else [1000, 5000, 10000]
        best = None
        for kernels in kernel_grid:
            print(f"Training MiniROCKET with {kernels} kernels", flush=True)
            clf = MiniRocketClassifier(
                n_kernels=kernels,
                estimator=LogisticRegression(max_iter=2000, n_jobs=-1, random_state=args.seed),
                random_state=args.seed,
            )
            clf.fit(Xae_train, y_train)
            if hasattr(clf, "predict_proba"):
                pv = clf.predict_proba(Xae_val)[:, 1]
            else:
                pv = clf.decision_function(Xae_val)
                pv = 1 / (1 + np.exp(-pv))
            auc = roc_auc_score(y_val, pv)
            if best is None or auc > best[0]:
                best = (auc, kernels, clf, pv)
        assert best is not None
        clf = best[2]
        if hasattr(clf, "predict_proba"):
            pt = clf.predict_proba(Xae_test)[:, 1]
            pv = clf.predict_proba(Xae_val)[:, 1]
        else:
            pv = 1 / (1 + np.exp(-clf.decision_function(Xae_val)))
            pt = 1 / (1 + np.exp(-clf.decision_function(Xae_test)))
        add_result("MiniROCKET", pv, pt, {"n_kernels": best[1], "input_shape": list(Xae_train.shape)})

    if "neural" in models_requested or "all" in models_requested:
        requested_neural = {m.strip() for m in args.neural_models.split(",") if m.strip()}
        neural_specs = [
            ("LSTM", FallRNN("lstm"), S_train, y_train, S_val, y_val, S_test, args.neural_epochs, 1e-3, 64, False),
            ("GRU", FallRNN("gru"), S_train, y_train, S_val, y_val, S_test, args.neural_epochs, 1e-3, 64, False),
            ("Transformer", FallTransformer(), S_train, y_train, S_val, y_val, S_test, args.neural_epochs, 1e-3, 64, False),
            ("DTS-Net", DTSNet(), Xs_train, y_train, Xs_val, y_val, Xs_test, args.neural_epochs if args.quick else max(args.neural_epochs, 40), 3e-3, 64, True),
            ("CompactST-GCN", STGCN([64, 64, 128], dropout=0.3), S_train, y_train, S_val, y_val, S_test, args.neural_epochs, 1e-3, 64, False),
            ("FullST-GCN-COCO", STGCN([64, 64, 64, 128, 128, 256], dropout=0.3), S_train, y_train, S_val, y_val, S_test, args.full_stgcn_epochs, 1e-3, 48, False),
        ]
        for name, model, xt, yt, xv, yv, xte, epochs, lr, batch, is_dts in neural_specs:
            if requested_neural != {"all"} and name not in requested_neural:
                continue
            pv, pt, config = train_torch_model(name, model, xt, yt, xv, yv, xte, epochs=epochs, lr=lr, batch_size=batch, seed=args.seed, is_dts=is_dts)
            add_result(name, pv, pt, config)

    np.savez_compressed(out / "score_vectors_twinfree.npz", **score_vectors)

    matched = {}
    if "DTS+HGB" in results and "MiniROCKET" in results:
        score_map = {
            "DTS+HGB": score_vectors["DTS+HGB_test"],
            "MiniROCKET": score_vectors["MiniROCKET_test"],
            "DTS+HGB__default_metrics": results["DTS+HGB"],
            "MiniROCKET__default_metrics": results["MiniROCKET"],
        }
        matched["DTS+HGB_vs_MiniROCKET"] = matched_operating_analysis(y_test, score_map, "DTS+HGB", "MiniROCKET")

    summary = {
        "protocol": "strict_unique_feature_group_twinfree",
        "status": "complete",
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "min_frames": args.min_frames,
        "dedup": dedup_meta,
        "split": {
            k: {
                "n": len(v),
                "falls": int(y[v].sum()),
                "nonfalls": int(len(v) - y[v].sum()),
            }
            for k, v in split.items()
        },
        "tau_star": tau_star,
        "models": results,
        "selected_configs": selected,
        "matched_operating_points": matched,
        "artifacts": {
            "split_manifest": "split_manifest_twinfree.csv",
            "dedup_audit": "dedup_audit.csv",
            "parse_failures": "parse_failures.csv",
            "score_vectors": "score_vectors_twinfree.npz",
        },
        "command": "python work/ninefive/run_twinfree_benchmark.py --models all",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "twinfree_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["split"], indent=2))
    print(f"Saved artifacts under {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models", default="all", help="comma list: dts,mini,neural,all")
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--neural_epochs", type=int, default=20)
    parser.add_argument("--full_stgcn_epochs", type=int, default=24)
    parser.add_argument("--neural_models", default="all", help="comma list of neural model names, or all")
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--tune_tau", action="store_true", help="select tau on training data; slower than fixed tau=0.30")
    parser.add_argument("--quick", action="store_true", help="reduced search spaces for smoke testing")
    parser.add_argument("--cache_npz", default=str(ROOT / "work" / "ninefive" / "fallvision_min10_tau030_cache.npz"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
