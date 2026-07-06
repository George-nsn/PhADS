#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Residue Pool-Gate SDH-ProtoNet inference script.

This script is the deployment inference counterpart for the best ablation model
selected from D:/pre_tran/result/ablation_gate/run_20260619_235417/
pool_loss_proto_only_judge.
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ResiduePoolGateSDHProtoNet(nn.Module):
    def __init__(self, esm_dim=1280, prost_dim=1024, evo_dim=212, latent_dim=512, cross_dropout=0.1):
        super().__init__()
        self.proj_esm2 = nn.Linear(esm_dim, latent_dim)
        self.proj_prost = nn.Linear(prost_dim, latent_dim)
        self.norm_esm2 = nn.LayerNorm(latent_dim)
        self.norm_prost = nn.LayerNorm(latent_dim)
        self.seq_gate = nn.Sequential(nn.Linear(latent_dim * 2, 1), nn.Sigmoid())
        self.evo_net = nn.Sequential(nn.Linear(evo_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(latent_dim * 2, 1), nn.Sigmoid())
        self.projector = nn.Linear(latent_dim, latent_dim)
        self.align_esm2 = nn.Sequential(nn.Linear(esm_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.align_prost = nn.Sequential(nn.Linear(prost_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.dropout = nn.Dropout(float(cross_dropout))

    @staticmethod
    def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=tensor.device, dtype=torch.bool)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(tensor.dtype)
        return (tensor * mask.unsqueeze(-1).to(tensor.dtype)).sum(dim=1) / denom

    def forward(self, esm2, prost, mask, x_evo, return_gate=False, return_branch=False):
        mask = mask.to(device=esm2.device, dtype=torch.bool)
        h_esm2 = self.dropout(self.norm_esm2(self.proj_esm2(esm2)))
        h_prost = self.dropout(self.norm_prost(self.proj_prost(prost)))
        e_esm2 = self.masked_mean(h_esm2, mask)
        e_prost = self.masked_mean(h_prost, mask)
        seq_g = self.seq_gate(torch.cat([e_esm2, e_prost], dim=-1))
        e_seq = seq_g * e_esm2 + (1.0 - seq_g) * e_prost
        f_e = self.evo_net(x_evo)
        g = self.gate(torch.cat([e_seq, f_e], dim=-1))
        fused = g * e_seq + (1.0 - g) * f_e
        z = F.normalize(self.projector(fused), p=2, dim=1)
        outputs = [z]
        if return_gate:
            outputs.append(g.squeeze(-1))
        if return_branch:
            outputs.append(seq_g.squeeze(-1))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class ResidueSingleSDHProtoNet(nn.Module):
    def __init__(self, residue_dim=1280, evo_dim=212, latent_dim=512, cross_dropout=0.1):
        super().__init__()
        self.proj_residue = nn.Linear(residue_dim, latent_dim)
        self.norm_residue = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(float(cross_dropout))
        self.evo_net = nn.Sequential(nn.Linear(evo_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(latent_dim * 2, 1), nn.Sigmoid())
        self.projector = nn.Linear(latent_dim, latent_dim)

    @staticmethod
    def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=tensor.device, dtype=torch.bool)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(tensor.dtype)
        return (tensor * mask.unsqueeze(-1).to(tensor.dtype)).sum(dim=1) / denom

    def forward(self, residue, mask, x_evo, return_gate=False, return_branch=False):
        mask = mask.to(device=residue.device, dtype=torch.bool)
        h_residue = self.dropout(self.norm_residue(self.proj_residue(residue)))
        e_residue = self.masked_mean(h_residue, mask)
        f_e = self.evo_net(x_evo)
        g = self.gate(torch.cat([e_residue, f_e], dim=-1))
        fused = g * e_residue + (1.0 - g) * f_e
        z = F.normalize(self.projector(fused), p=2, dim=1)
        outputs = [z]
        if return_gate:
            outputs.append(g.squeeze(-1))
        if return_branch:
            outputs.append(torch.ones(z.shape[0], device=z.device, dtype=z.dtype))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def parse_args():
    parser = argparse.ArgumentParser(description="Residue Pool-Gate SDH-ProtoNet predictor")
    parser.add_argument("--model-type", choices=["ESM2", "ProtT5", "mix"], default="mix")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--pair-dir", default="", help="Mix mode: directory containing paired_pt/*.pt and optional residue_pair_manifest.tsv")
    parser.add_argument("--residue-dir", default="", help="Single-source mode (ESM2 or ProtT5): directory containing *_embedding.pt files")
    parser.add_argument("--query-hmm", required=True, help="Query HMM feature .npy")
    parser.add_argument("--query-ids", required=True, help="Query ID file, one ID per line")
    parser.add_argument("--threshold-tsv", default="")
    parser.add_argument("--ads-detection-threshold-tsv", default="", help="External calibrated ADS/non-ADS judge_score threshold TSV")
    parser.add_argument("--ads-detection-cluster-threshold-tsv", default="", help="Optional cluster-guarded ADS/non-ADS threshold TSV")
    parser.add_argument("--ads-detection-threshold-mode", choices=["auto", "global", "cluster-guarded"], default="global")
    parser.add_argument("--cluster-annotation-txt", default="")
    parser.add_argument("--ads-function-txt", default="")
    parser.add_argument("--mode", choices=["prototype", "instance"], default="prototype")
    parser.add_argument("--use-multiprototype", action="store_true", help="Use parent-preserving sub-prototypes when available in family_map.pth")
    parser.add_argument("--judge-mode", choices=["off", "heuristic"], default="heuristic", help="Online prototype+HMM reranking mode")
    parser.add_argument("--topk-hmm", type=int, default=5)
    parser.add_argument("--filter-mode", choices=["strict", "moderate", "loose", "none"], default="moderate")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-tsv", default="prediction_results.tsv")
    parser.add_argument("--topk-tsv", default="prediction_topk.tsv")
    parser.add_argument("--print-topk", action="store_true")
    return parser.parse_args()


def safe_file_stem(seq_id: str) -> str:
    primary = re.split(r"[;|/,]", seq_id, maxsplit=1)[0] or seq_id
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", primary)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned if cleaned else "seq"


def read_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


class ResiduePairQueryDataset(Dataset):
    def __init__(self, pair_dir: str, query_hmm: str, ids_path: str):
        self.pair_dir = Path(pair_dir)
        self.ids = read_ids(ids_path)
        self.hmm = np.load(query_hmm, mmap_mode="r")
        if self.hmm.ndim != 2:
            raise ValueError(f"query_hmm must be 2D, got shape={self.hmm.shape}")
        if len(self.ids) != self.hmm.shape[0]:
            raise ValueError(f"query ids count ({len(self.ids)}) != HMM rows ({self.hmm.shape[0]})")
        self.manifest = self._load_manifest()
        self.paths = []
        missing = []
        for seq_id in self.ids:
            rel = self.manifest.get(seq_id)
            path = self.pair_dir / rel if rel else self.pair_dir / "paired_pt" / f"{safe_file_stem(seq_id)}.pt"
            if not path.exists():
                missing.append(seq_id)
            self.paths.append(path)
        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(f"Missing paired residue files for {len(missing)} queries, e.g. {preview}")

        first = torch.load(self.paths[0], map_location="cpu")
        self.esm_dim = int(first["esm2"].shape[1])
        self.prost_dim = int(first["prost5"].shape[1])

    def _load_manifest(self) -> Dict[str, str]:
        manifest_path = self.pair_dir / "residue_pair_manifest.tsv"
        out: Dict[str, str] = {}
        if not manifest_path.exists():
            return out
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("status") == "ok" and row.get("seq_id") and row.get("paired_file"):
                    out[row["seq_id"]] = row["paired_file"]
        return out

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        payload = torch.load(self.paths[index], map_location="cpu")
        esm2 = payload["esm2"].float()
        prost5 = payload["prost5"].float()
        if esm2.shape[0] != prost5.shape[0]:
            raise ValueError(f"Length mismatch for {self.ids[index]}: esm2={esm2.shape[0]}, prost5={prost5.shape[0]}")
        mask = payload.get("mask")
        if mask is None:
            mask = torch.ones(esm2.shape[0], dtype=torch.bool)
        return {
            "esm2": esm2,
            "prost5": prost5,
            "mask": mask.bool(),
            "evo": torch.from_numpy(np.array(self.hmm[index], copy=True)).float(),
            "query_id": self.ids[index],
        }


class ResidueSingleQueryDataset(Dataset):
    def __init__(self, residue_dir: str, query_hmm: str, ids_path: str):
        self.residue_dir = Path(residue_dir)
        self.ids = read_ids(ids_path)
        self.hmm = np.load(query_hmm, mmap_mode="r")
        if self.hmm.ndim != 2:
            raise ValueError(f"query_hmm must be 2D, got shape={self.hmm.shape}")
        if len(self.ids) != self.hmm.shape[0]:
            raise ValueError(f"query ids count ({len(self.ids)}) != HMM rows ({self.hmm.shape[0]})")
        self.paths = []
        missing = []
        for seq_id in self.ids:
            path = find_residue_tensor_file(self.residue_dir, seq_id)
            if path is None:
                missing.append(seq_id)
                self.paths.append(Path(""))
            else:
                self.paths.append(path)
        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(f"Missing single residue files for {len(missing)} queries, e.g. {preview}")
        first = load_residue_tensor(self.paths[0])
        self.residue_dim = int(first.shape[1])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        residue = load_residue_tensor(self.paths[index]).float()
        mask = torch.ones(residue.shape[0], dtype=torch.bool)
        return {
            "single_residue": residue,
            "mask": mask,
            "evo": torch.from_numpy(np.array(self.hmm[index], copy=True)).float(),
            "query_id": self.ids[index],
        }


def candidate_paths(root: Path, seq_id: str):
    stem = safe_file_stem(seq_id)
    names = [
        f"{stem}_esm2_residue.pt", f"{seq_id}_esm2_residue.pt",
        f"{stem}_embedding.pt", f"{stem}.pt",
        f"{seq_id}_embedding.pt", f"{seq_id}.pt",
    ]
    out = []
    for name in names:
        out.extend([root / name, root / stem / name, root / seq_id / name])
    return out


def find_residue_tensor_file(root: Path, seq_id: str) -> Optional[Path]:
    for path in candidate_paths(root, seq_id):
        if path.exists() and path.is_file():
            return path
    stem = safe_file_stem(seq_id)
    hits = sorted(root.glob(f"**/{stem}*.pt"))
    return hits[0] if hits else None


def load_residue_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    tensor = None
    if isinstance(obj, dict):
        for key in ("embedding", "emb", "residue", "residue_embedding", "esm2", "features"):
            if key in obj and torch.is_tensor(obj[key]):
                tensor = obj[key]
                break
    elif torch.is_tensor(obj):
        tensor = obj
    if tensor is None:
        raise TypeError(f"{path} is not a Tensor or supported dict")
    if tensor.dim() != 2:
        raise ValueError(f"{path} must be [L,D], got {tuple(tensor.shape)}")
    return tensor.cpu()


def residue_pair_collate(batch):
    batch_size = len(batch)
    max_len = max(item["esm2"].shape[0] for item in batch)
    esm_dim = batch[0]["esm2"].shape[1]
    prost_dim = batch[0]["prost5"].shape[1]
    esm2 = torch.zeros(batch_size, max_len, esm_dim, dtype=torch.float32)
    prost5 = torch.zeros(batch_size, max_len, prost_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    evo = torch.stack([item["evo"] for item in batch], dim=0)
    query_ids = [item["query_id"] for item in batch]
    for row, item in enumerate(batch):
        length = item["esm2"].shape[0]
        esm2[row, :length] = item["esm2"]
        prost5[row, :length] = item["prost5"]
        mask[row, :length] = item["mask"][:length]
    return {"esm2": esm2, "prost5": prost5, "mask": mask, "evo": evo, "query_ids": query_ids}


def residue_single_collate(batch):
    batch_size = len(batch)
    max_len = max(item["single_residue"].shape[0] for item in batch)
    residue_dim = batch[0]["single_residue"].shape[1]
    residue = torch.zeros(batch_size, max_len, residue_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    evo = torch.stack([item["evo"] for item in batch], dim=0)
    query_ids = [item["query_id"] for item in batch]
    for row, item in enumerate(batch):
        length = item["single_residue"].shape[0]
        residue[row, :length] = item["single_residue"]
        mask[row, :length] = item["mask"][:length]
    return {"single_residue": residue, "mask": mask, "evo": evo, "query_ids": query_ids}


def ensure_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available but --device={value} was requested")
    return device


def infer_latent_dim(state_dict: Dict[str, torch.Tensor], fallback: int = 512) -> int:
    for key, value in state_dict.items():
        if key.endswith("projector.weight") and value.ndim == 2:
            return int(value.shape[0])
    return fallback


def ensure_2d_hmm(path: str, expected_dim: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"query_hmm must be 1D or 2D, got shape={arr.shape}")
    if arr.shape[1] < expected_dim:
        pad = np.zeros((arr.shape[0], expected_dim - arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
        np.save(path, arr.astype(np.float32, copy=False))
    elif arr.shape[1] > expected_dim:
        arr = arr[:, :expected_dim]
        np.save(path, arr.astype(np.float32, copy=False))
    return arr


def is_missing(value: str) -> bool:
    text = str(value).strip()
    return (not text) or text.lower() in {"nan", "none", "na", "-"}


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def first_existing_column(col_idx: Dict[str, int], aliases: Sequence[str]) -> int:
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in col_idx:
            return col_idx[key]
    return -1


def parse_cluster_label(value: str) -> Optional[int]:
    match = re.search(r"cluster[_-]?(\d+)", str(value), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|\D)(\d+)(?:\D|$)", str(value))
    return int(match.group(1)) if match else None


def cluster_name_variants(value: str) -> List[str]:
    text = str(value).strip()
    variants = []
    if text:
        variants.extend([text, text.lower(), text.upper()])
        label = parse_cluster_label(text)
        if label is not None:
            variants.extend([f"cluster_{label}", f"Cluster_{label}", f"CLUSTER_{label}"])
    return list(dict.fromkeys(variants))


def load_cluster_annotation(path: str):
    if not path or not os.path.exists(path):
        return {}, [], {}
    out: Dict[int, Dict[str, str]] = {}
    label_to_rep: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col_idx = {normalize_column_name(name): idx for idx, name in enumerate(header)}
        label_col = first_existing_column(col_idx, ["label", "family_label", "family_id", "cluster_id"])
        cluster_col = first_existing_column(col_idx, ["cluster_name", "cluster", "family", "family_name"])
        rep_col = first_existing_column(col_idx, ["representative", "rep_name", "representative_id", "rep", "representative_sequence"])
        if label_col < 0 and cluster_col < 0:
            return {}, [], {}
        reserved = {idx for idx in (label_col, cluster_col, rep_col) if idx >= 0}
        func_cols = [name for idx, name in enumerate(header) if idx not in reserved]
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            label = None
            if label_col >= 0 and label_col < len(parts):
                label = parse_cluster_label(parts[label_col])
            if label is None and cluster_col >= 0 and cluster_col < len(parts):
                label = parse_cluster_label(parts[cluster_col])
            if label is None:
                continue
            if rep_col >= 0 and rep_col < len(parts) and not is_missing(parts[rep_col]):
                label_to_rep[label] = parts[rep_col].strip()
            one = {}
            for col in func_cols:
                idx = header.index(col)
                value = parts[idx].strip() if idx < len(parts) else ""
                one[col] = "" if is_missing(value) else value
            out[label] = one
    return out, func_cols, label_to_rep


def load_ads_function(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col_idx = {normalize_column_name(name): idx for idx, name in enumerate(header)}
        name_col = first_existing_column(col_idx, ["ADS_name", "ads_name", "name", "seq_id", "sequence_id", "representative", "rep_name"])
        cluster_col = first_existing_column(col_idx, ["cluster", "cluster_name", "family", "family_id"])
        func_col = first_existing_column(col_idx, ["Against/Function", "against_function", "function", "ads_function", "against"])
        if func_col < 0 or (name_col < 0 and cluster_col < 0):
            return {}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            func = parts[func_col].strip() if func_col < len(parts) else ""
            if is_missing(func):
                continue
            keys = []
            if name_col >= 0 and name_col < len(parts) and not is_missing(parts[name_col]):
                keys.append(parts[name_col].strip())
            if cluster_col >= 0 and cluster_col < len(parts) and not is_missing(parts[cluster_col]):
                keys.extend(cluster_name_variants(parts[cluster_col]))
            for key in keys:
                out.setdefault(key, [])
                if func not in out[key]:
                    out[key].append(func)
    return {key: "; ".join(values) for key, values in out.items()}


def lookup_ads_function(seq_id: str, ads_map: Dict[str, str]) -> str:
    if not seq_id:
        return ""
    if seq_id in ads_map:
        return ads_map[seq_id]
    for variant in cluster_name_variants(seq_id):
        if variant in ads_map:
            return ads_map[variant]
    if "." in seq_id:
        base = seq_id.rsplit(".", 1)[0]
        return ads_map.get(base, "")
    return ""


def load_thresholds(path: str) -> Dict[str, Dict[str, object]]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
        header_line = handle.readline().strip()
        if not header_line:
            return {}
        header = [item.strip() for item in header_line.split("\t") if item.strip()]
        if len(header) <= 1:
            header = header_line.split()
        col_idx = {name: idx for idx, name in enumerate(header)}
        required = ["family_id", "threshold_strict", "threshold_moderate", "threshold_loose"]
        if any(key not in col_idx for key in required):
            return {}
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = [item.strip() for item in line.split("\t") if item.strip()]
            if len(parts) <= 1:
                parts = line.split()
            if len(parts) < max(col_idx.values()) + 1:
                continue
            family_id = parts[col_idx["family_id"]]
            try:
                out[family_id] = {
                    "strict": float(parts[col_idx["threshold_strict"]]),
                    "moderate": float(parts[col_idx["threshold_moderate"]]),
                    "loose": float(parts[col_idx["threshold_loose"]]),
                    "strict_confidence": parts[col_idx["strict_confidence"]] if "strict_confidence" in col_idx and col_idx["strict_confidence"] < len(parts) else "",
                    "moderate_confidence": parts[col_idx["moderate_confidence"]] if "moderate_confidence" in col_idx and col_idx["moderate_confidence"] < len(parts) else "",
                    "loose_confidence": parts[col_idx["loose_confidence"]] if "loose_confidence" in col_idx and col_idx["loose_confidence"] < len(parts) else "",
                }
            except ValueError:
                continue
    return out


def load_ads_detection_threshold(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                return {
                    "score_field": row.get("score_field", "judge_score") or "judge_score",
                    "threshold": float(row["threshold"]),
                    "selection_rule": row.get("selection_rule", ""),
                    "source": row.get("source", ""),
                    "calibration_fraction": row.get("calibration_fraction", ""),
                    "n_splits": row.get("n_splits", ""),
                }
            except (KeyError, TypeError, ValueError):
                return {}
    return {}


def load_ads_detection_cluster_thresholds(path: str) -> Dict[str, Dict[str, object]]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cluster = str(row.get("cluster", "")).strip()
            if not cluster:
                continue
            try:
                final_threshold = float(row.get("final_threshold", ""))
            except (TypeError, ValueError):
                continue
            use_cluster = str(row.get("use_cluster_threshold", "0")).strip().lower() in {"1", "true", "yes", "y"}
            out[cluster] = {
                "final_threshold": final_threshold,
                "global_threshold": _safe_float(row.get("global_threshold", "")),
                "cluster_threshold_median": _safe_float(row.get("cluster_threshold_median", "")),
                "use_cluster_threshold": use_cluster,
                "selection_rule": row.get("selection_rule", ""),
                "threshold_mode": row.get("threshold_mode", "cluster-guarded"),
                "source": row.get("source", ""),
                "guarded_use_fraction": row.get("guarded_use_fraction", ""),
                "shrinkage_k": row.get("shrinkage_k", ""),
            }
    return out


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def resolve_ads_detection_mode(requested_mode: str, model_type: str, cluster_thresholds: Dict[str, Dict[str, object]]) -> str:
    if requested_mode == "auto":
        if model_type == "mix" and cluster_thresholds:
            return "cluster-guarded"
        return "global"
    if requested_mode == "cluster-guarded" and (model_type != "mix" or not cluster_thresholds):
        return "global"
    return requested_mode


def select_ads_detection_threshold(
    pred_cluster: str,
    global_threshold: float,
    mode: str,
    cluster_thresholds: Dict[str, Dict[str, object]],
) -> Tuple[float, str, str, str]:
    if mode == "cluster-guarded" and cluster_thresholds:
        info = cluster_thresholds.get(pred_cluster)
        if info and bool(info.get("use_cluster_threshold")):
            return (
                float(info["final_threshold"]),
                "cluster-guarded",
                str(info.get("selection_rule", "")),
                pred_cluster,
            )
        return global_threshold, "cluster-guarded", "global_threshold_fallback", pred_cluster
    return global_threshold, "global", "global_threshold", pred_cluster


def format_threshold(value: float) -> str:
    if value == float("inf") or value == float("-inf") or np.isnan(value):
        return "NA"
    return f"{value:.4f}"


def build_model_from_payload(payload: Dict, state: Dict[str, torch.Tensor], dataset) -> nn.Module:
    latent_dim = infer_latent_dim(state, fallback=int(payload.get("latent_dim", 512)))
    if payload.get("residue_feature_mode") == "esm2" or isinstance(dataset, ResidueSingleQueryDataset):
        return ResidueSingleSDHProtoNet(
            residue_dim=int(payload.get("single_residue_dim", getattr(dataset, "residue_dim", 1280))),
            evo_dim=int(payload.get("evo_dim", dataset.hmm.shape[1])),
            latent_dim=latent_dim,
            cross_dropout=float(payload.get("cross_dropout", 0.1)),
        )
    return ResiduePoolGateSDHProtoNet(
        esm_dim=int(payload.get("esm_dim", dataset.esm_dim)),
        prost_dim=int(payload.get("prost_dim", dataset.prost_dim)),
        evo_dim=int(payload.get("evo_dim", dataset.hmm.shape[1])),
        latent_dim=latent_dim,
        cross_dropout=float(payload.get("cross_dropout", 0.1)),
    )


def normalize_hmm_row(row: np.ndarray) -> np.ndarray:
    row = np.nan_to_num(np.asarray(row, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(row, 0.0, 1.0)


def hmm_score_for_label(hmm_row: np.ndarray, label: int) -> float:
    row = normalize_hmm_row(hmm_row)
    return float(row[label]) if 0 <= int(label) < row.shape[0] else 0.0


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def judge_score(proto_dist: float, proto_prob: float, proto_rank: int, hmm_score: float, hmm_rank: int, is_proto_top1: bool, is_hmm_top1: bool) -> float:
    # Lightweight online counterpart of train_judge_network.py. Distances are inverted,
    # while HMM evidence and rank agreement raise the rerank score.
    dist_term = 1.0 / (1.0 + max(float(proto_dist), 0.0))
    proto_rank_term = 1.0 / max(int(proto_rank), 1)
    hmm_rank_term = 1.0 / max(int(hmm_rank), 1) if hmm_score > 0 else 0.0
    agreement = 1.0 if is_proto_top1 and is_hmm_top1 else 0.0
    return (
        0.42 * dist_term
        + 0.20 * float(proto_prob)
        + 0.13 * proto_rank_term
        + 0.18 * float(hmm_score)
        + 0.05 * hmm_rank_term
        + 0.02 * agreement
    )


def maybe_filter_gallery(gallery, labels, ids, retained_labels: Optional[Sequence[int]], device: torch.device):
    if not retained_labels:
        return gallery, labels, ids
    retained = set(int(x) for x in retained_labels)
    keep = [idx for idx, label in enumerate(labels) if int(label) in retained]
    if not keep:
        raise ValueError("Retained label filtering removed all gallery entries")
    idx_tensor = torch.as_tensor(keep, dtype=torch.long, device=device)
    return gallery.index_select(0, idx_tensor), [labels[i] for i in keep], [ids[i] for i in keep]


def main():
    args = parse_args()
    device = ensure_device(args.device)

    payload = torch.load(args.map_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("family_map.pth must contain a dict payload")
    expected_evo_dim = int(payload.get("evo_dim", 212))
    ensure_2d_hmm(args.query_hmm, expected_evo_dim)

    if args.model_type in {"ESM2", "ProtT5"}:
        if not args.residue_dir:
            raise ValueError("--model-type ESM2/ProtT5 requires --residue-dir")
        dataset = ResidueSingleQueryDataset(args.residue_dir, args.query_hmm, args.query_ids)
        collate_fn = residue_single_collate
    else:
        if not args.pair_dir:
            raise ValueError("--model-type mix requires --pair-dir")
        dataset = ResiduePairQueryDataset(args.pair_dir, args.query_hmm, args.query_ids)
        collate_fn = residue_pair_collate
    if dataset.hmm.shape[1] != expected_evo_dim:
        raise ValueError(f"query_hmm dimension mismatch after normalization: {dataset.hmm.shape[1]} != {expected_evo_dim}")

    state = torch.load(args.model_path, map_location="cpu")
    model = build_model_from_payload(payload, state, dataset)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    cluster_annotation, cluster_func_cols, annotation_reps = load_cluster_annotation(args.cluster_annotation_txt)
    ads_function = load_ads_function(args.ads_function_txt)
    thresholds = load_thresholds(args.threshold_tsv) if args.filter_mode != "none" else {}
    ads_detection_threshold = load_ads_detection_threshold(args.ads_detection_threshold_tsv)
    ads_detection_cluster_thresholds = load_ads_detection_cluster_thresholds(args.ads_detection_cluster_threshold_tsv)
    ads_detection_mode = resolve_ads_detection_mode(args.ads_detection_threshold_mode, args.model_type, ads_detection_cluster_thresholds)
    cluster_names: Dict[int, str] = {int(k): str(v) for k, v in payload.get("cluster_names", {}).items()}
    temperature = float(args.temperature) if args.temperature is not None else float(payload.get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    using_multi = bool(args.use_multiprototype and args.mode == "prototype" and "multi_prototypes" in payload)
    if args.mode == "prototype":
        if using_multi:
            gallery = payload["multi_prototypes"].float().to(device)
            gallery_labels = [int(x) for x in payload["multi_parent_labels"]]
            gallery_ids = [str(x) for x in payload.get("multi_names", [f"subprototype_{i}" for i in range(len(gallery_labels))])]
        else:
            gallery = payload["prototypes"].float().to(device)
            gallery_labels = [int(x) for x in payload["labels"]]
            gallery_ids = [f"prototype_of_{label}" for label in gallery_labels]
    else:
        gallery = payload["instance_features"].float().to(device)
        gallery_labels = [int(x) for x in payload["instance_labels"]]
        gallery_ids = [str(x) for x in payload["instance_ids"]]

    instance_gallery = None
    instance_labels: List[int] = []
    instance_ids: List[str] = []
    if "instance_features" in payload and "instance_labels" in payload and "instance_ids" in payload:
        instance_gallery = payload["instance_features"].float().to(device)
        instance_labels = [int(x) for x in payload["instance_labels"]]
        instance_ids = [str(x) for x in payload["instance_ids"]]

    topk = max(1, min(int(args.topk), len(gallery_labels)))
    loader = DataLoader(dataset, batch_size=max(1, int(args.batch_size)), shuffle=False, collate_fn=collate_fn)
    results = []

    with torch.no_grad():
        for batch in loader:
            mask = batch["mask"].to(device)
            evo = batch["evo"].to(device)
            query_ids = batch["query_ids"]
            if args.model_type in {"ESM2", "ProtT5"}:
                residue = batch["single_residue"].to(device)
                qz, hmm_gate, seq_gate = model(residue, mask, evo, return_gate=True, return_branch=True)
            else:
                esm2 = batch["esm2"].to(device)
                prost5 = batch["prost5"].to(device)
                qz, hmm_gate, seq_gate = model(esm2, prost5, mask, evo, return_gate=True, return_branch=True)
            d2 = torch.cdist(qz, gallery, p=2).pow(2)
            probs = F.softmax(-d2 / temperature, dim=1)
            top_probs, top_idx = torch.topk(probs, k=topk, dim=1)
            top_d2 = torch.gather(d2, 1, top_idx)

            if instance_gallery is not None:
                d2_ins = torch.cdist(qz, instance_gallery, p=2).pow(2)
                ins_idx = torch.argmin(d2_ins, dim=1)
                ins_d2 = d2_ins[torch.arange(d2_ins.shape[0], device=device), ins_idx]
            else:
                ins_idx = None
                ins_d2 = None

            for row, query_id in enumerate(query_ids):
                d2_row = d2[row].detach().cpu().numpy().astype(np.float64)
                prob_row = probs[row].detach().cpu().numpy().astype(np.float64)
                proto_ranks = rank_asc(d2_row)
                hmm_row = dataset.hmm[dataset.ids.index(query_id)]
                hmm_values = np.asarray([hmm_score_for_label(hmm_row, label) for label in gallery_labels], dtype=np.float64)
                hmm_ranks = rank_desc(hmm_values)
                hmm_top_pos = int(np.argmax(hmm_values)) if len(hmm_values) else 0
                prototype_top1_gallery_pos = int(top_idx[row, 0].item())
                best_gallery_pos = prototype_top1_gallery_pos
                if args.judge_mode == "heuristic" and args.mode == "prototype":
                    candidate_positions = set(int(x.item()) for x in top_idx[row])
                    for pos in np.argsort(-hmm_values, kind="mergesort")[:max(1, int(args.topk_hmm))]:
                        if hmm_values[int(pos)] > 0:
                            candidate_positions.add(int(pos))
                    best_gallery_pos = max(
                        candidate_positions,
                        key=lambda pos: judge_score(
                            proto_dist=float(d2_row[pos]),
                            proto_prob=float(prob_row[pos]),
                            proto_rank=int(proto_ranks[pos]),
                            hmm_score=float(hmm_values[pos]),
                            hmm_rank=int(hmm_ranks[pos]),
                            is_proto_top1=(int(proto_ranks[pos]) == 1),
                            is_hmm_top1=(pos == hmm_top_pos and hmm_values[pos] > 0),
                        ),
                    )
                best_label = int(gallery_labels[best_gallery_pos])
                best_id = str(gallery_ids[best_gallery_pos])
                pred_cluster = cluster_names.get(best_label, f"cluster_{best_label}")
                best_dist = float(d2_row[best_gallery_pos])
                best_conf = float(prob_row[best_gallery_pos])
                pred_rep = annotation_reps.get(best_label, payload.get("cluster_representatives", {}).get(best_label, "NA"))

                if args.mode == "instance":
                    nearest_id = best_id
                    nearest_label = best_label
                    nearest_dist = best_dist
                elif instance_gallery is not None:
                    nearest_pos = int(ins_idx[row].item())
                    nearest_id = instance_ids[nearest_pos]
                    nearest_label = int(instance_labels[nearest_pos])
                    nearest_dist = float(ins_d2[row].item())
                else:
                    nearest_id = ""
                    nearest_label = -1
                    nearest_dist = float("nan")

                filter_status = "Pass"
                threshold_limit = float("inf")
                threshold_confidence = ""
                if args.filter_mode != "none" and thresholds:
                    if pred_cluster in thresholds:
                        threshold_limit = float(thresholds[pred_cluster][args.filter_mode])
                        threshold_confidence = str(thresholds[pred_cluster].get(f"{args.filter_mode}_confidence", ""))
                        filter_status = "Pass" if best_dist <= threshold_limit else "Fail"
                    else:
                        filter_status = "NoThreshold"

                candidates = []
                candidate_positions_for_output = [int(x.item()) for x in top_idx[row]]
                if best_gallery_pos not in candidate_positions_for_output:
                    candidate_positions_for_output = [best_gallery_pos] + candidate_positions_for_output[:-1]
                for rank, gallery_pos in enumerate(candidate_positions_for_output, start=1):
                    label = int(gallery_labels[gallery_pos])
                    cluster = cluster_names.get(label, f"cluster_{label}")
                    rep = annotation_reps.get(label, payload.get("cluster_representatives", {}).get(label, "NA"))
                    distance = float(d2_row[gallery_pos])
                    confidence = float(prob_row[gallery_pos])
                    candidate_status = "Pass"
                    candidate_limit = float("inf")
                    if args.filter_mode != "none" and thresholds:
                        if cluster in thresholds:
                            candidate_limit = float(thresholds[cluster][args.filter_mode])
                            candidate_status = "Pass" if distance <= candidate_limit else "Fail"
                        else:
                            candidate_status = "NoThreshold"
                    candidates.append({
                        "rank": rank,
                        "id": str(gallery_ids[gallery_pos]),
                        "label": label,
                        "cluster": cluster,
                        "rep": rep,
                        "distance2": distance,
                        "confidence": confidence,
                        "filter_status": candidate_status,
                        "threshold_limit": candidate_limit,
                        "ads_function": (
                            lookup_ads_function(str(gallery_ids[gallery_pos]), ads_function)
                            or lookup_ads_function(rep, ads_function)
                            or lookup_ads_function(cluster, ads_function)
                        ),
                        "hmm_score": float(hmm_values[gallery_pos]),
                        "hmm_rank": int(hmm_ranks[gallery_pos]),
                        "proto_rank": int(proto_ranks[gallery_pos]),
                        "judge_score": judge_score(
                            proto_dist=distance,
                            proto_prob=confidence,
                            proto_rank=int(proto_ranks[gallery_pos]),
                            hmm_score=float(hmm_values[gallery_pos]),
                            hmm_rank=int(hmm_ranks[gallery_pos]),
                            is_proto_top1=(int(proto_ranks[gallery_pos]) == 1),
                            is_hmm_top1=(gallery_pos == hmm_top_pos and hmm_values[gallery_pos] > 0),
                        ) if args.judge_mode == "heuristic" else float("nan"),
                    })

                if filter_status == "Fail" and args.filter_mode != "none":
                    best_pass = None
                    for candidate in candidates:
                        if candidate["filter_status"] in {"Pass", "NoThreshold"}:
                            if best_pass is None or candidate["distance2"] < best_pass["distance2"]:
                                best_pass = candidate
                    if best_pass is not None:
                        best_label = int(best_pass["label"])
                        best_id = str(best_pass["id"])
                        pred_cluster = best_pass["cluster"]
                        pred_rep = best_pass["rep"]
                        best_dist = float(best_pass["distance2"])
                        best_conf = float(best_pass["confidence"])
                        filter_status = best_pass["filter_status"]
                        threshold_limit = float(best_pass["threshold_limit"])
                        threshold_confidence = str(thresholds.get(pred_cluster, {}).get(f"{args.filter_mode}_confidence", "")) if thresholds else ""

                selected_judge_score = judge_score(
                    proto_dist=float(d2_row[prototype_top1_gallery_pos]),
                    proto_prob=float(prob_row[prototype_top1_gallery_pos]),
                    proto_rank=int(proto_ranks[prototype_top1_gallery_pos]),
                    hmm_score=float(hmm_values[prototype_top1_gallery_pos]),
                    hmm_rank=int(hmm_ranks[prototype_top1_gallery_pos]),
                    is_proto_top1=True,
                    is_hmm_top1=(prototype_top1_gallery_pos == hmm_top_pos and hmm_values[prototype_top1_gallery_pos] > 0),
                ) if args.judge_mode == "heuristic" else float("nan")
                global_detection_threshold = float(ads_detection_threshold.get("threshold", float("nan"))) if ads_detection_threshold else float("nan")
                detection_threshold, detection_threshold_mode, detection_threshold_rule, detection_threshold_cluster = select_ads_detection_threshold(
                    pred_cluster=pred_cluster,
                    global_threshold=global_detection_threshold,
                    mode=ads_detection_mode,
                    cluster_thresholds=ads_detection_cluster_thresholds,
                )
                if ads_detection_threshold and np.isfinite(selected_judge_score):
                    ads_detection_status = "ADS_candidate" if selected_judge_score >= detection_threshold else "Unknown_or_non_ADS"
                else:
                    ads_detection_status = "NotCalibrated"

                results.append({
                    "query_id": query_id,
                    "mode": args.mode,
                    "closest_member": nearest_id,
                    "pred_id": best_id,
                    "pred_label": best_label,
                    "pred_cluster": pred_cluster,
                    "pred_cluster_rep": pred_rep,
                    "pred_distance2": best_dist,
                    "confidence": best_conf,
                    "filter_mode": args.filter_mode,
                    "threshold_confidence": threshold_confidence,
                    "filter_status": filter_status,
                    "threshold_limit": threshold_limit,
                    "nearest_sequence_id": nearest_id,
                    "nearest_sequence_label": nearest_label,
                    "nearest_sequence_distance2": nearest_dist,
                    "nearest_sequence_function_summary": lookup_ads_function(nearest_id, ads_function),
                    "pred_ads_function": (
                        lookup_ads_function(best_id, ads_function)
                        or lookup_ads_function(pred_rep, ads_function)
                        or lookup_ads_function(pred_cluster, ads_function)
                    ),
                    "hmm_gate": float(hmm_gate[row].detach().cpu().item()),
                    "esm_gate": float(seq_gate[row].detach().cpu().item()),
                    "hmm_score": float(hmm_score_for_label(hmm_row, best_label)),
                    "judge_score": selected_judge_score,
                    "ads_detection_score": selected_judge_score,
                    "ads_detection_threshold": detection_threshold,
                    "ads_detection_status": ads_detection_status,
                    "ads_detection_threshold_mode": detection_threshold_mode,
                    "ads_detection_threshold_rule": detection_threshold_rule,
                    "ads_detection_threshold_cluster": detection_threshold_cluster,
                    "ads_detection_threshold_source": ads_detection_threshold.get("source", "") if ads_detection_threshold else "",
                    "prototype_source": "multi_prototype" if using_multi else "prototype",
                    "cluster_func_values": cluster_annotation.get(best_label, {}),
                    "topk": candidates,
                })

    cluster_func_out_cols = [f"cluster_func_{name}" for name in cluster_func_cols]
    with open(args.output_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "query_id", "mode", "closest_member", "pred_id", "pred_label", "pred_cluster", "pred_cluster_rep",
            "pred_distance2", "confidence", "filter_mode", "threshold_confidence", "filter_status", "threshold_limit",
            "nearest_sequence_id", "nearest_sequence_label", "nearest_sequence_distance2", "nearest_sequence_function_summary",
            "pred_ads_function", "hmm_gate", "esm_gate", "hmm_score", "judge_score", "prototype_source",
            "ads_detection_score", "ads_detection_threshold", "ads_detection_status", "ads_detection_threshold_source",
            "ads_detection_threshold_mode", "ads_detection_threshold_rule", "ads_detection_threshold_cluster",
            *cluster_func_out_cols,
        ])
        for item in results:
            if item["filter_status"] == "Fail" and args.filter_mode != "none":
                continue
            func_values = [item["cluster_func_values"].get(name, "") for name in cluster_func_cols]
            nearest_dist = "" if not np.isfinite(item["nearest_sequence_distance2"]) else f"{item['nearest_sequence_distance2']:.6f}"
            writer.writerow([
                item["query_id"], item["mode"], item["closest_member"], item["pred_id"], item["pred_label"],
                item["pred_cluster"], item["pred_cluster_rep"], f"{item['pred_distance2']:.6f}", f"{item['confidence']:.6f}",
                item["filter_mode"], item["threshold_confidence"], item["filter_status"], format_threshold(item["threshold_limit"]),
                item["nearest_sequence_id"], item["nearest_sequence_label"], nearest_dist,
                item["nearest_sequence_function_summary"], item["pred_ads_function"],
                f"{item['hmm_gate']:.6f}", f"{item['esm_gate']:.6f}",
                f"{item['hmm_score']:.6f}", f"{item['judge_score']:.6f}" if np.isfinite(item["judge_score"]) else "",
                item["prototype_source"],
                f"{item['ads_detection_score']:.6f}" if np.isfinite(item["ads_detection_score"]) else "",
                f"{item['ads_detection_threshold']:.6f}" if np.isfinite(item["ads_detection_threshold"]) else "",
                item["ads_detection_status"], item["ads_detection_threshold_source"],
                item["ads_detection_threshold_mode"], item["ads_detection_threshold_rule"], item["ads_detection_threshold_cluster"],
                *func_values,
            ])

    with open(args.topk_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "query_id", "mode", "rank", "candidate_id", "candidate_label", "candidate_cluster", "candidate_rep",
            "candidate_distance2", "candidate_confidence", "filter_mode", "filter_status", "threshold_limit",
            "proto_rank", "hmm_score", "hmm_rank", "judge_score", "candidate_ads_function",
        ])
        for item in results:
            for candidate in item["topk"]:
                writer.writerow([
                    item["query_id"], item["mode"], candidate["rank"], candidate["id"], candidate["label"], candidate["cluster"],
                    candidate["rep"], f"{candidate['distance2']:.6f}", f"{candidate['confidence']:.6f}",
                    item["filter_mode"], candidate["filter_status"], format_threshold(candidate["threshold_limit"]),
                    candidate.get("proto_rank", ""), f"{candidate.get('hmm_score', 0.0):.6f}", candidate.get("hmm_rank", ""),
                    f"{candidate.get('judge_score', float('nan')):.6f}" if np.isfinite(candidate.get("judge_score", float("nan"))) else "",
                    candidate["ads_function"],
                ])

    passed = sum(1 for item in results if item["filter_status"] != "Fail" or args.filter_mode == "none")
    print(f"Predictions written: {args.output_tsv} ({passed}/{len(results)} rows passed)")
    print(f"Top-k written: {args.topk_tsv}")
    if args.print_topk:
        for item in results:
            print(f"{item['query_id']}: {item['pred_cluster']} d2={item['pred_distance2']:.4f} conf={item['confidence']:.2%} status={item['filter_status']}")
            for candidate in item["topk"]:
                print(f"  top{candidate['rank']}: {candidate['cluster']} d2={candidate['distance2']:.4f} p={candidate['confidence']:.2%} status={candidate['filter_status']}")


if __name__ == "__main__":
    main()
