#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best-ablation PHADS-style residue inference pipeline.

Pipeline stages:
  1. Optional DNA gene prediction with pyrodigal-gv
  2. ESM2 residue embedding generation
  3. ProtT5 residue embedding generation
  4. HMMER hmmscan and HMM feature matrix construction
  5. ESM2/ProtT5 residue pairing
  6. Residue Pool-Gate SDH-ProtoNet prediction
  7. QC report generation
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


APP_VERSION = "0.6-best-ablation-triple-route"
ESM2_MODEL_DIRNAME = "esm2_t33_650M_UR50D"
PROST_MODEL_DIRNAME = "prot-t5-xl-uniref50-enc-onnx"
DEFAULT_MODEL_REL = Path("database") / "PhADS_model" / "best_model"
MODEL_TYPE_CHOICES = ["ESM2", "ProtT5", "mix"]
MODEL_DIR_BY_TYPE = {
    "ESM2": "single_model",
    "ProtT5": "prott5_model",
    "mix": "mix_model",
}


def parse_args():
    argv = [sys.argv[0]]
    for item in sys.argv[1:]:
        argv.append("--db" if item == "-db" else item)
    sys.argv = argv

    parser = argparse.ArgumentParser(
        description="PHADS-style inference pipeline for the best residue Pool-Gate ablation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
    db_parser = sub.add_parser("database", help="Download ESM2 and ProtT5 embedding models")
    db_parser.add_argument("--path", "-P", required=True, dest="download_path")

    parser.add_argument("-i", "--input-fasta", help="Input FASTA file; protein or DNA")
    parser.add_argument("-d", "--dir", dest="input_dir", help="Directory of FASTA files to process")
    parser.add_argument("-o", "--work-dir", default="run_auto", help="Final output directory")
    parser.add_argument("-temp", "--temp", dest="temp_dir", default=None, help="Temporary directory; default <work-dir>/temp")
    parser.add_argument("--db", "--mode-path", dest="mode_path", default=None, help="Embedding model root containing ESM2 and ProtT5 subdirs")
    parser.add_argument("-n", "--cpu", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--predict-mode", choices=["prototype", "instance", "foldseek", "hmm"], default="prototype")
    parser.add_argument("--model-type", choices=MODEL_TYPE_CHOICES, default="mix", help="ESM2=best trainable ESM2 layer30 residue model; ProtT5=best trainable ProtT5 residue model; mix=best Pool-Gate ESM2+ProtT5 residue model")
    parser.add_argument("--use-multiprototype", action="store_true", help="Use family_map multi_prototypes for prototype prediction when available")
    parser.add_argument("--judge-mode", choices=["off", "heuristic"], default="heuristic", help="Online HMM+prototype rerank mode")
    parser.add_argument("--ads-detection-threshold-mode", choices=["auto", "global", "cluster-guarded"], default="auto", help="ADS/non-ADS threshold mode; auto uses cluster-guarded for mix and global for ESM2/ProtT5")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--filter-mode", choices=["strict", "moderate", "loose", "none"], default="moderate")
    parser.add_argument("--prob-threshold", "-prob", dest="prob_threshold", type=float, default=0.8)
    parser.add_argument("--predict-output", default="prediction_results.tsv")
    parser.add_argument("--predict-topk-output", default="prediction_topk.tsv")
    parser.add_argument("--qc-txt", default="qc_report.txt")
    parser.add_argument("--qc-json", default="qc_report.json")
    parser.add_argument("--print-topk", action="store_true")
    parser.add_argument("--allow-length-mismatch", action="store_true", help="Compatibility flag: trim ESM2/ProtT5 residue embeddings to the shorter length")
    parser.add_argument("--strict-residue-length", action="store_true", help="Fail if ESM2 and ProtT5 residue lengths differ; by default the shorter aligned length is used")
    parser.add_argument("--residue-batch-size", type=int, default=8, help="Prediction batch size for residue model")
    parser.add_argument("-v", "--version", action="store_true", help="Run database/model self-check")
    return parser.parse_args()


def box_width() -> int:
    return 72


def print_header(items: List[Tuple[str, str]]):
    width = box_width()
    title = f"  PHADS {APP_VERSION} - Best Residue Pool-Gate Pipeline"
    print()
    print("=" * width)
    print(title)
    print("=" * width)
    for key, value in items:
        print(f"  {key:<24} {value}")
    print("-" * width)


def print_step(step: int, total: int, name: str, status: str = "start", detail: str = ""):
    if status == "start":
        print(f"[{step}/{total}] {name}...", flush=True)
    elif status == "done":
        suffix = f" ({detail})" if detail else ""
        print(f"[{step}/{total}] OK {name}{suffix}")
    elif status == "skip":
        print(f"[{step}/{total}] SKIP {name}")


def resolve_temp_dir(temp_arg: Optional[str], work_dir: Path) -> Path:
    if temp_arg:
        path = Path(temp_arg)
        return (path if path.is_absolute() else Path.cwd() / path).resolve()
    return (work_dir / "temp").resolve()


def resolve_downloaded_model_dirs(mode_path: Optional[str]) -> Dict[str, Optional[Path]]:
    if not mode_path:
        return {"model_root": None, "esm2_model": None, "prost_model": None}
    root = Path(mode_path).resolve()
    if root.name == ESM2_MODEL_DIRNAME:
        model_root = root.parent
        esm2_model = root
        prost_model = model_root / PROST_MODEL_DIRNAME
    elif root.name == PROST_MODEL_DIRNAME:
        model_root = root.parent
        esm2_model = model_root / ESM2_MODEL_DIRNAME
        prost_model = root
    else:
        model_root = root
        esm2_model = root / ESM2_MODEL_DIRNAME
        prost_model = root / PROST_MODEL_DIRNAME
    return {"model_root": model_root, "esm2_model": esm2_model, "prost_model": prost_model}


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def resolve_paths(script_dir: Path, args) -> Dict[str, Path]:
    db_dir = script_dir / "database"
    anno_dir = db_dir / "anno_database"
    model_type = getattr(args, "model_type", "mix")
    model_subdir = MODEL_DIR_BY_TYPE.get(model_type, "mix_model")
    model_dir = db_dir / "PhADS_model" / model_subdir
    model_dirs = resolve_downloaded_model_dirs(args.mode_path)
    out = {
        "translate_py": script_dir / "scripts" / "translate_to_embedding.py",
        "pair_py": script_dir / "scripts" / "pair_residue_embeddings.py",
        "hmm_to_npy_py": script_dir / "scripts" / "hmm_to_npy.py",
        "predict_py": script_dir / "scripts" / "predict_residue_pool_gate.py",
        "pyrodigal_py": script_dir / "scripts" / "pyrodigal_viral.py",
        "structure_compare_py": script_dir / "scripts" / "structure_compare.py",
        "download_py": script_dir / "scripts" / "download_model.py",
        "hmm_db": db_dir / "hmm_model" / "anti_defense_system.hmm",
        "foldseek_db": db_dir / "foldseek_db" / "phads_db",
        "cnn_weights": db_dir / "cnn_chkpnt" / "model.pt",
        "model": model_dir / "sdh_protonet_best.pth",
        "map": model_dir / "family_map.pth",
        "thresholds": model_dir / "family_thresholds.tsv",
        "ads_detection_threshold": model_dir / "ads_detection_threshold.tsv",
        "ads_detection_cluster_thresholds": model_dir / "ads_detection_cluster_thresholds.tsv",
        "cluster_report": model_dir / "cluster_selection_report.txt",
        "cluster_annotation": first_existing_path(anno_dir / "cluster_annotation.tsv", anno_dir / "cluster_annotation.txt"),
        "ads_function": first_existing_path(anno_dir / "ADS_function.tsv", anno_dir / "ADS_function.txt"),
        "esm2_model": model_dirs["esm2_model"],
        "prost_model": model_dirs["prost_model"],
        "model_root": model_dirs["model_root"],
        "model_dir": model_dir,
    }
    return out


def run_cmd(cmd: List[str], cwd: Path, log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write("\n" + "-" * 72 + "\n")
        log.write("[CMD] " + " ".join(str(x) for x in cmd) + "\n")
        log.write("[CWD] " + str(cwd) + "\n")
        log.flush()
        ret = subprocess.run([str(x) for x in cmd], cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        tail = []
        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            tail = lines[-30:]
        if tail:
            print("Last log lines:")
            for line in tail:
                print("  " + line)
        raise RuntimeError(f"Command failed with code={ret.returncode}: {' '.join(str(x) for x in cmd)}")


def read_fasta_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0]
                if seq_id:
                    ids.append(seq_id)
    if not ids:
        raise ValueError(f"No FASTA IDs found in {path}")
    return ids


def is_missing(value: str) -> bool:
    text = str(value).strip()
    return (not text) or text.lower() in {"nan", "none", "na", "-"}


def normalize_column_name(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def first_existing_column(col_idx: Dict[str, int], aliases: List[str]) -> int:
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in col_idx:
            return col_idx[key]
    return -1


def parse_cluster_label(value: str) -> Optional[int]:
    import re
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


def load_cluster_annotation_summary(path: Path):
    if not path.exists():
        return {}, []
    out: Dict[int, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col_idx = {normalize_column_name(name): idx for idx, name in enumerate(header)}
        label_col = first_existing_column(col_idx, ["label", "family_label", "family_id", "cluster_id"])
        cluster_col = first_existing_column(col_idx, ["cluster_name", "cluster", "family", "family_name"])
        rep_col = first_existing_column(col_idx, ["representative", "rep_name", "representative_id", "rep"])
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
            cluster_name = parts[cluster_col].strip() if cluster_col >= 0 and cluster_col < len(parts) and not is_missing(parts[cluster_col]) else f"cluster_{label}"
            representative = parts[rep_col].strip() if rep_col >= 0 and rep_col < len(parts) and not is_missing(parts[rep_col]) else ""
            funcs = {}
            for col in func_cols:
                idx = header.index(col)
                value = parts[idx].strip() if idx < len(parts) else ""
                funcs[col] = "" if is_missing(value) else value
            out[label] = {"cluster_name": cluster_name, "representative": representative, "funcs": funcs}
    return out, func_cols


def load_ads_function_summary(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col_idx = {normalize_column_name(name): idx for idx, name in enumerate(header)}
        name_col = first_existing_column(col_idx, ["ADS_name", "name", "seq_id", "representative", "rep_name"])
        cluster_col = first_existing_column(col_idx, ["cluster", "cluster_name", "family", "family_id"])
        func_col = first_existing_column(col_idx, ["Against/Function", "against_function", "function", "ads_function", "against"])
        if func_col < 0:
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


def lookup_summary(key: str, mapping: Dict[str, str]) -> str:
    if key in mapping:
        return mapping[key]
    for variant in cluster_name_variants(key):
        if variant in mapping:
            return mapping[variant]
    return ""


def is_dna_sequence(fasta_path: Path) -> bool:
    nuc_chars = set("ATGCUatgcu")
    total = 0
    nuc = 0
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            for ch in line.strip():
                total += 1
                if ch in nuc_chars:
                    nuc += 1
            if total > 10000:
                break
    return total > 0 and nuc / total > 0.9


def load_gene_positions(path: Path) -> Dict[str, Dict[str, int]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                out[parts[0]] = {"gene_start": int(parts[1]), "gene_end": int(parts[2])}
            except ValueError:
                continue
    return out


def add_gene_positions_to_tsv(path: Path, positions: Dict[str, Dict[str, int]]):
    if not path.exists() or not positions:
        return
    rows = list(csv.reader(path.open("r", encoding="utf-8"), delimiter="\t"))
    if not rows:
        return
    header = rows[0]
    if "gene_start" in header:
        return
    try:
        qid_idx = header.index("query_id")
    except ValueError:
        return
    new_rows = [header[:qid_idx + 1] + ["gene_start", "gene_end"] + header[qid_idx + 1:]]
    for row in rows[1:]:
        qid = row[qid_idx] if qid_idx < len(row) else ""
        pos = positions.get(qid, {})
        new_rows.append(row[:qid_idx + 1] + [str(pos.get("gene_start", "")), str(pos.get("gene_end", ""))] + row[qid_idx + 1:])
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(new_rows)


def load_model_evo_dim(map_path: Path) -> int:
    payload = torch.load(map_path, map_location="cpu")
    return int(payload.get("evo_dim", 212))


def load_model_esm_layer(map_path: Path) -> int:
    payload = torch.load(map_path, map_location="cpu")
    value = payload.get("esm_layer", 30)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 30


def analyze_npy(path: Path) -> Dict:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} is not 2D: {arr.shape}")
    nonzero = np.count_nonzero(arr, axis=1)
    row_norm = np.linalg.norm(arr, axis=1)
    zero_rows = int(np.sum(nonzero == 0))
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "dtype": str(arr.dtype),
        "zero_row_count": zero_rows,
        "zero_row_ratio": float(zero_rows / arr.shape[0]) if arr.shape[0] else 0.0,
        "row_norm_min": float(np.min(row_norm)) if arr.shape[0] else 0.0,
        "row_norm_mean": float(np.mean(row_norm)) if arr.shape[0] else 0.0,
        "row_norm_max": float(np.max(row_norm)) if arr.shape[0] else 0.0,
        "nonzero_per_row_mean": float(np.mean(nonzero)) if arr.shape[0] else 0.0,
    }


def count_domtblout_hits(domtblout_path: Path, ids_set: set) -> Dict:
    total = 0
    matched_lines = 0
    matched_ids = set()
    if not domtblout_path.exists():
        return {"total_hit_lines": 0, "matched_query_hits": 0, "matched_query_count": 0}
    with domtblout_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            total += 1
            parts = text.split()
            if len(parts) > 3 and parts[3] in ids_set:
                matched_lines += 1
                matched_ids.add(parts[3])
    return {"total_hit_lines": total, "matched_query_hits": matched_lines, "matched_query_count": len(matched_ids)}


def write_qc(seq_ids: List[str], residue_artifact_dir: Path, hmm_npy: Path, domtblout: Path, txt_path: Path, json_path: Path, model_type: str):
    manifest = residue_artifact_dir / "residue_pair_manifest.tsv"
    ok_count = 0
    total_manifest = 0
    if model_type == "mix" and manifest.exists():
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                total_manifest += 1
                if row.get("status") == "ok":
                    ok_count += 1
    elif model_type in {"ESM2", "ProtT5"}:
        total_manifest = len(seq_ids)
        for seq_id in seq_ids:
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in seq_id.split(";")[0].split("|")[0].split(",")[0].split("/")[0]).strip("._-") or "seq"
            if (residue_artifact_dir / f"{safe}_embedding.pt").exists() or (residue_artifact_dir / f"{safe}.pt").exists():
                ok_count += 1
    hmm_stats = analyze_npy(hmm_npy)
    dom_stats = count_domtblout_hits(domtblout, set(seq_ids))
    report = {
        "input": {
            "sequence_count": len(seq_ids),
            "unique_id_count": len(set(seq_ids)),
            "duplicate_id_count": len(seq_ids) - len(set(seq_ids)),
        },
        "residue_embedding": {
            "model_type": model_type,
            "ok_count": ok_count,
            "manifest_rows": total_manifest,
            "missing_or_error_count": max(0, total_manifest - ok_count),
        },
        "hmmscan": {
            "dom_stats": dom_stats,
            "matched_query_ratio": float(dom_stats["matched_query_count"] / len(seq_ids)) if seq_ids else 0.0,
        },
        "hmm_feature": {"matrix": hmm_stats},
        "quality_flags": {
            "ok_pairing_complete": ok_count == len(seq_ids),
            "ok_hmm_zero_row_ratio_lt_0_5": hmm_stats["zero_row_ratio"] < 0.5,
        },
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["=== QC Report ===", "", "[Input]"]
    lines.append(f"- sequence_count: {len(seq_ids)}")
    lines.append(f"- unique_id_count: {len(set(seq_ids))}")
    lines.append("")
    lines.append("[Residue Embeddings]")
    lines.append(f"- model_type: {model_type}")
    lines.append(f"- ok_count: {ok_count}/{len(seq_ids)}")
    lines.append("")
    lines.append("[HMM Feature Matrix]")
    lines.append(f"- shape: {tuple(hmm_stats['shape'])}, dtype={hmm_stats['dtype']}")
    lines.append(f"- zero rows: {hmm_stats['zero_row_count']} ({hmm_stats['zero_row_ratio']:.2%})")
    lines.append(f"- mean nonzero features: {hmm_stats['nonzero_per_row_mean']:.2f}")
    lines.append("")
    lines.append("[Quality Flags]")
    for key, value in report["quality_flags"].items():
        lines.append(f"- {key}: {'PASS' if value else 'WARN'}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_embedding(paths: Dict[str, Path], script_dir: Path, input_fasta: Path, out_dir: Path, model_path: Path, args, log_file: Path, model_kind: str):
    cmd = [
        sys.executable,
        paths["translate_py"],
        "-i", input_fasta,
        "-o", out_dir,
        "-n", str(args.cpu),
        "--model-path", model_path,
        "--model-kind", model_kind,
        "--cnn-weights", paths["cnn_weights"],
        "--device", args.device,
    ]
    if model_kind == "esm2":
        cmd.extend(["--esm-layer", str(getattr(args, "esm_layer", 30))])
    run_cmd(cmd, cwd=script_dir, log_file=log_file)


def cmd_database(args, script_dir: Path):
    paths = resolve_paths(script_dir, args)
    target = Path(args.download_path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    run_cmd([sys.executable, paths["download_py"], "-o", target], cwd=script_dir, log_file=target / "download.log")
    print(f"Downloaded embedding models to {target}")


def run_self_check(script_dir: Path, args) -> int:
    paths = resolve_paths(script_dir, args)
    required = [
        paths["translate_py"], paths["pair_py"], paths["hmm_to_npy_py"], paths["predict_py"],
        paths["hmm_db"], paths["model"], paths["map"], paths["cluster_annotation"], paths["ads_function"],
    ]
    if args.filter_mode != "none":
        required.append(paths["thresholds"])
    missing = [str(path) for path in required if path is not None and not Path(path).exists()]
    if paths.get("esm2_model") and not paths["esm2_model"].exists():
        missing.append(str(paths["esm2_model"]))
    if paths.get("prost_model") and not paths["prost_model"].exists():
        missing.append(str(paths["prost_model"]))
    if missing:
        print("Self-check FAILED. Missing:")
        for item in missing:
            print("  - " + item)
        return 1
    evo_dim = load_model_evo_dim(paths["map"])
    args.esm_layer = load_model_esm_layer(paths["map"])
    print(f"PHADS {APP_VERSION} self-check OK. model_type={args.model_type}, model_evo_dim={evo_dim}")
    return 0


def cmd_hmm_only(args, paths: Dict[str, Path], script_dir: Path, input_fasta: Path, work_dir: Path, temp_dir: Path, log_file: Path, output_stem: Optional[str] = None):
    stem = f"{output_stem}_" if output_stem else ""
    domtblout = temp_dir / f"{stem}hmm_results.domtblout"
    out_tsv = work_dir / f"{stem}hmm_result.tsv"
    run_cmd(["hmmscan", "--cpu", str(args.cpu), "--domtblout", domtblout, paths["hmm_db"], input_fasta], cwd=script_dir, log_file=log_file)
    cluster_info, func_cols = load_cluster_annotation_summary(paths["cluster_annotation"])
    ads_function = load_ads_function_summary(paths["ads_function"])
    hits: Dict[str, Dict[str, str]] = {}
    with domtblout.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) < 22:
                continue
            query = parts[3]
            try:
                acc = float(parts[21])
            except ValueError:
                continue
            if query not in hits or acc > float(hits[query]["acc"]):
                label = parse_cluster_label(parts[0])
                info = cluster_info.get(label, {}) if label is not None else {}
                cluster_name = str(info.get("cluster_name", f"cluster_{label}" if label is not None else parts[0]))
                representative = str(info.get("representative", ""))
                rep_function = lookup_summary(representative, ads_function) or lookup_summary(cluster_name, ads_function)
                funcs = info.get("funcs", {}) if isinstance(info.get("funcs", {}), dict) else {}
                hits[query] = {
                    "query": query,
                    "target": parts[0],
                    "cluster_name": cluster_name,
                    "label": "" if label is None else str(label),
                    "acc": f"{acc:.4f}",
                    "qstart": parts[17],
                    "qend": parts[18],
                    "tstart": parts[15],
                    "tend": parts[16],
                    "representative": representative,
                    "rep_ads_function": rep_function,
                    "funcs": funcs,
                }
    with out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "query", "target", "label", "cluster_name", "acc", "qstart", "qend", "tstart", "tend",
            "representative", "rep_ads_function", *[f"cluster_func_{name}" for name in func_cols],
        ])
        for query in sorted(hits):
            row = hits[query]
            funcs = row.get("funcs", {})
            writer.writerow([
                row["query"], row["target"], row["label"], row["cluster_name"], row["acc"],
                row["qstart"], row["qend"], row["tstart"], row["tend"],
                row["representative"], row["rep_ads_function"],
                *[funcs.get(name, "") for name in func_cols],
            ])
    print(f"HMM-only output: {out_tsv}")
    return out_tsv


def cmd_foldseek(args, paths: Dict[str, Path], script_dir: Path, input_path: Path, work_dir: Path, temp_dir: Path, log_file: Path, output_stem: Optional[str] = None):
    stem = f"{output_stem}_" if output_stem else ""
    foldseek_tmp = temp_dir / f"{stem}foldseek_tmp"
    out_tsv = work_dir / f"{stem}foldseek_result.tsv"
    run_cmd([sys.executable, paths["structure_compare_py"], "-i", input_path, "-o", foldseek_tmp, "-d", paths["foldseek_db"]], cwd=script_dir, log_file=log_file)
    raw = foldseek_tmp / "result.tsv"
    if not raw.exists():
        raise FileNotFoundError(f"Foldseek raw result missing: {raw}")
    top_hits: Dict[str, Dict[str, str]] = {}
    with raw.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col = {name: idx for idx, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            query = parts[col.get("query", 0)]
            prob = float(parts[col.get("prob", 2)])
            if query not in top_hits or prob > float(top_hits[query]["prob"]):
                top_hits[query] = {"query": query, "target": parts[col.get("target", 1)], "prob": f"{prob:.4f}", "fident": parts[col.get("fident", 3)], "qstart": parts[col.get("qstart", 7)], "qend": parts[col.get("qend", 8)]}
    with out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query", "target", "prob", "fident", "qstart", "qend"])
        for query in sorted(top_hits):
            row = top_hits[query]
            if float(row["prob"]) >= args.prob_threshold:
                writer.writerow([row["query"], row["target"], row["prob"], row["fident"], row["qstart"], row["qend"]])
    print(f"Foldseek output: {out_tsv}")
    return out_tsv


def run_deep_single(args, paths: Dict[str, Path], script_dir: Path, input_fasta: Path, work_dir: Path, temp_dir: Path, output_stem: Optional[str] = None):
    if not args.mode_path:
        raise ValueError("Deep prototype/instance mode requires --db/-db embedding model root")
    if args.model_type in {"ESM2", "mix"} and (not paths["esm2_model"] or not paths["esm2_model"].exists()):
        raise FileNotFoundError(f"ESM2 model directory missing: {paths['esm2_model']}")
    if args.model_type in {"ProtT5", "mix"} and (not paths["prost_model"] or not paths["prost_model"].exists()):
        raise FileNotFoundError(f"ProtT5 model directory missing: {paths['prost_model']}")

    input_is_dna = is_dna_sequence(input_fasta)
    stem = output_stem or input_fasta.stem
    protein_fasta = input_fasta
    pyrodigal_pos = None
    if input_is_dna:
        protein_fasta = temp_dir / f"{stem}_translated.faa"
        pyrodigal_pos = temp_dir / f"{stem}_translated_pos.tsv"
        run_cmd([sys.executable, paths["pyrodigal_py"], "-i", input_fasta, "-a", protein_fasta, "-g", temp_dir / f"{stem}_translated.gff", "-p", pyrodigal_pos], cwd=script_dir, log_file=temp_dir / "pipeline.log")

    prefix = f"{stem}_" if output_stem else ""
    log_file = temp_dir / "pipeline.log"
    ids_path = temp_dir / f"{prefix}protein_ids.txt"
    esm_dir = temp_dir / f"{prefix}esm2_embeddings"
    prost_dir = temp_dir / f"{prefix}prost_embeddings"
    pair_dir = temp_dir / f"{prefix}paired_residue_embeddings"
    domtblout = temp_dir / f"{prefix}hmm_results.domtblout"
    hmm_npy = temp_dir / f"{prefix}query_hmm_features_L2.npy"
    hmm_txt = temp_dir / f"{prefix}query_hmm_features_L2.txt"
    predict_out = work_dir / (f"{stem}_results.tsv" if output_stem else args.predict_output)
    topk_out = work_dir / (f"{stem}_prediction_topk.tsv" if output_stem else args.predict_topk_output)

    seq_ids = read_fasta_ids(protein_fasta)
    ids_path.write_text("\n".join(seq_ids) + "\n", encoding="utf-8")
    evo_dim = load_model_evo_dim(paths["map"])
    args.esm_layer = load_model_esm_layer(paths["map"])

    total_steps = (7 if args.model_type == "mix" else 5) + (1 if input_is_dna else 0)
    step = 0
    if input_is_dna:
        step += 1
        print_step(step, total_steps, "Gene prediction", "done", f"{len(seq_ids)} proteins")

    single_residue_dir = esm_dir
    if args.model_type in {"ESM2", "mix"}:
        step += 1
        t0 = time.time()
        print_step(step, total_steps, f"Generate ESM2 residue embeddings (layer {args.esm_layer})")
        run_embedding(paths, script_dir, protein_fasta, esm_dir, paths["esm2_model"], args, log_file, "esm2")
        print_step(step, total_steps, f"Generate ESM2 residue embeddings (layer {args.esm_layer})", "done", f"{time.time() - t0:.1f}s")
        single_residue_dir = esm_dir

    if args.model_type in {"ProtT5", "mix"}:
        step += 1
        t0 = time.time()
        print_step(step, total_steps, "Generate ProtT5 residue embeddings")
        run_embedding(paths, script_dir, protein_fasta, prost_dir, paths["prost_model"], args, log_file, "prostt5_onnx")
        print_step(step, total_steps, "Generate ProtT5 residue embeddings", "done", f"{time.time() - t0:.1f}s")
        if args.model_type == "ProtT5":
            single_residue_dir = prost_dir

    step += 1
    t0 = time.time()
    print_step(step, total_steps, "Run HMM alignment")
    run_cmd(["hmmscan", "--cpu", str(args.cpu), "--domtblout", domtblout, paths["hmm_db"], protein_fasta], cwd=script_dir, log_file=log_file)
    print_step(step, total_steps, "Run HMM alignment", "done", f"{time.time() - t0:.1f}s")

    step += 1
    t0 = time.time()
    print_step(step, total_steps, "Build HMM feature matrix")
    run_cmd([sys.executable, paths["hmm_to_npy_py"], "-d", domtblout, "-i", protein_fasta, "-k", str(evo_dim), "-on", hmm_npy, "-ot", hmm_txt], cwd=script_dir, log_file=log_file)
    print_step(step, total_steps, "Build HMM feature matrix", "done", f"{time.time() - t0:.1f}s")

    if args.model_type == "mix":
        step += 1
        t0 = time.time()
        print_step(step, total_steps, "Pair residue embeddings")
        pair_cmd = [sys.executable, paths["pair_py"], "--ids", ids_path, "--esm-root", esm_dir, "--prost-root", prost_dir, "--output-dir", pair_dir, "--esm-dim", "1280", "--prost-dim", "1024", "--dtype", "float16"]
        if args.allow_length_mismatch or not args.strict_residue_length:
            pair_cmd.append("--allow-length-mismatch")
        run_cmd(pair_cmd, cwd=script_dir, log_file=log_file)
        print_step(step, total_steps, "Pair residue embeddings", "done", f"{time.time() - t0:.1f}s")

    step += 1
    t0 = time.time()
    print_step(step, total_steps, f"Run {args.model_type} residue Pool-Gate prediction")
    predict_cmd = [
        sys.executable, paths["predict_py"],
        "--model-type", args.model_type,
        "--model-path", paths["model"],
        "--map-path", paths["map"],
        "--query-hmm", hmm_npy,
        "--query-ids", ids_path,
        "--threshold-tsv", paths["thresholds"],
        "--ads-detection-threshold-tsv", paths["ads_detection_threshold"],
        "--ads-detection-cluster-threshold-tsv", paths["ads_detection_cluster_thresholds"],
        "--ads-detection-threshold-mode", args.ads_detection_threshold_mode,
        "--cluster-annotation-txt", paths["cluster_annotation"],
        "--ads-function-txt", paths["ads_function"],
        "--mode", args.predict_mode,
        "--filter-mode", args.filter_mode,
        "--topk", str(args.topk),
        "--topk-hmm", str(args.topk),
        "--judge-mode", args.judge_mode,
        "--batch-size", str(args.residue_batch_size),
        "--device", args.device,
        "--output-tsv", predict_out,
        "--topk-tsv", topk_out,
    ]
    if args.model_type == "mix":
        predict_cmd.extend(["--pair-dir", pair_dir])
    else:
        predict_cmd.extend(["--residue-dir", single_residue_dir])
    if args.use_multiprototype:
        predict_cmd.append("--use-multiprototype")
    if args.print_topk:
        predict_cmd.append("--print-topk")
    run_cmd(predict_cmd, cwd=script_dir, log_file=log_file)
    print_step(step, total_steps, f"Run {args.model_type} residue Pool-Gate prediction", "done", f"{time.time() - t0:.1f}s")

    step += 1
    t0 = time.time()
    print_step(step, total_steps, "Write QC report")
    qc_txt = work_dir / (f"{stem}_qc_report.txt" if output_stem else args.qc_txt)
    qc_json = work_dir / (f"{stem}_qc_report.json" if output_stem else args.qc_json)
    write_qc(seq_ids, pair_dir if args.model_type == "mix" else single_residue_dir, hmm_npy, domtblout, qc_txt, qc_json, args.model_type)
    if input_is_dna and pyrodigal_pos:
        positions = load_gene_positions(pyrodigal_pos)
        add_gene_positions_to_tsv(predict_out, positions)
        add_gene_positions_to_tsv(topk_out, positions)
    print_step(step, total_steps, "Write QC report", "done", f"{time.time() - t0:.1f}s")

    return predict_out, topk_out


def merge_tsv(paths: List[Path], out_path: Path):
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    with out_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.writer(out_handle, delimiter="\t")
        header = None
        for path in existing:
            with path.open("r", encoding="utf-8") as in_handle:
                reader = csv.reader(in_handle, delimiter="\t")
                for row in reader:
                    if not row:
                        continue
                    if header is None:
                        header = row
                        writer.writerow(row)
                    elif row == header:
                        continue
                    else:
                        writer.writerow(row)


def cmd_dir(args, script_dir: Path):
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    fasta_exts = {".fasta", ".faa", ".fa", ".fna", ".ffn", ".frn", ".fas"}
    fasta_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in fasta_exts)
    if not fasta_files:
        raise ValueError(f"No FASTA files found in {input_dir}")
    paths = resolve_paths(script_dir, args)
    work_dir = (Path(args.work_dir) if Path(args.work_dir).is_absolute() else Path.cwd() / args.work_dir).resolve()
    temp_dir = resolve_temp_dir(args.temp_dir, work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    print_header([("Input directory", str(input_dir)), ("Output directory", str(work_dir)), ("Files", str(len(fasta_files))), ("Mode", args.predict_mode)])
    result_paths: List[Path] = []
    topk_paths: List[Path] = []
    for idx, fasta in enumerate(fasta_files, 1):
        print(f"\n=== [{idx}/{len(fasta_files)}] {fasta.name} ===")
        if args.predict_mode == "foldseek":
            result_paths.append(cmd_foldseek(args, paths, script_dir, fasta, work_dir, temp_dir, temp_dir / "pipeline.log", output_stem=fasta.stem))
        elif args.predict_mode == "hmm":
            result_paths.append(cmd_hmm_only(args, paths, script_dir, fasta, work_dir, temp_dir, temp_dir / "pipeline.log", output_stem=fasta.stem))
        else:
            pred, topk = run_deep_single(args, paths, script_dir, fasta, work_dir, temp_dir, output_stem=fasta.stem)
            result_paths.append(pred)
            topk_paths.append(topk)
    if args.predict_mode in {"prototype", "instance"}:
        merge_tsv(result_paths, work_dir / "all_results.tsv")
        topk_dir = work_dir / "topk"
        topk_dir.mkdir(exist_ok=True)
        merge_tsv(topk_paths, topk_dir / "all_prediction_topk.tsv")
        for path in topk_paths:
            if path.exists():
                dest = topk_dir / path.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(path), str(dest))
    elif args.predict_mode == "hmm":
        merge_tsv(result_paths, work_dir / "all_hmm_result.tsv")
    elif args.predict_mode == "foldseek":
        merge_tsv(result_paths, work_dir / "all_foldseek_result.tsv")
    print(f"Batch complete: {work_dir}")


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.command == "database":
        cmd_database(args, script_dir)
        return
    if args.version:
        raise SystemExit(run_self_check(script_dir, args))
    if args.input_dir:
        cmd_dir(args, script_dir)
        return
    if not args.input_fasta:
        raise ValueError("Provide -i/--input-fasta or -d/--dir")

    input_fasta = Path(args.input_fasta).resolve()
    if not input_fasta.exists():
        raise FileNotFoundError(input_fasta)
    paths = resolve_paths(script_dir, args)
    work_dir = (Path(args.work_dir) if Path(args.work_dir).is_absolute() else Path.cwd() / args.work_dir).resolve()
    temp_dir = resolve_temp_dir(args.temp_dir, work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_file = temp_dir / "pipeline.log"

    print_header([
        ("Input", str(input_fasta)),
        ("Output", str(work_dir)),
        ("Temp", str(temp_dir)),
        ("Mode", args.predict_mode),
        ("Model type", args.model_type),
        ("Model", str(paths["model"])),
        ("Family map", str(paths["map"])),
        ("HMM DB", str(paths["hmm_db"])),
        ("Embedding model root", str(paths.get("model_root"))),
        ("Filter", args.filter_mode),
        ("ADS threshold mode", args.ads_detection_threshold_mode),
        ("TopK", str(args.topk)),
    ])

    if args.predict_mode == "foldseek":
        cmd_foldseek(args, paths, script_dir, input_fasta, work_dir, temp_dir, log_file)
    elif args.predict_mode == "hmm":
        cmd_hmm_only(args, paths, script_dir, input_fasta, work_dir, temp_dir, log_file)
    else:
        run_deep_single(args, paths, script_dir, input_fasta, work_dir, temp_dir)
    print(f"\nDONE. Output directory: {work_dir}")


if __name__ == "__main__":
    main()
