#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主流程脚本：自动生成预测所需 .npy 并直接调用 predict.py（支持多序列 FASTA）

默认自动路径（相对 main.py 所在目录）：
- HMM DB: database/hmm_model/anti_defense_system.hmm
- CNN 权重: database/cnn_chkpnt/model.pt
- Single PhADS model: database/PhADS_model/one_model/sdh_protonet_best.pth
- Single PhADS map: database/PhADS_model/one_model/family_map.pth
- Single PhADS thresholds: database/PhADS_model/one_model/family_thresholds.tsv
- Mix PhADS model: database/PhADS_model/mix_model/sdh_protonet_crossattn_best.pth
- Mix PhADS map: database/PhADS_model/mix_model/family_map.pth
- Mix PhADS thresholds: database/PhADS_model/mix_model/family_thresholds.tsv
- 注释库: database/anno_database/{cluster_annotation.txt, ADS_function.txt}

说明：
- ESM2 / ProtT5 下载模型根目录通过 `-db/--mode-path` 指定
- 该目录应包含 esm2_t33_650M_UR50D 和 prot-t5-xl-uniref50-enc-onnx 两个子目录
- 不再依赖 cluster_merge_final.txt

临时目录规则：
- 指定 -temp/--temp：使用指定目录
- 未指定：在调用脚本的当前目录创建 temp/

query 相关文件统一写入 temp 目录：
- protein_ids.txt
- query_embeddings.npy
- hmm_results.domtblout
- query_hmm_features_L2.npy
- query_hmm_features_L2.txt
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


APP_VERSION = "0.1"
ESM2_MODEL_DIRNAME = "esm2_t33_650M_UR50D"
PROST_MODEL_DIRNAME = "prot-t5-xl-uniref50-enc-onnx"
DEFAULT_ESM2_DIM = 1280
DEFAULT_PROST_DIM = 1024


def parse_args():
    # ── 兼容 -db → --db（argparse 将 -db 拆为 -d -b）──
    _new_argv = [sys.argv[0]]
    for a in sys.argv[1:]:
        if a == "-db":
            _new_argv.append("--db")
        else:
            _new_argv.append(a)
    sys.argv = _new_argv

    p = argparse.ArgumentParser(
        description=(
            "PhADS end-to-end inference pipeline: embedding generation, HMM feature extraction, and prediction.\n\n"
            "Output schema:\n"
            "  1) --predict-output (TSV)\n"
            "     query_id, [gene_start, gene_end (if DNA input)], mode, pred_cluster, pred_cluster_rep,\n"
            "     pred_distance2, confidence, filter_mode, filter_status, threshold_limit,\n"
            "     nearest_sequence_id, nearest_sequence_label, nearest_sequence_distance2,\n"
            "     nearest_sequence_function_summary, cluster_func_*\n\n"
            "  2) --predict-topk-output (TSV)\n"
            "     query_id, [gene_start, gene_end (if DNA input)], mode, rank, candidate_id, candidate_cluster, candidate_rep,\n"
            "     candidate_distance2, candidate_confidence, filter_mode, filter_status,\n"
            "     threshold_limit, candidate_ads_function"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Subcommands ──
    sub = p.add_subparsers(dest="command", help="Available subcommands")

    db_parser = sub.add_parser("database", help="Download ESM2 and ProtT5 models to a specified directory")
    db_parser.add_argument(
        "--path", "-P", required=True, dest="download_path",
        help="Target root directory to download esm2_t33_650M_UR50D and prot-t5-xl-uniref50-enc-onnx into."
    )

    # ── Pipeline arguments (used when command is None) ──
    p.add_argument("-i", "--input-fasta", required=False, help="Input FASTA file (single or multiple sequences).")
    p.add_argument("-d", "--dir", dest="input_dir", required=False,
                   help="Input directory containing multiple FASTA files for batch processing. "
                        "Each file is processed independently; individual and merged results are generated.")
    p.add_argument("-o", "--work-dir", default="run_auto", help="Final output directory for prediction and QC reports.")
    p.add_argument("-n", "--cpu", type=int, default=8, help="Number of CPU threads for embedding and hmmscan.")
    p.add_argument("--device", default="auto", help="Embedding device: auto | cpu | cuda.")

    p.add_argument(
        "-temp", "--temp", dest="temp_dir", default=None,
        help="Temporary directory for intermediate query files (.pt/.npy/.domtblout). Defaults to ./temp."
    )

    p.add_argument("--db", "--mode-path", dest="mode_path", required=False,
                   help="Path to downloaded embedding model root containing esm2_t33_650M_UR50D and prot-t5-xl-uniref50-enc-onnx.")

    p.add_argument("--predict-mode", choices=["prototype", "instance", "foldseek", "hmm"], default="prototype",
                   help="Prediction mode: prototype | instance | foldseek (structural) | hmm (HMM-only).")
    p.add_argument("--model-type", choices=["single", "mix"], default="single",
                   help="PhADS deep model type for prototype/instance mode: single | mix.")
    p.add_argument("--query-emb-esm2", default=None,
                   help="Mix mode only: prebuilt ESM2 query embedding .npy. If omitted, main.py generates it from --db/esm2_t33_650M_UR50D.")
    p.add_argument("--esm2-emb-dir", default=None,
                   help="Mix mode only: directory containing per-sequence ESM2 *_embedding.pt files to build query_embeddings_esm2.npy.")
    p.add_argument("--prob-threshold", "-prob", dest="prob_threshold", type=float, default=0.8,
                   help="Foldseek mode: minimum prob score to retain a hit (default: 0.8).")
    p.add_argument("--topk", type=int, default=5, help="Number of top candidates to report.")
    p.add_argument("--predict-output", default="prediction_results.tsv", help="Filename of the main prediction TSV in work-dir.")
    p.add_argument("--predict-topk-output", default="prediction_topk.tsv", help="Filename of the Top-k TSV in work-dir.")
    p.add_argument("--print-topk", action="store_true", help="Print Top-k candidates to stdout.")

    # 🌟 新增参数：过滤控制模式 (默认值为 moderate)
    p.add_argument(
        "--filter-mode",
        choices=["strict", "moderate", "loose", "none"],
        default="moderate",
        help=(
            "Control mode for filtering based on squared latent space distance thresholds.\n"
            "  strict   : Tight firewall limit (P90 training bounds, minimizes false positives).\n"
            "  moderate : Balanced firewall limit (P95 training bounds, recommended standard mode).\n"
            "  loose    : Relaxed firewall limit (Outlier boundaries, aims for divergent discovery).\n"
            "  none     : Disable online filtering (Backward compatible, preserves raw results)."
        )
    )

    p.add_argument("--qc-txt", default="qc_report.txt", help="Filename of the plain-text QC report in work-dir.")
    p.add_argument("--qc-json", default="qc_report.json", help="Filename of the JSON QC report in work-dir.")

    p.add_argument(
        "-v", "--version", action="store_true",
        help="Run database integrity self-check. Print version if all required files are present."
    )

    return p.parse_args()


def safe_file_stem(seq_id: str) -> str:
    primary = re.split(r"[;|/,]", seq_id, maxsplit=1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", primary)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned if cleaned else "seq"


# ═══════════════════════════════════════════════════════════════
#  DNA / Protein detection
# ═══════════════════════════════════════════════════════════════

def is_dna_sequence(fasta_path: Path) -> bool:
    """检测 FASTA 文件是 DNA 还是蛋白质序列。

    通过统计前 10000 个有效字符中核苷酸 (ATGCU) 的比例来判断。
    若核苷酸比例 > 90%，则判定为 DNA；否则为蛋白质。
    """
    nuc_chars = set("ATGCUatgcu")
    total_nuc = 0
    total_chars = 0

    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                continue
            for c in line.strip():
                total_chars += 1
                if c in nuc_chars:
                    total_nuc += 1
            if total_chars > 10000:  # 采样前 10000 个字符
                break

    if total_chars == 0:
        return False

    nuc_ratio = total_nuc / total_chars
    return nuc_ratio > 0.9


def load_gene_positions(pos_path: Path) -> Dict[str, Dict[str, int]]:
    """加载基因位置映射文件 (gene_id → {gene_start, gene_end})。"""
    positions: Dict[str, Dict[str, int]] = {}
    if not pos_path.exists():
        return positions

    with open(pos_path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gene_id = parts[0]
                try:
                    positions[gene_id] = {
                        "gene_start": int(parts[1]),
                        "gene_end": int(parts[2]),
                    }
                except ValueError:
                    continue
    return positions


def add_gene_positions_to_tsv(
    tsv_path: Path, gene_positions: Dict[str, Dict[str, int]], is_main_output: bool = True
):
    """将基因位置列（gene_start, gene_end）插入到预测结果 TSV 中 query_id 之后。

    Args:
        tsv_path: 预测结果 TSV 文件路径
        gene_positions: gene_id → {gene_start, gene_end} 映射
        is_main_output: True 表示主预测结果，False 表示 Top-k 明细
    """
    if not tsv_path.exists():
        return

    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return

    # 解析表头
    header = lines[0].rstrip("\n").split("\t")
    if "gene_start" in header or "gene_end" in header:
        return  # 已经包含基因位置列，跳过

    # 找到 query_id 列索引
    try:
        qid_idx = header.index("query_id")
    except ValueError:
        return

    # 在 query_id 之后插入 gene_start, gene_end
    new_header = header[: qid_idx + 1] + ["gene_start", "gene_end"] + header[qid_idx + 1 :]
    new_lines = ["\t".join(new_header) + "\n"]

    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < qid_idx + 1:
            new_lines.append(line)
            continue

        qid = parts[qid_idx]
        pos = gene_positions.get(qid, {})
        gs = str(pos.get("gene_start", ""))
        ge = str(pos.get("gene_end", ""))

        new_parts = parts[: qid_idx + 1] + [gs, ge] + parts[qid_idx + 1 :]
        new_lines.append("\t".join(new_parts) + "\n")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# ═══════════════════════════════════════════════════════════════
#  Clean output helpers
# ═══════════════════════════════════════════════════════════════

def _box_width() -> int:
    return 62


def print_header(config_items: List[tuple]):
    """Print a clean header box with configuration."""
    w = _box_width()
    title = f"  PhADS v{APP_VERSION} — Phage Anti-Defense System Inference Pipeline"
    print()
    print("╔" + "═" * w + "╗")
    print(f"║{title:<{w}}║")
    print("╚" + "═" * w + "╝")
    print()
    print("  ── Configuration " + "─" * (w - 19))
    for key, value in config_items:
        print(f"  {key:<22} {value}")
    print("  " + "─" * (w - 2))
    print()


def print_download_header(path: str):
    """Print header for download mode."""
    w = _box_width()
    title = f"  PhADS v{APP_VERSION} — Download Embedding Models"
    print()
    print("╔" + "═" * w + "╗")
    print(f"║{title:<{w}}║")
    print("╚" + "═" * w + "╝")
    print()
    print("  ── Download " + "─" * (w - 15))
    print(f"  {'Target directory':<22} {path}")
    print("  " + "─" * (w - 2))
    print()


def print_progress(step: int, total: int, name: str, status: str = "start", detail: str = ""):
    """Print a progress step with start/done/skip status."""
    w = _box_width()
    if status == "start":
        line = f"  [{step}/{total}] {name}..."
        print(f"{line:<{w-2}}", end="", flush=True)
    elif status == "done":
        detail_str = f" ({detail})" if detail else ""
        line = f"  [{step}/{total}] ✓ {name}{detail_str}"
        print(f"\r{line:<{w-2}}")
    elif status == "skip":
        line = f"  [{step}/{total}] → {name} (skipped)"
        print(f"{line:<{w-2}}")


def print_summary(title: str, items: List[tuple]):
    """Print a results summary section."""
    w = _box_width()
    print()
    header = f"  ── {title} "
    print(header + "─" * max(0, w - len(header) - 2))
    for key, value in items:
        print(f"  {key:<22} {value}")
    print("  " + "─" * (w - 2))
    print()


def print_done_box():
    """Print final completion box."""
    w = _box_width()
    line = "  ✓ Pipeline completed successfully!"
    print("╔" + "═" * w + "╗")
    print(f"║{line:<{w}}║")
    print("╚" + "═" * w + "╝")
    print()


# ═══════════════════════════════════════════════════════════════


def read_fasta_ids(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                sid = line[1:].strip().split()[0]
                if sid:
                    ids.append(sid)
    if not ids:
        raise ValueError(f"未在 FASTA 中读取到序列 ID: {path}")
    return ids


def run_cmd(cmd: List[str], cwd: Path, log_file: Optional[Path] = None):
    """Run a command, redirecting stdout/stderr to a log file for clean output."""
    if log_file is None:
        log_file = cwd / "pipeline.log"
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n{'─' * 60}\n")
        log.write(f"[CMD] {' '.join(cmd)}\n")
        log.write(f"[CWD] {cwd}\n")
        log.write(f"{'─' * 60}\n")
        log.flush()
        ret = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        # Print last 30 lines of log for debugging
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as log:
                lines = log.readlines()
                print("\n  ⚠ Error log (last 30 lines):")
                for line in lines[-30:]:
                    print(f"    {line.rstrip()}")
        raise RuntimeError(f"命令执行失败 (code={ret.returncode}): {' '.join(cmd)}")


def build_embedding_npy(seq_ids: List[str], emb_dir: Path, out_npy: Path, emb_dim: int = 1024) -> Dict:
    vecs = []
    missing = []

    for sid in seq_ids:
        f = emb_dir / f"{safe_file_stem(sid)}_embedding.pt"
        if not f.exists():
            missing.append(sid)
            vecs.append(np.zeros((emb_dim,), dtype=np.float32))
            continue

        t = torch.load(f, map_location="cpu")
        if isinstance(t, dict):
            if "embedding" in t:
                t = t["embedding"]
            else:
                raise ValueError(f"不支持的 embedding 文件结构: {f}")

        if t.dim() == 2:
            v = t.float().mean(dim=0).numpy()
        elif t.dim() == 1:
            v = t.float().numpy()
        else:
            raise ValueError(f"embedding 维度异常: {f}, shape={tuple(t.shape)}")

        if v.shape[0] != emb_dim:
            raise ValueError(f"embedding 维度不匹配: {f}, got={v.shape[0]}, expected={emb_dim}")

        vecs.append(v.astype(np.float32))

    arr = np.stack(vecs, axis=0)
    np.save(out_npy, arr)

    if missing:
        miss_log = out_npy.parent / "missing_embedding_ids.log"
        miss_log.write_text("\n".join(missing), encoding="utf-8")
        print(f"\n  ⚠ Warning: {len(missing)} sequences missing embedding, zero-filled. See: {miss_log}")

    return {
        "missing_count": len(missing),
        "missing_ids": missing,
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
    }


def run_embedding_generation(paths: Dict, script_dir: Path, input_fasta: Path, output_dir: Path,
                             model_path: Path, args, log_file: Path):
    run_cmd([
        sys.executable,
        str(paths["translate_py"]),
        "-i", str(input_fasta),
        "-o", str(output_dir),
        "-n", str(args.cpu),
        "--model-path", str(model_path),
        "--model-kind", "auto",
        "--cnn-weights", str(paths["cnn_weights"]),
        "--device", str(args.device),
    ], cwd=script_dir, log_file=log_file)


def resolve_mix_esm2_embedding(args, seq_ids: List[str], temp_dir: Path, output_stem: Optional[str] = None,
                               expected_dim: int = DEFAULT_ESM2_DIM) -> tuple:
    out_name = f"{output_stem}_query_embeddings_esm2.npy" if output_stem else "query_embeddings_esm2.npy"
    default_npy = temp_dir / out_name

    if args.query_emb_esm2:
        emb_arg = Path(args.query_emb_esm2)
        candidate = emb_arg / out_name if emb_arg.is_dir() else emb_arg
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Mix 模式找不到 ESM2 query embedding: {candidate}")
        arr = analyze_npy(candidate)
        if arr["shape"][0] != len(seq_ids):
            raise ValueError(f"ESM2 query embedding 样本数不一致: got={arr['shape'][0]}, expect={len(seq_ids)}")
        if arr["shape"][1] != expected_dim:
            raise ValueError(f"ESM2 query embedding 维度不一致: got={arr['shape'][1]}, expect={expected_dim}")
        return candidate, {"source": "prebuilt_npy", "shape": arr["shape"]}

    if args.esm2_emb_dir:
        emb_dir = Path(args.esm2_emb_dir).resolve()
        if output_stem and (emb_dir / output_stem).is_dir():
            emb_dir = emb_dir / output_stem
        if not emb_dir.is_dir():
            raise NotADirectoryError(f"Mix 模式 ESM2 embedding 目录不存在: {emb_dir}")
        stats = build_embedding_npy(seq_ids=seq_ids, emb_dir=emb_dir, out_npy=default_npy, emb_dim=expected_dim)
        stats["source"] = "pt_dir"
        return default_npy.resolve(), stats

    if default_npy.exists():
        arr = analyze_npy(default_npy)
        if arr["shape"][0] != len(seq_ids):
            raise ValueError(f"默认 ESM2 query embedding 样本数不一致: got={arr['shape'][0]}, expect={len(seq_ids)}")
        if arr["shape"][1] != expected_dim:
            raise ValueError(f"默认 ESM2 query embedding 维度不一致: got={arr['shape'][1]}, expect={expected_dim}")
        return default_npy.resolve(), {"source": "temp_default", "shape": arr["shape"]}

    raise FileNotFoundError(
        "Mix 模式需要 ESM2 query embedding。请提供 --query-emb-esm2 <npy>，"
        "或提供 --esm2-emb-dir <包含 *_embedding.pt 的目录>，"
        f"或预先生成默认文件: {default_npy}"
    )


def count_domtblout_hits(domtblout_path: Path, seq_ids_set: set) -> Dict:
    total_hit_lines = 0
    matched_query_hits = 0
    matched_queries = set()

    if not domtblout_path.exists():
        return {
            "total_hit_lines": 0,
            "matched_query_hits": 0,
            "matched_query_count": 0,
        }

    with domtblout_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            total_hit_lines += 1
            parts = s.split()
            if len(parts) > 3:
                qid = parts[3]
                if qid in seq_ids_set:
                    matched_query_hits += 1
                    matched_queries.add(qid)

    return {
        "total_hit_lines": int(total_hit_lines),
        "matched_query_hits": int(matched_query_hits),
        "matched_query_count": int(len(matched_queries)),
    }


def analyze_npy(path: Path) -> Dict:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path.name} 不是二维矩阵，shape={arr.shape}")

    row_norm = np.linalg.norm(arr, axis=1)
    nonzero = np.count_nonzero(arr, axis=1)
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


def make_qc_report(seq_ids: List[str], emb_stats: Dict, dom_stats: Dict, emb_npy_path: Path, hmm_npy_path: Path) -> Dict:
    total = len(seq_ids)
    unique_count = len(set(seq_ids))
    duplicate_count = total - unique_count

    emb_mat = analyze_npy(emb_npy_path)
    hmm_mat = analyze_npy(hmm_npy_path)

    matched_query_count = dom_stats["matched_query_count"]
    hmm_match_ratio = (matched_query_count / total) if total else 0.0

    return {
        "input": {
            "sequence_count": int(total),
            "unique_id_count": int(unique_count),
            "duplicate_id_count": int(duplicate_count),
        },
        "embedding": {
            "missing_embedding_count": int(emb_stats.get("missing_count", 0)),
            "matrix": emb_mat,
        },
        "hmmscan": {
            "dom_stats": dom_stats,
            "matched_query_ratio": float(hmm_match_ratio),
        },
        "hmm_feature": {
            "matrix": hmm_mat,
        },
        "quality_flags": {
            "ok_embedding_missing_ratio_lt_0_2": emb_stats.get("missing_count", 0) / total < 0.2 if total else True,
            "ok_hmmscan_match_ratio_gt_0_5": hmm_match_ratio > 0.5 if total else True,
            "ok_hmm_zero_row_ratio_lt_0_5": hmm_mat["zero_row_ratio"] < 0.5,
        },
    }


def write_qc_reports(report: Dict, txt_path: Path, json_path: Path):
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("=== 自动质检报告 (QC) ===")
    lines.append("")
    lines.append("[输入]")
    lines.append(f"- 序列总数: {report['input']['sequence_count']}")
    lines.append(f"- 唯一ID数: {report['input']['unique_id_count']}")
    lines.append(f"- 重复ID数: {report['input']['duplicate_id_count']}")
    lines.append("")

    lines.append("[Embedding]")
    lines.append(f"- 缺失 embedding 数: {report['embedding']['missing_embedding_count']}")
    em = report['embedding']['matrix']
    lines.append(f"- 矩阵: shape={tuple(em['shape'])}, dtype={em['dtype']}")
    lines.append(f"- 零行: {em['zero_row_count']} ({em['zero_row_ratio']:.2%})")
    lines.append(f"- 行范数(min/mean/max): {em['row_norm_min']:.4f} / {em['row_norm_mean']:.4f} / {em['row_norm_max']:.4f}")
    lines.append("")

    lines.append("[hmmscan]")
    hs = report['hmmscan']['dom_stats']
    lines.append(f"- domtblout 命中行数: {hs['total_hit_lines']}")
    lines.append(f"- 命中输入序列的行数: {hs['matched_query_hits']}")
    lines.append(f"- 至少命中1次的输入序列数: {hs['matched_query_count']} ({report['hmmscan']['matched_query_ratio']:.2%})")
    lines.append("")

    lines.append("[HMM特征矩阵]")
    hm = report['hmm_feature']['matrix']
    lines.append(f"- 矩阵: shape={tuple(hm['shape'])}, dtype={hm['dtype']}")
    lines.append(f"- 零行: {hm['zero_row_count']} ({hm['zero_row_ratio']:.2%})")
    lines.append(f"- 行范数(min/mean/max): {hm['row_norm_min']:.4f} / {hm['row_norm_mean']:.4f} / {hm['row_norm_max']:.4f}")
    lines.append(f"- 每行非零特征平均数: {hm['nonzero_per_row_mean']:.2f}")
    lines.append("")

    lines.append("[质量标记]")
    for k, v in report['quality_flags'].items():
        lines.append(f"- {k}: {'PASS' if v else 'WARN'}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_temp_dir(temp_arg: str | None) -> Path:
    if temp_arg:
        p = Path(temp_arg)
        return (p if p.is_absolute() else (Path.cwd() / p)).resolve()
    return (Path.cwd() / "temp").resolve()


def resolve_downloaded_model_dirs(mode_path: str | None) -> Dict[str, Optional[Path]]:
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
        esm2_model = model_root / ESM2_MODEL_DIRNAME
        prost_model = model_root / PROST_MODEL_DIRNAME

    return {"model_root": model_root, "esm2_model": esm2_model, "prost_model": prost_model}


def load_family_map_dims(map_path: Path) -> Dict[str, Optional[int]]:
    if not map_path.exists():
        return {}
    payload = torch.load(map_path, map_location="cpu")
    out: Dict[str, Optional[int]] = {}
    for key in ("seq_dim", "dim_esm2", "dim_prost", "evo_dim", "latent_dim"):
        value = payload.get(key) if isinstance(payload, dict) else None
        out[key] = int(value) if value is not None else None
    return out


def infer_single_embedding_kind(seq_dim: Optional[int]) -> str:
    return "esm2"


def embedding_model_for_kind(model_dirs: Dict[str, Optional[Path]], kind: str) -> Optional[Path]:
    return model_dirs["prost_model"] if kind.startswith("prost") else model_dirs["esm2_model"]


def resolve_auto_paths(script_dir: Path, args):
    db_dir = script_dir / "database"
    phads_model_dir = db_dir / "PhADS_model"
    single_model_dir = phads_model_dir / "one_model"
    mix_model_dir = phads_model_dir / "mix_model"
    use_mix_model = getattr(args, "model_type", "single") == "mix"
    selected_model_dir = mix_model_dir if use_mix_model else single_model_dir
    selected_model_name = "sdh_protonet_crossattn_best.pth" if use_mix_model else "sdh_protonet_best.pth"
    selected_map = selected_model_dir / "family_map.pth"
    selected_map_dims = load_family_map_dims(selected_map)
    model_dirs = resolve_downloaded_model_dirs(args.mode_path)
    single_embedding_dim = selected_map_dims.get("seq_dim") or DEFAULT_ESM2_DIM
    single_embedding_kind = infer_single_embedding_kind(single_embedding_dim)
    mix_esm2_dim = selected_map_dims.get("dim_esm2") or DEFAULT_ESM2_DIM
    mix_prost_dim = selected_map_dims.get("dim_prost") or DEFAULT_PROST_DIM
    selected_threshold = selected_model_dir / "family_thresholds.tsv"

    out = {
        "translate_py": script_dir / "scripts" / "translate_to_embedding.py",
        "hmm_to_npy_py": script_dir / "scripts" / "hmm_to_npy.py",
        "predict_py": script_dir / "scripts" / "predict.py",
        "predict_mix_py": script_dir / "scripts" / "predict_mix.py",
        "pyrodigal_gv_py": script_dir / "scripts" / "pyrodigal_viral.py",
        "structure_compare_py": script_dir / "scripts" / "structure_compare.py",
        "foldseek_db": (db_dir / "foldseek_db" / "phads_db").resolve(),
        "hmm_db": (db_dir / "hmm_model" / "anti_defense_system.hmm").resolve(),
        "model_root": model_dirs["model_root"],
        "esm2_model": model_dirs["esm2_model"],
        "prost_model": model_dirs["prost_model"],
        "single_embedding_kind": single_embedding_kind,
        "single_embedding_dim": single_embedding_dim,
        "single_embedding_model": embedding_model_for_kind(model_dirs, single_embedding_kind),
        "mix_esm2_dim": mix_esm2_dim,
        "mix_prost_dim": mix_prost_dim,
        "cnn_weights": (db_dir / "cnn_chkpnt" / "model.pt").resolve(),
        "predict_script": (script_dir / "scripts" / ("predict_mix.py" if use_mix_model else "predict.py")).resolve(),
        "predict_model": (selected_model_dir / selected_model_name).resolve(),
        "predict_map": selected_map.resolve(),
        # 🌟 按照要求：无缝集成 family_thresholds.tsv 路径到自动路径树中
        "family_thresholds": selected_threshold.resolve(),
        "cluster_annotation": (db_dir / "anno_database" / "cluster_annotation.txt").resolve(),
        "ads_function": (db_dir / "anno_database" / "ADS_function.txt").resolve(),
    }

    for k, v in out.items():
        if k.endswith("_py"):
            if k == "predict_mix_py" and not use_mix_model:
                continue
            if not v.exists():
                raise FileNotFoundError(f"找不到脚本: {v}")
        else:
            if k in ("single_embedding_kind", "single_embedding_dim", "mix_esm2_dim", "mix_prost_dim"):
                continue
            if v is None:
                continue
            # foldseek / hmm 模式下仅校验各自相关路径
            pm = getattr(args, "predict_mode", None)
            if pm == "foldseek":
                if k in ("foldseek_db",):
                    if not v.exists():
                        raise FileNotFoundError(f"找不到路径({k}): {v}")
                continue
            if pm == "hmm":
                if k in ("hmm_db", "cluster_annotation", "ads_function"):
                    if not v.exists():
                        raise FileNotFoundError(f"找不到路径({k}): {v}")
                continue
            if k == "model_root":
                continue
            if k in ("esm2_model", "prost_model"):
                if use_mix_model or (k == "esm2_model" and single_embedding_kind == "esm2") or (k == "prost_model" and single_embedding_kind == "prostt5"):
                    if not v.exists():
                        raise FileNotFoundError(f"找不到下载模型目录({k}): {v}")
                continue
            if k == "single_embedding_model":
                if not v.exists():
                    raise FileNotFoundError(f"找不到 single 模式 embedding 模型目录: {v}")
                continue
            # 如果没有开启过滤，允许缺失阈值矩阵而不引发崩溃异常
            if k == "family_thresholds" and getattr(args, "filter_mode", "moderate") == "none":
                continue
            if not v.exists():
                raise FileNotFoundError(f"找不到路径({k}): {v}")
    return out


def run_self_check(script_dir: Path):
    db_dir = (script_dir / "database").resolve()
    required = [
        db_dir / "hmm_model" / "anti_defense_system.hmm",
        db_dir / "cnn_chkpnt" / "model.pt",
        db_dir / "PhADS_model" / "one_model" / "sdh_protonet_best.pth",
        db_dir / "PhADS_model" / "one_model" / "family_map.pth",
        db_dir / "PhADS_model" / "one_model" / "family_thresholds.tsv",
        db_dir / "PhADS_model" / "mix_model" / "sdh_protonet_crossattn_best.pth",
        db_dir / "PhADS_model" / "mix_model" / "family_map.pth",
        db_dir / "PhADS_model" / "mix_model" / "family_thresholds.tsv",
        db_dir / "anno_database" / "cluster_annotation.txt",
        db_dir / "anno_database" / "ADS_function.txt",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("  ✗ Self-check FAILED — missing files:")
        for m in missing:
            print(f"    - {m}")
        return 1

    print(f"  ✓ PhADS version {APP_VERSION} — all database files present")
    return 0


def cmd_foldseek(args, paths: dict, script_dir: Path, work_dir: Path, temp_dir: Path, log_file: Path,
                 prob_threshold: float = 0.8, output_stem: Optional[str] = None):
    """Handle --predict-mode foldseek: structural homology search via Foldseek.

    Args:
        output_stem: If provided, output file is named '{output_stem}_foldseek_result.tsv'.
                     Used for batch mode to generate per-file results.
    Returns:
        Path to the output TSV file.
    """

    input_path = Path(args.input_fasta).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    # Foldseek 临时目录（batch 模式下每个文件独立子目录避免冲突）
    foldseek_tmp_suffix = f"_{output_stem}" if output_stem else ""
    foldseek_tmp = temp_dir / f"temp_foldseek{foldseek_tmp_suffix}"
    foldseek_tmp.mkdir(parents=True, exist_ok=True)

    # 输出文件
    out_name = f"{output_stem}_foldseek_result.tsv" if output_stem else "foldseek_result.tsv"
    foldseek_out = work_dir / out_name

    total_steps = 3
    step = 0

    # Step 1: Run Foldseek
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Running Foldseek structural search", "start")
    run_cmd([
        sys.executable,
        str(paths["structure_compare_py"]),
        "-i", str(input_path),
        "-o", str(foldseek_tmp),
        "-d", str(paths["foldseek_db"]),
    ], cwd=script_dir, log_file=log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, "Running Foldseek structural search", "done", f"{elapsed:.1f}s")

    # Step 2: Post-process — extract top hit per query
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Post-processing Foldseek results", "start")

    result_tsv = foldseek_tmp / "result.tsv"
    if not result_tsv.exists():
        raise FileNotFoundError(f"Foldseek 未生成结果文件: {result_tsv}")

    # 加载 ADS 功能注释
    ads_func = {}
    ads_func_path = Path(paths["ads_function"])
    if ads_func_path.exists():
        with open(ads_func_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    ads_func[parts[0]] = parts[1]

    # 读取 Foldseek 结果，按 query 去重取最高 prob 的命中
    top_hits: Dict[str, Dict] = {}
    with open(result_tsv, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        cols = header_line.split("\t")
        # 列: query, target, prob, fident, alnlen, mismatch, gapopen, qstart, qend, tstart, tend, evalue, bits, lddt, alntmscore
        col_map = {c: i for i, c in enumerate(cols)}

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            q = parts[col_map.get("query", 0)]
            t = parts[col_map.get("target", 1)]
            prob = float(parts[col_map.get("prob", 2)])
            fident = float(parts[col_map.get("fident", 3)])
            qstart = parts[col_map.get("qstart", 7)]
            qend = parts[col_map.get("qend", 8)]

            if q not in top_hits or prob > top_hits[q]["prob"]:
                top_hits[q] = {
                    "query": q,
                    "target": t,
                    "prob": prob,
                    "fident": fident,
                    "qstart": qstart,
                    "qend": qend,
                }

    # 写入最终结果（仅保留 prob >= threshold 的命中）
    filtered_count = 0
    with open(foldseek_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query", "target", "prob", "fident", "qstart", "qend", "ads_function"])
        for q, hit in sorted(top_hits.items()):
            if hit["prob"] < prob_threshold:
                filtered_count += 1
                continue  # 低于阈值，不写入最终结果
            target_func = ads_func.get(hit["target"], "")
            writer.writerow([
                hit["query"], hit["target"],
                f"{hit['prob']:.4f}", f"{hit['fident']:.4f}",
                hit["qstart"], hit["qend"],
                target_func,
            ])

    elapsed = time.time() - t0
    print_progress(step, total_steps, "Post-processing Foldseek results", "done",
                   f"{len(top_hits)} queries, {filtered_count} below prob<{prob_threshold}, {elapsed:.1f}s")

    # Step 3: Print summary
    step += 1
    print_progress(step, total_steps, "Foldseek pipeline complete", "done", "")

    print_summary("Foldseek Output", [
        ("Result TSV", str(foldseek_out)),
        ("Queries processed", str(len(top_hits))),
        ("Prob threshold", str(prob_threshold)),
        ("Passed / Filtered", f"{len(top_hits) - filtered_count} / {filtered_count}"),
    ])

    print_summary("Intermediate Files (temp)", [
        ("Foldseek raw result", str(result_tsv) + " (all hits, unsorted)"),
        ("Foldseek tmp dir", str(foldseek_tmp)),
    ])

    return foldseek_out


def cmd_hmm(args, paths: dict, script_dir: Path, work_dir: Path, temp_dir: Path,
            domtblout_path: Path, log_file: Path, output_stem: Optional[str] = None):
    """Handle --predict-mode hmm: HMM-only alignment + annotation.

    Args:
        output_stem: If provided, output file is named '{output_stem}_hmm_result.tsv'.
                     Used for batch mode to generate per-file results.
    Returns:
        Path to the output TSV file.
    """

    input_fasta = Path(args.input_fasta).resolve()
    out_name = f"{output_stem}_hmm_result.tsv" if output_stem else "hmm_result.tsv"
    hmm_out = work_dir / out_name

    total_steps = 3
    step = 0

    # Step 1: Run hmmscan
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Running HMM alignment (hmmscan)", "start")
    run_cmd([
        "hmmscan",
        "--cpu", str(args.cpu),
        "--domtblout", str(domtblout_path),
        str(paths["hmm_db"]),
        str(input_fasta),
    ], cwd=script_dir, log_file=log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, "Running HMM alignment (hmmscan)", "done", f"{elapsed:.1f}s")

    # Step 2: Post-process — extract top acc hit per query + annotate
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Post-processing HMM results", "start")

    if not domtblout_path.exists():
        raise FileNotFoundError(f"hmmscan 未生成结果文件: {domtblout_path}")

    # 加载簇注释（label → {cluster_name, representative, funcs}）
    cluster_info: Dict[int, Dict] = {}
    cluster_func_cols_ordered: List[str] = []
    annot_path = Path(paths["cluster_annotation"])
    if annot_path.exists():
        with open(annot_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().rstrip("\n").split("\t")
            col_idx = {c: i for i, c in enumerate(header)}
            if "label" in col_idx and "representative" in col_idx and "cluster_name" in col_idx:
                func_cols = header[3:] if len(header) > 3 else []
                cluster_func_cols_ordered = func_cols
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    try:
                        lb = int(parts[col_idx["label"]])
                    except Exception:
                        continue
                    one: Dict[str, str] = {}
                    for c in func_cols:
                        i = col_idx.get(c, -1)
                        one[c] = parts[i].strip() if 0 <= i < len(parts) else ""
                    cluster_info[lb] = {
                        "cluster_name": parts[col_idx["cluster_name"]].strip(),
                        "representative": parts[col_idx["representative"]].strip(),
                        "funcs": one,
                    }

    # 加载 ADS 功能注释
    ads_func: Dict[str, str] = {}
    ads_path = Path(paths["ads_function"])
    if ads_path.exists():
        with open(ads_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    ads_func[parts[0]] = parts[1]

    # 解析 domtblout, 按 query 取最高 acc 的命中
    # domtblout 列: 0=target, 1=accession, 2=tlen, 3=query, 4=qacc, 5=qlen,
    #   6=E-value, 7=score, 8=bias, 9=#, 10=of, 11=c-Evalue, 12=i-Evalue, 13=dom_score,
    #   14=dom_bias, 15=hmm_from, 16=hmm_to, 17=ali_from, 18=ali_to, 19=env_from, 20=env_to, 21=acc
    top_hits: Dict[str, Dict] = {}
    with open(domtblout_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 22:
                continue
            q = parts[3]
            target_name = parts[0]
            try:
                acc = float(parts[21])
            except ValueError:
                continue
            ali_from = parts[17]
            ali_to = parts[18]
            hmm_from = parts[15]
            hmm_to = parts[16]

            if q not in top_hits or acc > top_hits[q]["acc"]:
                # 从 target name 提取簇编号 → label
                m = re.search(r"(\d+)(?!.*\d)", target_name)
                label = int(m.group(1)) - 1 if m else -1
                cinfo = cluster_info.get(label, {})
                rep_name = cinfo.get("representative", "")
                rep_func = ads_func.get(rep_name, "")
                cluster_name = cinfo.get("cluster_name", target_name)
                funcs = cinfo.get("funcs", {})

                top_hits[q] = {
                    "query": q,
                    "target": target_name,
                    "cluster_name": cluster_name,
                    "acc": acc,
                    "qstart": ali_from,
                    "qend": ali_to,
                    "tstart": hmm_from,
                    "tend": hmm_to,
                    "representative": rep_name,
                    "rep_ads_function": rep_func,
                    "funcs": funcs,
                }

    # 写入最终结果
    func_out_cols = [f"cluster_func_{c}" for c in cluster_func_cols_ordered]
    with open(hmm_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "query", "target", "cluster_name", "acc",
            "qstart", "qend", "tstart", "tend",
            "representative", "rep_ads_function",
            *func_out_cols,
        ])
        for q in sorted(top_hits.keys()):
            hit = top_hits[q]
            func_vals = [hit["funcs"].get(c, "") for c in cluster_func_cols_ordered]
            writer.writerow([
                hit["query"], hit["target"], hit["cluster_name"],
                f"{hit['acc']:.4f}",
                hit["qstart"], hit["qend"], hit["tstart"], hit["tend"],
                hit["representative"], hit["rep_ads_function"],
                *func_vals,
            ])

    elapsed = time.time() - t0
    print_progress(step, total_steps, "Post-processing HMM results", "done",
                   f"{len(top_hits)} queries, {elapsed:.1f}s")

    # Step 3: Summary
    step += 1
    print_progress(step, total_steps, "HMM pipeline complete", "done", "")

    print_summary("HMM Output", [
        ("Result TSV", str(hmm_out)),
        ("Queries processed", str(len(top_hits))),
    ])

    print_summary("Intermediate Files (temp)", [
        ("HMM raw output", str(domtblout_path)),
        ("Pipeline log", str(log_file)),
    ])

    return hmm_out


# ═══════════════════════════════════════════════════════════════
#  Batch directory processing
# ═══════════════════════════════════════════════════════════════

# Fast-forward prototype/instance pipeline for a single input FASTA.
# Returns (predict_out_path, predict_topk_out_path).
def run_prototype_instance_single(
    args,
    paths: dict,
    script_dir: Path,
    input_fasta: Path,
    work_dir: Path,
    temp_dir: Path,
    output_stem: str,
) -> tuple:
    """Run the full prototype/instance pipeline on a single FASTA file.

    Returns (predict_out_path, predict_topk_out_path).
    """

    # ── 检测输入类型：DNA 还是蛋白质 ──
    input_is_dna = is_dna_sequence(input_fasta)

    # 如果是 DNA，先运行 pyrodigal_gv 翻译为蛋白质
    protein_fasta_for_pipeline = input_fasta
    gene_positions: Dict[str, Dict[str, int]] = {}
    pyrodigal_faa_path: Optional[Path] = None
    pyrodigal_pos_path: Optional[Path] = None

    if input_is_dna:
        pyrodigal_faa_path = temp_dir / (output_stem + "_translated.faa")
        pyrodigal_pos_path = temp_dir / (output_stem + "_translated_pos.tsv")
        pyrodigal_gff_path = temp_dir / (output_stem + "_translated.gff")
        protein_fasta_for_pipeline = pyrodigal_faa_path

    # 每个文件使用独立的 embedding 子目录，避免跨文件序列 ID 冲突
    if args.model_type == "mix":
        esm2_emb_subdir = temp_dir / f"{output_stem}_esm2"
        prost_emb_subdir = temp_dir / f"{output_stem}_prost"
        esm2_emb_subdir.mkdir(parents=True, exist_ok=True)
        prost_emb_subdir.mkdir(parents=True, exist_ok=True)
        emb_subdir = prost_emb_subdir
    else:
        esm2_emb_subdir = None
        prost_emb_subdir = None
        emb_subdir = temp_dir / output_stem
        emb_subdir.mkdir(parents=True, exist_ok=True)

    # query 文件固定放 temp（batch 模式每个文件独立命名避免覆盖）
    protein_ids_path = temp_dir / f"{output_stem}_protein_ids.txt"
    emb_npy_path = temp_dir / f"{output_stem}_query_embeddings.npy"
    domtblout_path = temp_dir / f"{output_stem}_hmm_results.domtblout"
    hmm_npy_path = temp_dir / f"{output_stem}_query_hmm_features_L2.npy"
    hmm_txt_path = temp_dir / f"{output_stem}_query_hmm_features_L2.txt"
    log_file = temp_dir / "pipeline.log"

    device_str = str(args.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    total_steps = 9 if input_is_dna else 7
    step = 0

    # ── Step 0 (if DNA): Gene prediction ──
    if input_is_dna:
        step += 1
        t0 = time.time()
        print_progress(step, total_steps, f"[{output_stem}] Predicting genes (pyrodigal-gv)", "start")
        run_cmd([
            sys.executable,
            str(paths["pyrodigal_gv_py"]),
            "-i", str(input_fasta),
            "-a", str(pyrodigal_faa_path),
            "-g", str(pyrodigal_gff_path),
            "-p", str(pyrodigal_pos_path),
        ], cwd=script_dir, log_file=log_file)
        elapsed = time.time() - t0
        translated_ids = read_fasta_ids(pyrodigal_faa_path)
        print_progress(step, total_steps, f"[{output_stem}] Predicting genes (pyrodigal-gv)", "done",
                       f"{len(translated_ids)} proteins, {elapsed:.1f}s")

    # ── Step 1: Read FASTA ──
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, f"[{output_stem}] Reading input FASTA", "start")
    seq_ids = read_fasta_ids(protein_fasta_for_pipeline)
    protein_ids_path.write_text("\n".join(seq_ids) + "\n", encoding="utf-8")
    elapsed = time.time() - t0
    print_progress(step, total_steps, f"[{output_stem}] Reading input FASTA", "done",
                   f"{len(seq_ids)} sequences, {elapsed:.1f}s")

    # ── Step 2: Embedding generation ──
    step += 1
    t0 = time.time()
    embed_label = "ESM2 + ProtT5" if args.model_type == "mix" else paths["single_embedding_kind"].upper()
    print_progress(step, total_steps, f"[{output_stem}] Generating {embed_label} embeddings", "start")
    if args.model_type == "mix":
        if not args.query_emb_esm2 and not args.esm2_emb_dir:
            run_embedding_generation(paths, script_dir, protein_fasta_for_pipeline, esm2_emb_subdir,
                                     paths["esm2_model"], args, log_file)
        run_embedding_generation(paths, script_dir, protein_fasta_for_pipeline, prost_emb_subdir,
                                 paths["prost_model"], args, log_file)
    else:
        run_embedding_generation(paths, script_dir, protein_fasta_for_pipeline, emb_subdir,
                                 paths["single_embedding_model"], args, log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, f"[{output_stem}] Generating {embed_label} embeddings", "done", f"{elapsed:.1f}s")

    # ── Step 3: Build embedding matrix ──
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, f"[{output_stem}] Building embedding matrix", "start")
    emb_dim = int(paths["mix_prost_dim"] if args.model_type == "mix" else paths["single_embedding_dim"])
    emb_stats = build_embedding_npy(seq_ids=seq_ids, emb_dir=emb_subdir, out_npy=emb_npy_path, emb_dim=emb_dim)
    esm2_emb_path = None
    if args.model_type == "mix":
        if args.query_emb_esm2 or args.esm2_emb_dir:
            esm2_emb_path, esm2_stats = resolve_mix_esm2_embedding(
                args, seq_ids, temp_dir, output_stem=output_stem, expected_dim=int(paths["mix_esm2_dim"])
            )
        else:
            esm2_emb_path = temp_dir / f"{output_stem}_query_embeddings_esm2.npy"
            esm2_stats = build_embedding_npy(seq_ids=seq_ids, emb_dir=esm2_emb_subdir,
                                             out_npy=esm2_emb_path, emb_dim=int(paths["mix_esm2_dim"]))
    elapsed = time.time() - t0
    emb_detail = f"{emb_stats['shape'][0]}×{emb_stats['shape'][1]}"
    if args.model_type == "mix":
        emb_detail += f" + ESM2 {esm2_stats['shape'][0]}×{esm2_stats['shape'][1]}"
    print_progress(step, total_steps, f"[{output_stem}] Building embedding matrix", "done",
                   f"{emb_detail}, {elapsed:.1f}s")

    # ── Step 4: HMM alignment ──
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, f"[{output_stem}] Running HMM alignment (hmmscan)", "start")
    run_cmd([
        "hmmscan",
        "--cpu", str(args.cpu),
        "--domtblout", str(domtblout_path),
        str(paths["hmm_db"]),
        str(protein_fasta_for_pipeline),
    ], cwd=script_dir, log_file=log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, f"[{output_stem}] Running HMM alignment (hmmscan)", "done", f"{elapsed:.1f}s")

    # ── Step 5: Build HMM feature matrix ──
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, f"[{output_stem}] Building HMM feature matrix", "start")
    run_cmd([
        sys.executable,
        str(paths["hmm_to_npy_py"]),
        "-d", str(domtblout_path),
        "-i", str(protein_fasta_for_pipeline),
        "-on", str(hmm_npy_path),
        "-ot", str(hmm_txt_path),
    ], cwd=script_dir, log_file=log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, f"[{output_stem}] Building HMM feature matrix", "done", f"{elapsed:.1f}s")

    # ── Step 6: QC report ──
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, f"[{output_stem}] Generating QC report", "start")
    dom_stats = count_domtblout_hits(domtblout_path, set(seq_ids))
    qc_report = make_qc_report(seq_ids, emb_stats, dom_stats, emb_npy_path, hmm_npy_path)
    qc_txt_path = work_dir / f"{output_stem}_qc_report.txt"
    qc_json_path = work_dir / f"{output_stem}_qc_report.json"
    write_qc_reports(qc_report, qc_txt_path, qc_json_path)
    elapsed = time.time() - t0
    print_progress(step, total_steps, f"[{output_stem}] Generating QC report", "done", f"{elapsed:.1f}s")

    # ── Step 7: SDH-ProtoNet prediction ──
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, f"[{output_stem}] Running SDH-ProtoNet prediction", "start")
    predict_out = (work_dir / f"{output_stem}_results.tsv").resolve()
    predict_topk_out = (work_dir / f"{output_stem}_prediction_topk.tsv").resolve()
    cmd_predict = [
        sys.executable,
        str(paths["predict_script"]),
        "--model-path", str(paths["predict_model"]),
        "--map-path", str(paths["predict_map"]),
    ]
    if args.model_type == "mix":
        cmd_predict.extend([
            "--query-emb-esm2", str(esm2_emb_path),
            "--query-emb-prost5", str(emb_npy_path),
        ])
    else:
        cmd_predict.extend(["--query-emb", str(emb_npy_path)])
    cmd_predict.extend([
        "--query-hmm", str(hmm_npy_path),
        "--query-ids", str(protein_ids_path),
        "--cluster-annotation-txt", str(paths["cluster_annotation"]),
        "--ads-function-txt", str(paths["ads_function"]),
        "--mode", str(args.predict_mode),
        "--topk", str(args.topk),
        "--output-tsv", str(predict_out),
        "--topk-tsv", str(predict_topk_out),
        "--threshold-tsv", str(paths["family_thresholds"]),
        "--filter-mode", str(args.filter_mode),
    ])
    if args.print_topk:
        cmd_predict.append("--print-topk")

    run_cmd(cmd_predict, cwd=script_dir, log_file=log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, f"[{output_stem}] Running SDH-ProtoNet prediction", "done", f"{elapsed:.1f}s")

    # ── Step 8 (if DNA): Add gene positions ──
    if input_is_dna:
        step += 1
        t0 = time.time()
        print_progress(step, total_steps, f"[{output_stem}] Adding gene positions to output", "start")
        gene_positions = load_gene_positions(pyrodigal_pos_path)
        add_gene_positions_to_tsv(predict_out, gene_positions, is_main_output=True)
        add_gene_positions_to_tsv(predict_topk_out, gene_positions, is_main_output=False)
        elapsed = time.time() - t0
        print_progress(step, total_steps, f"[{output_stem}] Adding gene positions to output", "done",
                       f"{len(gene_positions)} genes mapped, {elapsed:.1f}s")

    return predict_out, predict_topk_out


def _merge_tsv_files_v2(file_paths: List[Path], merged_path: Path):
    """Merge multiple TSV files — write header once, then all data rows."""
    if not file_paths:
        return
    existing = [p for p in file_paths if p.exists()]
    if not existing:
        return

    with open(merged_path, "w", encoding="utf-8", newline="") as outf:
        writer = csv.writer(outf, delimiter="\t")
        header = None
        for p in existing:
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if not row:
                        continue
                    if header is None:
                        header = row
                        writer.writerow(row)
                    elif row == header:
                        continue  # skip duplicate headers
                    else:
                        writer.writerow(row)


def cmd_dir(args, script_dir: Path):
    """Batch process all FASTA files in a directory (-d/--dir)."""

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"不是有效目录: {input_dir}")

    # 查找所有 FASTA 文件
    fasta_exts = {".fasta", ".faa", ".fa", ".fna", ".ffn", ".frn", ".fas"}
    fasta_files: List[Path] = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in fasta_exts
    )
    if not fasta_files:
        raise ValueError(f"目录中未找到 FASTA 文件 (ext={fasta_exts}): {input_dir}")

    # 设置输出目录
    work_dir_arg = Path(args.work_dir)
    work_dir = work_dir_arg if work_dir_arg.is_absolute() else (Path.cwd() / work_dir_arg)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    topk_dir = work_dir / "topk"
    topk_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = resolve_temp_dir(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_file = temp_dir / "pipeline.log"

    pm = args.predict_mode
    if pm in ("prototype", "instance") and not args.mode_path:
        raise ValueError("prototype/instance 模式必须提供 -db/--mode-path 下载模型根目录")
    paths = resolve_auto_paths(script_dir, args)

    device_str = str(args.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Header ──
    w = _box_width()
    title = f"  PhADS v{APP_VERSION} — Batch Directory Mode"
    print()
    print("╔" + "═" * w + "╗")
    print(f"║{title:<{w}}║")
    print("╚" + "═" * w + "╝")
    print()
    print("  ── Batch Configuration " + "─" * (w - 25))
    print(f"  {'Input Directory':<22} {input_dir}")
    print(f"  {'Output Directory':<22} {work_dir}")
    print(f"  {'Temp Directory':<22} {temp_dir}")
    print(f"  {'Prediction Mode':<22} {pm}")
    print(f"  {'Files Found':<22} {len(fasta_files)}")
    if pm == "foldseek":
        print(f"  {'Prob Threshold':<22} {args.prob_threshold}")
    if pm in ("prototype", "instance"):
        print(f"  {'Model Type':<22} {args.model_type}")
        print(f"  {'Model Root':<22} {paths['model_root']}")
        print(f"  {'Thresholds':<22} {paths['family_thresholds']}")
        print(f"  {'Filter Mode':<22} {args.filter_mode}")
        print(f"  {'Device':<22} {device_str}")
        print(f"  {'Top-K':<22} {args.topk}")
    print(f"  {'CPU Threads':<22} {args.cpu}")
    print("  " + "─" * (w - 2))
    print()

    # ── 收集各文件输出路径 ──
    individual_results: List[Path] = []
    individual_topk: List[Path] = []

    t_start = time.time()

    for idx, fasta_file in enumerate(fasta_files, 1):
        stem = fasta_file.stem
        print(f"  ══ [{idx}/{len(fasta_files)}] Processing: {fasta_file.name} ══")
        print()

        # 临时修改 args.input_fasta 指向当前文件
        args.input_fasta = str(fasta_file)

        try:
            if pm == "foldseek":
                out_path = cmd_foldseek(
                    args, paths, script_dir, work_dir, temp_dir, log_file,
                    prob_threshold=args.prob_threshold,
                    output_stem=stem,
                )
                individual_results.append(out_path)

            elif pm == "hmm":
                domtblout_path = temp_dir / f"{stem}_hmm_results.domtblout"
                out_path = cmd_hmm(
                    args, paths, script_dir, work_dir, temp_dir,
                    domtblout_path, log_file,
                    output_stem=stem,
                )
                individual_results.append(out_path)

            else:  # prototype / instance
                predict_out, predict_topk_out = run_prototype_instance_single(
                    args, paths, script_dir, fasta_file, work_dir, temp_dir,
                    output_stem=stem,
                )
                individual_results.append(predict_out)
                individual_topk.append(predict_topk_out)

        except Exception as e:
            print(f"\n  ⚠ ERROR processing {fasta_file.name}: {e}")
            print(f"  → Skipping and continuing with next file...\n")
            continue

    elapsed_total = time.time() - t_start

    # ── 合并结果 ──
    print()
    print("  ── Merging results " + "─" * (w - 21))

    if pm == "foldseek":
        all_out = work_dir / "all_foldseek_result.tsv"
        _merge_tsv_files_v2(individual_results, all_out)
        print(f"  {'Merged output':<22} {all_out} ({len(individual_results)} files)")

    elif pm == "hmm":
        all_out = work_dir / "all_hmm_result.tsv"
        _merge_tsv_files_v2(individual_results, all_out)
        print(f"  {'Merged output':<22} {all_out} ({len(individual_results)} files)")

    else:  # prototype / instance
        all_results_out = work_dir / "all_results.tsv"
        all_topk_out = topk_dir / "all_prediction_topk.tsv"
        _merge_tsv_files_v2(individual_results, all_results_out)
        _merge_tsv_files_v2(individual_topk, all_topk_out)

        # 移动单独的 topk 文件到 topk/ 目录
        for topk_file in individual_topk:
            if topk_file.exists():
                dest = topk_dir / topk_file.name
                if dest.exists():
                    dest.unlink()
                topk_file.rename(dest)

        print(f"  {'Merged results':<22} {all_results_out} ({len(individual_results)} files)")
        print(f"  {'Merged topk':<22} {all_topk_out} ({len(individual_topk)} files)")
        print(f"  {'TopK directory':<22} {topk_dir}")

    print("  " + "─" * (w - 2))
    print()

    # ── Summary ──
    print_summary("Batch Complete", [
        ("Files processed", f"{len(fasta_files)}"),
        ("Successfully", f"{len([p for p in individual_results if p.exists()])}"),
        ("Total time", f"{elapsed_total:.1f}s"),
        ("Output directory", str(work_dir)),
    ])

    print_done_box()


def cmd_database(args, script_dir: Path):
    """Handle the 'database' subcommand: download ESM2 and ProtT5 models."""
    download_path = Path(args.download_path).resolve()
    print_download_header(str(download_path))

    download_script = script_dir / "scripts" / "download_model.py"
    if not download_script.exists():
        raise FileNotFoundError(f"找不到下载脚本: {download_script}")

    t0 = time.time()
    print_progress(1, 1, "Downloading ESM2 and ProtT5 models from HuggingFace", "start")
    run_cmd([
        sys.executable,
        str(download_script),
        "-o", str(download_path),
    ], cwd=script_dir)
    elapsed = time.time() - t0
    print_progress(1, 1, "Downloading ESM2 and ProtT5 models from HuggingFace", "done", f"{elapsed:.1f}s")

    print_summary("Download Complete", [
        ("Model directory", str(download_path)),
        ("Time elapsed", f"{elapsed:.1f}s"),
    ])
    print("  Usage: python main.py -i <fasta> --db " + str(download_path))
    print()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    # ── Subcommand: database ──
    if args.command == "database":
        cmd_database(args, script_dir)
        return

    # ── Self-check ──
    if args.version:
        rc = run_self_check(script_dir)
        if rc != 0:
            raise SystemExit(rc)
        return

    # ── Batch directory mode (-d/--dir) ──
    if args.input_dir:
        cmd_dir(args, script_dir)
        return

    # ── Foldseek mode: structural pipeline (early return) ──
    if args.predict_mode == "foldseek":
        if not args.input_fasta:
            raise ValueError("必须提供 -i/--input-fasta（结构文件或目录）")

        input_fasta = Path(args.input_fasta).resolve()
        if not input_fasta.exists():
            raise FileNotFoundError(f"输入路径不存在: {input_fasta}")

        paths = resolve_auto_paths(script_dir, args)

        work_dir_arg = Path(args.work_dir)
        work_dir = work_dir_arg if work_dir_arg.is_absolute() else (Path.cwd() / work_dir_arg)
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = resolve_temp_dir(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        log_file = temp_dir / "pipeline.log"

        device_str = str(args.device)
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        config_items = [
            ("Input Path", str(input_fasta)),
            ("Mode", "foldseek (structural)"),
            ("Output Directory", str(work_dir)),
            ("Temp Directory", str(temp_dir)),
            ("Foldseek DB", str(paths["foldseek_db"])),
            ("Prob Threshold", str(args.prob_threshold)),
            ("CPU Threads", str(args.cpu)),
        ]
        print_header(config_items)

        cmd_foldseek(args, paths, script_dir, work_dir, temp_dir, log_file,
                     prob_threshold=args.prob_threshold)
        return

    # ── HMM mode: HMM-only pipeline (early return) ──
    if args.predict_mode == "hmm":
        if not args.input_fasta:
            raise ValueError("必须提供 -i/--input-fasta")

        input_fasta = Path(args.input_fasta).resolve()
        if not input_fasta.exists():
            raise FileNotFoundError(f"输入 FASTA 不存在: {input_fasta}")

        paths = resolve_auto_paths(script_dir, args)

        work_dir_arg = Path(args.work_dir)
        work_dir = work_dir_arg if work_dir_arg.is_absolute() else (Path.cwd() / work_dir_arg)
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = resolve_temp_dir(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        log_file = temp_dir / "pipeline.log"
        domtblout_path = temp_dir / "hmm_results.domtblout"

        config_items = [
            ("Input FASTA", str(input_fasta)),
            ("Mode", "hmm (HMM-only)"),
            ("Output Directory", str(work_dir)),
            ("Temp Directory", str(temp_dir)),
            ("HMM Database", str(paths["hmm_db"])),
            ("CPU Threads", str(args.cpu)),
        ]
        print_header(config_items)

        cmd_hmm(args, paths, script_dir, work_dir, temp_dir, domtblout_path, log_file)
        return

    # ── Validate pipeline arguments ──
    if not args.input_fasta:
        raise ValueError("必须提供 -i/--input-fasta")
    if not args.mode_path:
        raise ValueError("必须提供 -db/--mode-path")

    input_fasta = Path(args.input_fasta).resolve()
    if not input_fasta.exists():
        raise FileNotFoundError(f"输入 FASTA 不存在: {input_fasta}")

    paths = resolve_auto_paths(script_dir, args)

    # work_dir
    work_dir_arg = Path(args.work_dir)
    work_dir = work_dir_arg if work_dir_arg.is_absolute() else (Path.cwd() / work_dir_arg)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # temp_dir
    temp_dir = resolve_temp_dir(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ── 检测输入类型：DNA 还是蛋白质 ──
    input_is_dna = is_dna_sequence(input_fasta)
    input_type_label = "DNA (基因组)" if input_is_dna else "蛋白质"

    # 如果是 DNA，先运行 pyrodigal_gv 翻译为蛋白质
    protein_fasta_for_pipeline = input_fasta  # 下游使用的蛋白 FASTA
    gene_positions: Dict[str, Dict[str, int]] = {}
    pyrodigal_faa_path: Optional[Path] = None
    pyrodigal_pos_path: Optional[Path] = None

    if input_is_dna:
        pyrodigal_faa_path = temp_dir / (input_fasta.stem + "_translated.faa")
        pyrodigal_pos_path = temp_dir / (input_fasta.stem + "_translated_pos.tsv")
        pyrodigal_gff_path = temp_dir / (input_fasta.stem + "_translated.gff")
        protein_fasta_for_pipeline = pyrodigal_faa_path

    # query 文件固定放 temp
    protein_ids_path = temp_dir / "protein_ids.txt"
    emb_npy_path = temp_dir / "query_embeddings.npy"
    esm2_emb_dir = temp_dir / "esm2_embeddings"
    prost_emb_dir = temp_dir / "prost_embeddings"
    domtblout_path = temp_dir / "hmm_results.domtblout"
    hmm_npy_path = temp_dir / "query_hmm_features_L2.npy"
    hmm_txt_path = temp_dir / "query_hmm_features_L2.txt"
    log_file = temp_dir / "pipeline.log"

    # ═══════════════════════════════════════════════════
    #  Print clean header
    # ═══════════════════════════════════════════════════
    device_str = str(args.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    config_items = [
        ("Input FASTA", str(input_fasta)),
        ("Input Type", input_type_label),
        ("Output Directory", str(work_dir)),
        ("Temp Directory", str(temp_dir)),
        ("Model Root", str(paths["model_root"])),
        ("ESM2 Model", str(paths["esm2_model"])),
        ("ProtT5 Model", str(paths["prost_model"])),
        ("HMM Database", str(paths["hmm_db"])),
        ("Thresholds", str(paths["family_thresholds"])),
        ("Device", device_str),
        ("Prediction Mode", str(args.predict_mode)),
        ("Model Type", str(args.model_type)),
        ("Filter Mode", str(args.filter_mode)),
        ("Top-K Candidates", str(args.topk)),
        ("CPU Threads", str(args.cpu)),
    ]
    print_header(config_items)

    # 步骤总数：蛋白质输入 7 步，DNA 输入 9 步（+基因预测+基因位置后处理）
    total_steps = 9 if input_is_dna else 7
    step = 0

    # ═══════════════════════════════════════════════════
    #  Step 0 (if DNA): Gene prediction with pyrodigal-gv
    # ═══════════════════════════════════════════════════
    if input_is_dna:
        step += 1
        t0 = time.time()
        print_progress(step, total_steps, "Predicting genes (pyrodigal-gv)", "start")
        run_cmd([
            sys.executable,
            str(paths["pyrodigal_gv_py"]),
            "-i", str(input_fasta),
            "-a", str(pyrodigal_faa_path),
            "-g", str(pyrodigal_gff_path),
            "-p", str(pyrodigal_pos_path),
        ], cwd=script_dir, log_file=log_file)
        elapsed = time.time() - t0
        # 统计翻译得到的蛋白序列数
        translated_ids = read_fasta_ids(pyrodigal_faa_path)
        print_progress(step, total_steps, "Predicting genes (pyrodigal-gv)", "done",
                       f"{len(translated_ids)} proteins translated, {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Step 1: Read FASTA (protein)
    # ═══════════════════════════════════════════════════
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Reading input FASTA", "start")
    seq_ids = read_fasta_ids(protein_fasta_for_pipeline)
    protein_ids_path.write_text("\n".join(seq_ids) + "\n", encoding="utf-8")
    elapsed = time.time() - t0
    print_progress(step, total_steps, "Reading input FASTA", "done", f"{len(seq_ids)} sequences, {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Step 2: Embedding generation
    # ═══════════════════════════════════════════════════
    step += 1
    t0 = time.time()
    embed_label = "ESM2 + ProtT5" if args.model_type == "mix" else paths["single_embedding_kind"].upper()
    print_progress(step, total_steps, f"Generating {embed_label} embeddings", "start")
    if args.model_type == "mix":
        if not args.query_emb_esm2 and not args.esm2_emb_dir:
            esm2_emb_dir.mkdir(parents=True, exist_ok=True)
            run_embedding_generation(paths, script_dir, protein_fasta_for_pipeline, esm2_emb_dir,
                                     paths["esm2_model"], args, log_file)
        prost_emb_dir.mkdir(parents=True, exist_ok=True)
        run_embedding_generation(paths, script_dir, protein_fasta_for_pipeline, prost_emb_dir,
                                 paths["prost_model"], args, log_file)
    else:
        run_embedding_generation(paths, script_dir, protein_fasta_for_pipeline, temp_dir,
                                 paths["single_embedding_model"], args, log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, f"Generating {embed_label} embeddings", "done", f"{elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Step 3: Build embedding matrix
    # ═══════════════════════════════════════════════════
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Building embedding matrix", "start")
    emb_dir_for_matrix = prost_emb_dir if args.model_type == "mix" else temp_dir
    emb_dim = int(paths["mix_prost_dim"] if args.model_type == "mix" else paths["single_embedding_dim"])
    emb_stats = build_embedding_npy(seq_ids=seq_ids, emb_dir=emb_dir_for_matrix, out_npy=emb_npy_path, emb_dim=emb_dim)
    esm2_emb_path = None
    if args.model_type == "mix":
        if args.query_emb_esm2 or args.esm2_emb_dir:
            esm2_emb_path, esm2_stats = resolve_mix_esm2_embedding(
                args, seq_ids, temp_dir, expected_dim=int(paths["mix_esm2_dim"])
            )
        else:
            esm2_emb_path = temp_dir / "query_embeddings_esm2.npy"
            esm2_stats = build_embedding_npy(seq_ids=seq_ids, emb_dir=esm2_emb_dir,
                                             out_npy=esm2_emb_path, emb_dim=int(paths["mix_esm2_dim"]))
    elapsed = time.time() - t0
    emb_detail = f"{emb_stats['shape'][0]}×{emb_stats['shape'][1]}"
    if args.model_type == "mix":
        emb_detail += f" + ESM2 {esm2_stats['shape'][0]}×{esm2_stats['shape'][1]}"
    print_progress(step, total_steps, "Building embedding matrix", "done",
                   f"{emb_detail}, {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Step 4: HMM alignment (hmmscan)
    # ═══════════════════════════════════════════════════
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Running HMM alignment (hmmscan)", "start")
    run_cmd([
        "hmmscan",
        "--cpu", str(args.cpu),
        "--domtblout", str(domtblout_path),
        str(paths["hmm_db"]),
        str(protein_fasta_for_pipeline),
    ], cwd=script_dir, log_file=log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, "Running HMM alignment (hmmscan)", "done", f"{elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Step 5: Build HMM feature matrix
    # ═══════════════════════════════════════════════════
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Building HMM feature matrix", "start")
    run_cmd([
        sys.executable,
        str(paths["hmm_to_npy_py"]),
        "-d", str(domtblout_path),
        "-i", str(protein_fasta_for_pipeline),
        "-on", str(hmm_npy_path),
        "-ot", str(hmm_txt_path),
    ], cwd=script_dir, log_file=log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, "Building HMM feature matrix", "done", f"{elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Step 6: QC report
    # ═══════════════════════════════════════════════════
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Generating QC report", "start")
    dom_stats = count_domtblout_hits(domtblout_path, set(seq_ids))
    qc_report = make_qc_report(seq_ids, emb_stats, dom_stats, emb_npy_path, hmm_npy_path)
    qc_txt_path = work_dir / args.qc_txt
    qc_json_path = work_dir / args.qc_json
    write_qc_reports(qc_report, qc_txt_path, qc_json_path)
    elapsed = time.time() - t0
    print_progress(step, total_steps, "Generating QC report", "done", f"{elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Step 7: SDH-ProtoNet prediction
    # ═══════════════════════════════════════════════════
    step += 1
    t0 = time.time()
    print_progress(step, total_steps, "Running SDH-ProtoNet prediction", "start")
    predict_out = (work_dir / args.predict_output).resolve()
    predict_topk_out = (work_dir / args.predict_topk_output).resolve()
    cmd_predict = [
        sys.executable,
        str(paths["predict_script"]),
        "--model-path", str(paths["predict_model"]),
        "--map-path", str(paths["predict_map"]),
    ]
    if args.model_type == "mix":
        cmd_predict.extend([
            "--query-emb-esm2", str(esm2_emb_path),
            "--query-emb-prost5", str(emb_npy_path),
        ])
    else:
        cmd_predict.extend(["--query-emb", str(emb_npy_path)])
    cmd_predict.extend([
        "--query-hmm", str(hmm_npy_path),
        "--query-ids", str(protein_ids_path),
        "--cluster-annotation-txt", str(paths["cluster_annotation"]),
        "--ads-function-txt", str(paths["ads_function"]),
        "--mode", str(args.predict_mode),
        "--topk", str(args.topk),
        "--output-tsv", str(predict_out),
        "--topk-tsv", str(predict_topk_out),
        "--threshold-tsv", str(paths["family_thresholds"]),
        "--filter-mode", str(args.filter_mode),
    ])
    if args.print_topk:
        cmd_predict.append("--print-topk")

    run_cmd(cmd_predict, cwd=script_dir, log_file=log_file)
    elapsed = time.time() - t0
    print_progress(step, total_steps, "Running SDH-ProtoNet prediction", "done", f"{elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Step 8 (if DNA): Post-process — add gene positions
    # ═══════════════════════════════════════════════════
    if input_is_dna:
        step += 1
        t0 = time.time()
        print_progress(step, total_steps, "Adding gene positions to output", "start")
        gene_positions = load_gene_positions(pyrodigal_pos_path)
        add_gene_positions_to_tsv(predict_out, gene_positions, is_main_output=True)
        add_gene_positions_to_tsv(predict_topk_out, gene_positions, is_main_output=False)
        elapsed = time.time() - t0
        print_progress(step, total_steps, "Adding gene positions to output", "done",
                       f"{len(gene_positions)} genes mapped, {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════
    #  Results summary
    # ═══════════════════════════════════════════════════
    summary_items = [
        ("Predictions (TSV)", str(predict_out)),
        ("Top-K Details (TSV)", str(predict_topk_out)),
        ("QC Report (txt)", str(qc_txt_path)),
        ("QC Report (json)", str(qc_json_path)),
    ]
    print_summary("Output Files", summary_items)

    temp_summary_items = [
        ("Embedding matrix", str(emb_npy_path)),
        ("HMM feature matrix", str(hmm_npy_path)),
        ("HMM raw output", str(domtblout_path)),
        ("Pipeline log", str(log_file)),
    ]
    if args.model_type == "mix":
        temp_summary_items.insert(1, ("ESM2 embedding matrix", str(esm2_emb_path)))
    if input_is_dna:
        temp_summary_items.append(("Translated protein (.faa)", str(pyrodigal_faa_path)))
        temp_summary_items.append(("Gene positions (.tsv)", str(pyrodigal_pos_path)))
        temp_summary_items.append(("GFF3 annotation (.gff)", str(pyrodigal_gff_path)))
    print_summary("Intermediate Files (temp)", temp_summary_items)

    print_done_box()


if __name__ == "__main__":
    main()