#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主流程脚本：自动生成预测所需 .npy 并直接调用 predict.py（支持多序列 FASTA）

默认自动路径（相对 main.py 所在目录）：
- HMM DB: database/hmm_model/anti_defense_system.hmm
- CNN 权重: database/cnn_chkpnt/model.pt
- PhADS model: database/PhADS_model/sdh_protonet_best.pth
- PhADS map: database/PhADS_model/family_map.pth
- PhADS thresholds: database/PhADS_model/family_thresholds.tsv
- 注释库: database/anno_database/{cluster_annotation.txt, ADS_function.txt}

说明：
- ProstT5 模型路径必须通过 `-db/--mode-path` 指定
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
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


APP_VERSION = "0.1"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "PhADS end-to-end inference pipeline: embedding generation, HMM feature extraction, and prediction.\n\n"
            "Output schema:\n"
            "  1) --predict-output (TSV)\n"
            "     query_id, mode, pred_id, pred_cluster, pred_cluster_rep,\n"
            "     pred_distance2, confidence, filter_mode, filter_status, threshold_limit,\n"
            "     nearest_sequence_id, nearest_sequence_label, nearest_sequence_distance2,\n"
            "     nearest_sequence_function_summary, cluster_func_*\n\n"
            "  2) --predict-topk-output (TSV)\n"
            "     query_id, mode, rank, candidate_id, candidate_cluster, candidate_rep,\n"
            "     candidate_distance2, candidate_confidence, filter_mode, filter_status,\n"
            "     threshold_limit, candidate_ads_function"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("-i", "--input-fasta", required=False, help="Input FASTA file (single or multiple sequences).")
    p.add_argument("-o", "--work-dir", default="run_auto", help="Final output directory for prediction and QC reports.")
    p.add_argument("-n", "--cpu", type=int, default=8, help="Number of CPU threads for embedding and hmmscan.")
    p.add_argument("--device", default="auto", help="Embedding device: auto | cpu | cuda.")

    p.add_argument(
        "-temp", "--temp", dest="temp_dir", default=None,
        help="Temporary directory for intermediate query files (.pt/.npy/.domtblout). Defaults to ./temp."
    )

    p.add_argument("-db", "--mode-path", dest="mode_path", required=False, help="Path to the ProstT5 model directory.")

    p.add_argument("--predict-mode", choices=["prototype", "instance"], default="prototype", help="Prediction mode.")
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


def run_cmd(cmd: List[str], cwd: Path):
    print("\n[RUN]", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(cwd))
    if ret.returncode != 0:
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

    print(f"✅ embedding npy 已生成: {out_npy} | shape={arr.shape}")
    if missing:
        miss_log = out_npy.parent / "missing_embedding_ids.log"
        miss_log.write_text("\n".join(missing), encoding="utf-8")
        print(f"⚠️ 有 {len(missing)} 个序列未找到 embedding，已用零向量填充，详见: {miss_log}")

    return {
        "missing_count": len(missing),
        "missing_ids": missing,
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
    }


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


def resolve_auto_paths(script_dir: Path, args):
    db_dir = script_dir / "database"
    out = {
        "translate_py": script_dir / "scripts" / "translate_to_embedding.py",
        "hmm_to_npy_py": script_dir / "scripts" / "hmm_to_npy.py",
        "predict_py": script_dir / "scripts" / "predict.py",
        "hmm_db": (db_dir / "hmm_model" / "anti_defense_system.hmm").resolve(),
        "prost_model": Path(args.mode_path).resolve() if args.mode_path else None,
        "cnn_weights": (db_dir / "cnn_chkpnt" / "model.pt").resolve(),
        "predict_model": (db_dir / "PhADS_model" / "sdh_protonet_best.pth").resolve(),
        "predict_map": (db_dir / "PhADS_model" / "family_map.pth").resolve(),
        # 🌟 按照要求：无缝集成 family_thresholds.tsv 路径到自动路径树中
        "family_thresholds": (db_dir / "PhADS_model" / "family_thresholds.tsv").resolve(),
        "cluster_annotation": (db_dir / "anno_database" / "cluster_annotation.txt").resolve(),
        "ads_function": (db_dir / "anno_database" / "ADS_function.txt").resolve(),
    }

    for k, v in out.items():
        if k.endswith("_py"):
            if not v.exists():
                raise FileNotFoundError(f"找不到脚本: {v}")
        else:
            if v is None:
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
        db_dir / "PhADS_model" / "sdh_protonet_best.pth",
        db_dir / "PhADS_model" / "family_map.pth",
        # 🌟 自检模块同步加入该判定，保障数据库绝对完整性
        db_dir / "PhADS_model" / "family_thresholds.tsv",
        db_dir / "anno_database" / "cluster_annotation.txt",
        db_dir / "anno_database" / "ADS_function.txt",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("❌ 自检失败：database 内容不完整，缺少以下文件：")
        for m in missing:
            print(f"- {m}")
        return 1

    print(f"PhADS version {APP_VERSION}")
    return 0


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.version:
        rc = run_self_check(script_dir)
        if rc != 0:
            raise SystemExit(rc)
        return

    if not args.input_fasta:
        raise ValueError("必须提供 -i/--input-fasta")
    if not args.mode_path:
        raise ValueError("必须提供 -db/--mode-path")

    input_fasta = Path(args.input_fasta).resolve()
    if not input_fasta.exists():
        raise FileNotFoundError(f"输入 FASTA 不存在: {input_fasta}")

    paths = resolve_auto_paths(script_dir, args)

    # work_dir: 最终输出目录（predict结果 + QC报告）
    work_dir_arg = Path(args.work_dir)
    work_dir = work_dir_arg if work_dir_arg.is_absolute() else (Path.cwd() / work_dir_arg)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = resolve_temp_dir(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"🗂️ 临时目录(temp): {temp_dir}")

    # query 文件固定放 temp（不再暴露冗余命名参数）
    protein_ids_path = temp_dir / "protein_ids.txt"
    emb_npy_path = temp_dir / "query_embeddings.npy"
    domtblout_path = temp_dir / "hmm_results.domtblout"
    hmm_npy_path = temp_dir / "query_hmm_features_L2.npy"
    hmm_txt_path = temp_dir / "query_hmm_features_L2.txt"

    # 1) 读取 FASTA 并写入 query IDs
    seq_ids = read_fasta_ids(input_fasta)
    protein_ids_path.write_text("\n".join(seq_ids) + "\n", encoding="utf-8")
    print(f"✅ 读取到 {len(seq_ids)} 条序列，ID 文件已写入: {protein_ids_path}")

    # 2) embedding（.pt 直接输出到 temp）
    run_cmd([
        sys.executable,
        str(paths["translate_py"]),
        "-i", str(input_fasta),
        "-o", str(temp_dir),
        "-n", str(args.cpu),
        "--model-path", str(paths["prost_model"]),
        "--cnn-weights", str(paths["cnn_weights"]),
        "--device", str(args.device),
    ], cwd=script_dir)

    # 3) embedding -> npy
    emb_stats = build_embedding_npy(seq_ids=seq_ids, emb_dir=temp_dir, out_npy=emb_npy_path, emb_dim=1024)

    # 4) hmmscan
    run_cmd([
        "hmmscan",
        "--cpu", str(args.cpu),
        "--domtblout", str(domtblout_path),
        str(paths["hmm_db"]),
        str(input_fasta),
    ], cwd=script_dir)

    # 5) hmm -> npy（不依赖 cluster_merge_final.txt，簇数内置在脚本）
    run_cmd([
        sys.executable,
        str(paths["hmm_to_npy_py"]),
        "-d", str(domtblout_path),
        "-i", str(input_fasta),
        "-on", str(hmm_npy_path),
        "-ot", str(hmm_txt_path),
    ], cwd=script_dir)

    # 6) 质检
    dom_stats = count_domtblout_hits(domtblout_path, set(seq_ids))
    qc_report = make_qc_report(seq_ids, emb_stats, dom_stats, emb_npy_path, hmm_npy_path)

    qc_txt_path = work_dir / args.qc_txt
    qc_json_path = work_dir / args.qc_json
    write_qc_reports(qc_report, qc_txt_path, qc_json_path)

    # 7) 调用 predict.py（注入新增的动态阈值控制参数）
    predict_out = (work_dir / args.predict_output).resolve()
    predict_topk_out = (work_dir / args.predict_topk_output).resolve()
    cmd_predict = [
        sys.executable,
        str(paths["predict_py"]),
        "--model-path", str(paths["predict_model"]),
        "--map-path", str(paths["predict_map"]),
        "--query-emb", str(emb_npy_path),
        "--query-hmm", str(hmm_npy_path),
        "--query-ids", str(protein_ids_path),
        "--cluster-annotation-txt", str(paths["cluster_annotation"]),
        "--ads-function-txt", str(paths["ads_function"]),
        "--mode", str(args.predict_mode),
        "--topk", str(args.topk),
        "--output-tsv", str(predict_out),
        "--topk-tsv", str(predict_topk_out),
        # 🌟 透传动态拦截所需元数据路径与模式开关
        "--threshold-tsv", str(paths["family_thresholds"]),
        "--filter-mode", str(args.filter_mode),
    ]
    if args.print_topk:
        cmd_predict.append("--print-topk")

    run_cmd(cmd_predict, cwd=script_dir)

    print("\n" + "=" * 60)
    print("🎉 全部完成！")
    print(f"- query emb : {emb_npy_path}")
    print(f"- query hmm : {hmm_npy_path}")
    print(f"- query ids : {protein_ids_path}")
    print(f"- qc txt    : {qc_txt_path}")
    print(f"- qc json   : {qc_json_path}")
    print(f"- predict   : {predict_out}")
    print(f"- topk      : {predict_topk_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()