#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精简版 ProstT5 embedding 脚本（仅保留 embedding + 置信度）

保留能力：
1) 支持单个 FASTA 文件或目录（批量 FASTA）
2) 多进程 CPU 推理，或单进程 GPU 推理
3) 输出每条序列的 embedding: <seq_id>_embedding.pt
4) 输出 summary.csv（含 avg_conf）
5) 支持断点续跑（checkpoint.idx）

移除内容：
- 3Di 序列输出
- decoder/seq2seq 分支
- is_3Di / encoder_only 开关
"""

import argparse
import csv
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from urllib import request

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, set_seed

set_seed(42)

# 置信度相关（基于 CNN 对 residue embedding 的 softmax 最大概率均值）
SS_MAPPING = {i: c for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
CNN_WEIGHTS_URL = "https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"
CNN_LOCAL_PATH = Path.cwd() / "cnn_chkpnt" / "model.pt"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        return self.classifier(x).squeeze(dim=-1)


_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_GLOBAL_PREDICTOR = None
_GLOBAL_DEVICE = None


def parse_args():
    p = argparse.ArgumentParser(description="ProstT5 embedding 生成脚本（精简版）")
    p.add_argument("-i", "--input", required=True, help="输入 FASTA 文件或目录")
    p.add_argument("-o", "--output", required=True, help="输出目录")
    p.add_argument("-n", "--cpu", type=int, default=4, help="CPU 进程数（GPU 模式下自动使用 1）")
    p.add_argument("--model-path", "--model", dest="model_path", required=True, help="ProstT5 模型目录")
    p.add_argument("--cnn-weights", default=None, help="CNN 权重路径（默认使用 cwd/cnn_chkpnt/model.pt，不存在则自动下载）")
    p.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    return p.parse_args()


def safe_file_stem(seq_id: str) -> str:
    primary = re.split(r"[;|/,]", seq_id, maxsplit=1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", primary)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned if cleaned else "seq"


def ensure_cnn_weights(cnn_path: Path):
    if not cnn_path.exists():
        print("📥 CNN 权重不存在，正在下载...")
        cnn_path.parent.mkdir(parents=True, exist_ok=True)
        req = request.Request(CNN_WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with request.urlopen(req) as response, open(cnn_path, "wb") as f:
            f.write(response.read())
        print("✅ CNN 权重下载完成")


def read_fasta_file(file_path: Path):
    seqs = {}
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        curr, curr_lines = None, []
        for line in f:
            if line.startswith(">"):
                if curr is not None:
                    seqs[curr] = "".join(curr_lines)
                curr = line[1:].split()[0]
                curr_lines = []
            else:
                curr_lines.append(line.strip())
        if curr is not None:
            seqs[curr] = "".join(curr_lines)
    return seqs


def init_worker(model_dir, threads_per_proc, device_str, cnn_weights_path):
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR, _GLOBAL_DEVICE

    torch.set_num_threads(threads_per_proc)
    device = torch.device(device_str)
    _GLOBAL_DEVICE = device

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = T5Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model = T5EncoderModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=dtype).to(device).eval()

    predictor = CNN().to(device).eval()
    checkpoint = torch.load(cnn_weights_path, map_location="cpu")["state_dict"]
    predictor.load_state_dict(checkpoint)
    predictor = predictor.to(dtype=dtype).eval()

    _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR = model, tokenizer, predictor


def embed_core(seq: str):
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR, _GLOBAL_DEVICE

    clean_seq = "".join([aa if aa in STANDARD_AA else "X" for aa in seq.upper()])
    encoded = _GLOBAL_TOKENIZER("<AA2fold> " + " ".join(clean_seq), return_tensors="pt").to(_GLOBAL_DEVICE)

    with torch.no_grad():
        emb_full = _GLOBAL_MODEL(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask).last_hidden_state
        residue_emb = emb_full[:, 1:]  # 去掉起始 token

        # 置信度
        pred = _GLOBAL_PREDICTOR(residue_emb)
        probs = F.softmax(pred, dim=1)
        max_p, _ = torch.max(probs, dim=1)
        avg_conf = float(max_p.mean().item())

    return residue_emb.squeeze(0), avg_conf


def embed_with_window(seq: str):
    L = len(seq)
    window, overlap = 1024, 64

    if L <= window:
        emb, conf = embed_core(seq)
        return emb, conf, L

    embs, confs = [], []
    for start in range(0, L, window - overlap):
        end = min(L, start + window)
        emb, conf = embed_core(seq[start:end])

        if not embs:
            embs.append(emb)
        else:
            embs.append(emb[overlap:])

        confs.append(conf)
        if end >= L:
            break

    return torch.cat(embs, dim=0), float(sum(confs) / len(confs)), L


def worker_fn(args_tuple):
    sid, seq, out_dir = args_tuple
    out_path = Path(out_dir)
    sid_file = safe_file_stem(sid)

    out_emb = out_path / f"{sid_file}_embedding.pt"

    emb, conf, seq_len = embed_with_window(seq)
    torch.save(emb.detach().to("cpu").half(), out_emb)

    return {
        "seq_id": sid,
        "seq_len": int(seq_len),
        "avg_conf": float(conf),
        "emb_file": out_emb.name,
    }


def main():
    args = parse_args()
    cnn_weights_path = Path(args.cnn_weights).resolve() if args.cnn_weights else CNN_LOCAL_PATH.resolve()
    ensure_cnn_weights(cnn_weights_path)

    device_str = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA 不可用，但指定了 --device={device_str}")

    use_cuda = device_str.startswith("cuda")
    nproc_effective = 1 if use_cuda else max(1, int(args.cpu))
    threads_per_proc = 1 if use_cuda else max(1, cpu_count() // nproc_effective)

    input_path = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        fasta_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in {".fasta", ".fa", ".faa"}])
    else:
        fasta_files = [input_path]

    if not fasta_files:
        raise FileNotFoundError(f"未找到 FASTA 文件: {input_path}")

    print(f"🚀 初始化: device={device_str}, nproc={nproc_effective}")
    print("🧠 载入模型（全局一次）...")

    pool = None
    if use_cuda:
        init_worker(args.model_path, threads_per_proc, device_str, str(cnn_weights_path))
    else:
        pool = Pool(
            processes=nproc_effective,
            initializer=init_worker,
            initargs=(args.model_path, threads_per_proc, device_str, str(cnn_weights_path)),
        )

    try:
        for fasta_file in fasta_files:
            run_out_dir = out_root / fasta_file.stem if input_path.is_dir() else out_root
            run_out_dir.mkdir(parents=True, exist_ok=True)

            seqs = read_fasta_file(fasta_file)
            if not seqs:
                print(f"⚠️ 空文件，跳过: {fasta_file}")
                continue

            job_args = [(sid, seq, str(run_out_dir)) for sid, seq in seqs.items()]
            idx_path = run_out_dir / "checkpoint.idx"
            summary_path = run_out_dir / "summary.csv"

            completed_ids = set()
            existing_results = {}

            # 断点恢复
            if idx_path.exists() and summary_path.exists():
                try:
                    with idx_path.open("r", encoding="utf-8") as idx_f:
                        idx_set = {line.strip() for line in idx_f if line.strip()}
                    with summary_path.open("r", newline="", encoding="utf-8") as sf:
                        for row in csv.DictReader(sf):
                            sid = row.get("seq_id")
                            if sid in idx_set:
                                completed_ids.add(sid)
                                existing_results[sid] = row
                except Exception:
                    completed_ids, existing_results = set(), {}

            todo_jobs = [j for j in job_args if j[0] not in completed_ids]

            if len(completed_ids) == len(job_args):
                print(f"⏭️  {fasta_file.name} 已全部完成，跳过")
                continue

            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["seq_id", "seq_len", "avg_conf", "emb_file"])
                writer.writeheader()

                for job in job_args:
                    if job[0] in completed_ids:
                        writer.writerow(existing_results[job[0]])

                if todo_jobs:
                    with idx_path.open("a", buffering=1, encoding="utf-8") as idx_f:
                        if use_cuda:
                            for job in tqdm(todo_jobs, total=len(todo_jobs), desc=f"Embedding {fasta_file.name}"):
                                res = worker_fn(job)
                                writer.writerow(res)
                                idx_f.write(f"{job[0]}\n")
                        else:
                            for res in tqdm(pool.imap_unordered(worker_fn, todo_jobs), total=len(todo_jobs), desc=f"Embedding {fasta_file.name}"):
                                writer.writerow(res)
                                idx_f.write(f"{res['seq_id']}\n")

            print(f"✅ 完成: {run_out_dir}")

    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
