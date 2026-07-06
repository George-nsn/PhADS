#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate residue-level ESM2 or ProtT5 embeddings for PHADS inference."""

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
from transformers import AutoTokenizer, EsmModel, T5Tokenizer, T5EncoderModel, set_seed

try:
    import onnxruntime as ort
except ImportError:
    ort = None

set_seed(42)

# Confidence estimate based on the mean maximum softmax probability of the CNN head.
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
_GLOBAL_MODEL_KIND = None
_GLOBAL_ONNX_SESSION = None
_GLOBAL_ESM_LAYER = 30


def parse_args():
    p = argparse.ArgumentParser(description="Generate ESM2 or ProtT5 residue embeddings")
    p.add_argument("-i", "--input", required=True, help="Input FASTA file or directory")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("-n", "--cpu", type=int, default=4, help="Number of CPU worker processes; GPU mode uses one process")
    p.add_argument("--model-path", "--model", dest="model_path", required=True, help="ESM2 or ProtT5 model directory")
    p.add_argument("--model-kind", choices=["auto", "esm2", "prostt5", "prostt5_onnx"], default="auto",
                   help="Model type: auto/esm2/prostt5/prostt5_onnx")
    p.add_argument("--esm-layer", type=int, default=30, help="ESM2 hidden state layer to export when --model-kind esm2")
    p.add_argument("--cnn-weights", default=None, help="Path to ProstT5 CNN weights; ignored for ESM2/ONNX mode")
    p.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    return p.parse_args()


def safe_file_stem(seq_id: str) -> str:
    primary = re.split(r"[;|/,]", seq_id, maxsplit=1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", primary)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned if cleaned else "seq"


def ensure_cnn_weights(cnn_path: Path):
    if not cnn_path.exists():
        print("CNN weights are missing; downloading required weights...")
        cnn_path.parent.mkdir(parents=True, exist_ok=True)
        req = request.Request(CNN_WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with request.urlopen(req) as response, open(cnn_path, "wb") as f:
            f.write(response.read())
        print("CNN weights downloaded successfully")


def detect_model_kind(model_dir: str, model_kind: str) -> str:
    if model_kind != "auto":
        return model_kind
    name = Path(model_dir).name.lower()
    if "esm2" in name or "esm-2" in name:
        return "esm2"
    if "onnx" in name:
        return "prostt5_onnx"
    return "prostt5"


def find_onnx_file(model_dir: str) -> Path:
    root = Path(model_dir)
    preferred = ["model.onnx", "encoder_model.onnx", "prot_t5_xl_uniref50.onnx"]
    for name in preferred:
        p = root / name
        if p.exists():
            return p
    matches = sorted(root.rglob("*.onnx"))
    if not matches:
        raise FileNotFoundError(f"No .onnx file was found in ONNX model directory: {root}")
    return matches[0]


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


def init_worker(model_dir, threads_per_proc, device_str, cnn_weights_path, model_kind, esm_layer=30):
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR, _GLOBAL_DEVICE, _GLOBAL_MODEL_KIND, _GLOBAL_ONNX_SESSION, _GLOBAL_ESM_LAYER

    torch.set_num_threads(threads_per_proc)
    device = torch.device(device_str)
    _GLOBAL_DEVICE = device
    _GLOBAL_MODEL_KIND = model_kind
    _GLOBAL_ESM_LAYER = int(esm_layer)

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    if model_kind == "esm2":
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = EsmModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=dtype).to(device).eval()
        _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR = model, tokenizer, None
        return

    tokenizer = T5Tokenizer.from_pretrained(model_dir, local_files_only=True)
    if model_kind == "prostt5_onnx":
        if ort is None:
            raise ImportError("The prot-t5 ONNX model requires onnxruntime")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        _GLOBAL_ONNX_SESSION = ort.InferenceSession(str(find_onnx_file(model_dir)), providers=providers)
        _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR = None, tokenizer, None
        return

    model = T5EncoderModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=dtype).to(device).eval()
    predictor = CNN().to(device).eval()
    checkpoint = torch.load(cnn_weights_path, map_location="cpu")["state_dict"]
    predictor.load_state_dict(checkpoint)
    predictor = predictor.to(dtype=dtype).eval()
    _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR = model, tokenizer, predictor


def embed_core(seq: str):
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR, _GLOBAL_DEVICE, _GLOBAL_MODEL_KIND, _GLOBAL_ONNX_SESSION

    clean_seq = "".join([aa if aa in STANDARD_AA else "X" for aa in seq.upper()])

    if _GLOBAL_MODEL_KIND == "esm2":
        encoded = _GLOBAL_TOKENIZER(clean_seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
        encoded = {k: v.to(_GLOBAL_DEVICE) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = _GLOBAL_MODEL(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[_GLOBAL_ESM_LAYER] if outputs.hidden_states and len(outputs.hidden_states) > _GLOBAL_ESM_LAYER else outputs.last_hidden_state
        return hidden[0, 1:-1, :].detach().cpu(), float("nan")

    encoded = _GLOBAL_TOKENIZER("<AA2fold> " + " ".join(clean_seq), return_tensors="pt").to(_GLOBAL_DEVICE)

    if _GLOBAL_MODEL_KIND == "prostt5_onnx":
        inputs = {
            "input_ids": encoded.input_ids.detach().cpu().numpy(),
            "attention_mask": encoded.attention_mask.detach().cpu().numpy(),
        }
        input_names = {i.name for i in _GLOBAL_ONNX_SESSION.get_inputs()}
        inputs = {k: v for k, v in inputs.items() if k in input_names}
        with torch.no_grad():
            out = _GLOBAL_ONNX_SESSION.run(None, inputs)[0]
        return torch.from_numpy(out[:, 1:, :]).squeeze(0).float(), float("nan")

    with torch.no_grad():
        emb_full = _GLOBAL_MODEL(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask).last_hidden_state
        residue_emb = emb_full[:, 1:]

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
    model_kind = detect_model_kind(args.model_path, args.model_kind)
    cnn_weights_path = Path(args.cnn_weights).resolve() if args.cnn_weights else CNN_LOCAL_PATH.resolve()
    if model_kind == "prostt5":
        ensure_cnn_weights(cnn_weights_path)

    device_str = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if args.device != "auto":
        device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available, but --device={device_str} was requested")

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
        raise FileNotFoundError(f"No FASTA files were found: {input_path}")

    print(f"[INIT] model_kind={model_kind}, device={device_str}, nproc={nproc_effective}")
    print("[LOAD] Loading model once for the worker context...")

    pool = None
    if use_cuda:
        init_worker(args.model_path, threads_per_proc, device_str, str(cnn_weights_path), model_kind, args.esm_layer)
    else:
        pool = Pool(
            processes=nproc_effective,
            initializer=init_worker,
            initargs=(args.model_path, threads_per_proc, device_str, str(cnn_weights_path), model_kind, args.esm_layer),
        )

    try:
        for fasta_file in fasta_files:
            run_out_dir = out_root / fasta_file.stem if input_path.is_dir() else out_root
            run_out_dir.mkdir(parents=True, exist_ok=True)

            seqs = read_fasta_file(fasta_file)
            if not seqs:
                print(f"[WARN] Empty FASTA file skipped: {fasta_file}")
                continue

            job_args = [(sid, seq, str(run_out_dir)) for sid, seq in seqs.items()]
            idx_path = run_out_dir / "checkpoint.idx"
            summary_path = run_out_dir / "summary.csv"

            completed_ids = set()
            existing_results = {}

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
                print(f"[SKIP] All sequences have already been processed for {fasta_file.name}")
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

            print(f"[DONE] Embeddings written to: {run_out_dir}")

    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
