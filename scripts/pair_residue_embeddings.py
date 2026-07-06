#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pair ESM2 and ProtT5 residue embeddings for residue Pool-Gate inference."""

import argparse
import csv
import re
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Pack ESM2/ProtT5 residue embeddings into paired_pt files")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--esm-root", required=True)
    parser.add_argument("--prost-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--esm-dim", type=int, default=1280)
    parser.add_argument("--prost-dim", type=int, default=1024)
    parser.add_argument("--dtype", choices=["keep", "float16", "float32"], default="float16")
    parser.add_argument("--allow-length-mismatch", action="store_true")
    return parser.parse_args()


def safe_file_stem(seq_id: str) -> str:
    primary = re.split(r"[;|/,]", seq_id, maxsplit=1)[0] or seq_id
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", primary)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned if cleaned else "seq"


def read_ids(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_summary_index(root: Path):
    index = {}
    for summary in root.glob("**/summary.csv"):
        try:
            with summary.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    seq_id = str(row.get("seq_id", "")).strip()
                    emb_file = str(row.get("emb_file", "")).strip()
                    if not seq_id or not emb_file:
                        continue
                    candidate = summary.parent / emb_file
                    if candidate.exists() and candidate.is_file():
                        index[seq_id] = candidate
                        index[safe_file_stem(seq_id)] = candidate
        except Exception:
            continue
    return index


def build_pt_index(root: Path):
    index = {}
    for path in sorted(root.glob("**/*.pt")):
        if not path.is_file():
            continue
        stem = path.stem
        keys = {stem}
        if stem.endswith("_embedding"):
            keys.add(stem[: -len("_embedding")])
        for key in keys:
            index.setdefault(key, path)
    return index


def candidate_paths(root: Path, seq_id: str, suffixes):
    stem = safe_file_stem(seq_id)
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    names = []
    for suffix in suffixes:
        names.extend([
            f"{stem}_{suffix}_residue.pt",
            f"{seq_id}_{suffix}_residue.pt",
            f"{stem}_{suffix}.pt",
            f"{seq_id}_{suffix}.pt",
        ])
    names.extend([
        f"{stem}_embedding.pt",
        f"{stem}.pt",
        f"{seq_id}_embedding.pt",
        f"{seq_id}.pt",
    ])
    paths = []
    for name in names:
        paths.extend([root / name, root / stem / name, root / seq_id / name])
    return paths


def find_tensor_file(root: Path, seq_id: str, suffixes, summary_index=None, pt_index=None):
    for path in candidate_paths(root, seq_id, suffixes):
        if path.exists() and path.is_file():
            return path
    if summary_index:
        for key in [seq_id, safe_file_stem(seq_id)]:
            path = summary_index.get(key)
            if path and path.exists() and path.is_file():
                return path
    if pt_index:
        for key in [safe_file_stem(seq_id), seq_id, f"{safe_file_stem(seq_id)}_embedding", f"{seq_id}_embedding"]:
            path = pt_index.get(key)
            if path and path.exists() and path.is_file():
                return path
    stem = safe_file_stem(seq_id)
    patterns = []
    for suffix in (suffixes if isinstance(suffixes, list) else [suffixes]):
        patterns.append(f"**/{stem}*{suffix}*.pt")
    patterns.extend([f"**/{stem}*_embedding.pt", f"**/{stem}.pt"])
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    tensor = None
    if isinstance(obj, dict):
        for key in ("embedding", "emb", "residue", "residue_embedding", "esm2", "prost5", "features"):
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


def cast_tensor(tensor: torch.Tensor, dtype: str):
    if dtype == "float16":
        return tensor.half()
    if dtype == "float32":
        return tensor.float()
    return tensor


def main():
    args = parse_args()
    ids = read_ids(Path(args.ids))
    esm_root = Path(args.esm_root)
    prost_root = Path(args.prost_root)
    out_dir = Path(args.output_dir)
    paired_dir = out_dir / "paired_pt"
    paired_dir.mkdir(parents=True, exist_ok=True)
    esm_summary_index = load_summary_index(esm_root)
    prost_summary_index = load_summary_index(prost_root)
    esm_pt_index = build_pt_index(esm_root)
    prost_pt_index = build_pt_index(prost_root)
    rows = []
    missing = []

    for seq_id in tqdm(ids, desc="Pairing residue embeddings"):
        esm_path = find_tensor_file(esm_root, seq_id, ["esm2", "embedding"], esm_summary_index, esm_pt_index)
        prost_path = find_tensor_file(prost_root, seq_id, ["prot5", "prost5", "protxl", "prot_xl", "embedding"], prost_summary_index, prost_pt_index)
        if esm_path is None or prost_path is None:
            reason = f"missing_esm={esm_path is None}; missing_prot={prost_path is None}"
            missing.append(f"{seq_id}\t{reason}")
            rows.append({"seq_id": seq_id, "status": "missing", "length": 0, "esm_file": "", "prost_file": "", "paired_file": "", "reason": reason})
            continue
        try:
            esm = load_tensor(esm_path)
            prost = load_tensor(prost_path)
            if esm.shape[1] != args.esm_dim:
                raise ValueError(f"ESM2 dim mismatch: {tuple(esm.shape)} expected {args.esm_dim}")
            if prost.shape[1] != args.prost_dim:
                raise ValueError(f"ProtT5 dim mismatch: {tuple(prost.shape)} expected {args.prost_dim}")
            if esm.shape[0] != prost.shape[0]:
                if not args.allow_length_mismatch:
                    raise ValueError(f"length mismatch: esm={esm.shape[0]}, prost={prost.shape[0]}")
                min_len = min(esm.shape[0], prost.shape[0])
                esm = esm[:min_len]
                prost = prost[:min_len]
            length = int(esm.shape[0])
            mask = torch.ones(length, dtype=torch.bool)
            paired_file = f"{safe_file_stem(seq_id)}.pt"
            torch.save({
                "seq_id": seq_id,
                "length": length,
                "esm2": cast_tensor(esm, args.dtype),
                "prost5": cast_tensor(prost, args.dtype),
                "mask": mask,
                "source_esm2": str(esm_path),
                "source_prost5": str(prost_path),
                "feature_level": "residue",
            }, paired_dir / paired_file)
            rows.append({
                "seq_id": seq_id,
                "status": "ok",
                "length": length,
                "esm_file": str(esm_path),
                "prost_file": str(prost_path),
                "paired_file": str(Path("paired_pt") / paired_file),
                "reason": "",
            })
        except Exception as exc:
            reason = f"error={exc}"
            missing.append(f"{seq_id}\t{reason}")
            rows.append({"seq_id": seq_id, "status": "error", "length": 0, "esm_file": str(esm_path), "prost_file": str(prost_path), "paired_file": "", "reason": reason})

    with (out_dir / "residue_pair_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq_id", "status", "length", "esm_file", "prost_file", "paired_file", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    if missing:
        (out_dir / "missing_residue_pairs.log").write_text("\n".join(missing) + "\n", encoding="utf-8")
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"Paired residue embeddings: ok={ok_count}/{len(rows)} -> {out_dir}")
    if ok_count != len(rows):
        if missing:
            print(f"Pairing failed for {len(missing)} sequence(s). See {out_dir / 'missing_residue_pairs.log'} for details.")
            for item in missing[:5]:
                print(f"  {item}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
