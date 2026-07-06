#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate a min-max-normalized HMM feature matrix from hmmscan domtblout output."""

import os
import sys
import argparse
import numpy as np
import pandas as pd

DEFAULT_CLUSTER_COUNT = 211


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a min-max-normalized evolutionary feature matrix from hmmscan results")
    parser.add_argument("-d", "--domtblout", required=True, help="Path to the hmmscan domtblout file")
    parser.add_argument("-i", "--input-fasta", required=True, help="Input FASTA path used to determine query order")
    parser.add_argument("-k", "--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT,
                        help=f"Number of HMM feature columns K (default: {DEFAULT_CLUSTER_COUNT})")
    parser.add_argument("-on", "--output-npy", default="hmm_features_minmax.npy", help="Output path for the binary NumPy matrix")
    parser.add_argument("-ot", "--output-txt", default="hmm_features_minmax.txt", help="Output path for the tab-delimited text matrix")
    return parser.parse_args()


def read_fasta_ids(path: str):
    if not os.path.exists(path):
        print(f"FATAL: FASTA file not found: {path}")
        sys.exit(1)
    ids = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                sid = line[1:].strip().split()[0]
                if sid:
                    ids.append(sid)
    if not ids:
        print(f"ERROR: no sequence IDs were found in FASTA file: {path}")
        sys.exit(1)
    return ids


def minmax_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Apply row-wise min-max normalization; constant rows remain zero."""
    x = matrix.astype(np.float32, copy=True)
    row_min = x.min(axis=1, keepdims=True)
    row_max = x.max(axis=1, keepdims=True)
    denom = row_max - row_min
    out = np.zeros_like(x, dtype=np.float32)
    valid = (denom.squeeze(1) > 0)
    if np.any(valid):
        out[valid] = (x[valid] - row_min[valid]) / denom[valid]
    return out


def target_to_cluster_idx(target: str, K: int):
    """Extract the last numeric token from an hmmscan target and map it to [0, K)."""
    import re
    m = re.search(r"(\d+)(?!.*\d)", str(target))
    if not m:
        return None
    k = int(m.group(1)) - 1
    if 0 <= k < K:
        return k
    return None


def generate_hmm_matrix(domtblout_path, input_fasta, cluster_count,
                        output_npy="hmm_features_minmax.npy", output_txt="hmm_features_minmax.txt"):
    print("Reading query ID order from FASTA...")
    protein_ids = read_fasta_ids(input_fasta)
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    N = len(protein_ids)
    K = int(cluster_count)
    if K <= 0:
        print(f"ERROR: cluster_count must be > 0; received {K}")
        sys.exit(1)

    print(f"Matrix initialization: N={N} sequences, K={K} HMM feature columns")

    print(f"Parsing hmmscan domtblout file: {domtblout_path}")
    if not os.path.exists(domtblout_path):
        print(f"FATAL: hmmscan domtblout file not found: {domtblout_path}")
        sys.exit(1)

    df = pd.read_csv(
        domtblout_path,
        sep=r'\s+',
        comment='#',
        header=None,
        usecols=[0, 3, 13],
        names=['target', 'query', 'bit_score'],
        dtype={'target': str, 'query': str, 'bit_score': float}
    )

    # Keep only records belonging to the input query set.
    df = df[df['query'].isin(id_to_idx)].copy()
    if df.empty:
        print("ERROR: no hmmscan records matched the input FASTA query IDs")
        sys.exit(1)

    # Map hmmscan target names to column indices.
    df['k_idx'] = df['target'].map(lambda x: target_to_cluster_idx(x, K))
    df = df.dropna(subset=['k_idx']).copy()
    df['k_idx'] = df['k_idx'].astype(int)
    df['i_idx'] = df['query'].map(id_to_idx)

    if df.empty:
        print("ERROR: no hmmscan target names could be mapped to valid cluster indices")
        sys.exit(1)

    # Retain the maximum bit score for each query-column pair.
    df_grouped = df.groupby(['i_idx', 'k_idx'])['bit_score'].max().reset_index()

    # Populate the dense HMM matrix.
    hmm_matrix = np.zeros((N, K), dtype=np.float32)
    i_indices = df_grouped['i_idx'].values.astype(int)
    k_indices = df_grouped['k_idx'].values.astype(int)
    scores = df_grouped['bit_score'].values.astype(np.float32)
    hmm_matrix[i_indices, k_indices] = scores
    print(f"Extracted {len(df_grouped)} unique query-HMM score pairs")

    print("Applying row-wise min-max normalization to the HMM matrix...")
    hmm_features = minmax_normalize_rows(hmm_matrix)

    # Write outputs.
    np.save(output_npy, hmm_features)
    print(f"Writing text matrix: {output_txt}")
    np.savetxt(output_txt, hmm_features, fmt='%.6f', delimiter='\t')

    print("\n" + "=" * 45)
    print("HMM evolutionary feature matrix generated successfully (min-max normalized).")
    print(f"Matrix shape: {hmm_features.shape} (N={N}, K={K})")
    print(f"NPY output: {output_npy}")
    print(f"Text output: {output_txt}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    args = parse_args()
    generate_hmm_matrix(
        domtblout_path=args.domtblout,
        input_fasta=args.input_fasta,
        cluster_count=args.cluster_count,
        output_npy=args.output_npy,
        output_txt=args.output_txt,
    )
