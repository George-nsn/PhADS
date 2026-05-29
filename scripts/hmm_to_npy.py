#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 hmmscan domtblout 生成 HMM 特征矩阵（Min-Max 归一化版本）

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# 来自现有 cluster_merge_final 的簇数
DEFAULT_CLUSTER_COUNT = 211


def parse_args():
    parser = argparse.ArgumentParser(description="根据 hmmscan 结果自动生成进化流特征矩阵（Min-Max 归一化）")
    parser.add_argument("-d", "--domtblout", required=True, help="hmmscan 结果文件路径 (domtblout)")
    parser.add_argument("-i", "--input-fasta", required=True, help="输入 FASTA 路径（用于确定 query 顺序）")
    parser.add_argument("-k", "--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT,
                        help=f"簇数 K（默认内置: {DEFAULT_CLUSTER_COUNT}）")
    parser.add_argument("-on", "--output-npy", default="hmm_features_minmax.npy", help="输出的二进制矩阵路径")
    parser.add_argument("-ot", "--output-txt", default="hmm_features_minmax.txt", help="输出的文本矩阵路径")
    return parser.parse_args()


def read_fasta_ids(path: str):
    if not os.path.exists(path):
        print(f"❌ 致命错误：找不到 FASTA 文件 '{path}'")
        sys.exit(1)
    ids = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                sid = line[1:].strip().split()[0]
                if sid:
                    ids.append(sid)
    if not ids:
        print(f"❌ 错误：未在 FASTA 中读取到任何序列 ID: {path}")
        sys.exit(1)
    return ids


def minmax_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """按行 Min-Max 归一化，常量行保持全 0。"""
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
    """从 hmmscan target 名称提取簇编号并映射到 [0, K)。
    规则：提取最后一个数字串 N，映射为 N-1。
    """
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
    # 1) 读取 query 顺序（替代 protein_ids.txt）
    print("📥 正在从 FASTA 读取 query ID 顺序...")
    protein_ids = read_fasta_ids(input_fasta)
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    N = len(protein_ids)
    K = int(cluster_count)
    if K <= 0:
        print(f"❌ cluster_count 必须 > 0，当前: {K}")
        sys.exit(1)

    print(f"📊 矩阵初始化参数：蛋白质总数(N) = {N}, 内置簇总数(K) = {K}")

    # 2) 读取 domtblout
    print(f"🔍 正在高效解析 hmmscan 结果 '{domtblout_path}' ...")
    if not os.path.exists(domtblout_path):
        print(f"❌ 致命错误：找不到结果文件 '{domtblout_path}'！")
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

    # 3) 过滤到 query 集合
    df = df[df['query'].isin(id_to_idx)].copy()
    if df.empty:
        print("❌ 错误：未能从 hmmscan 结果中匹配到任何输入 FASTA 的 query ID！")
        sys.exit(1)

    # 4) target -> k_idx（不再依赖 cluster 文件）
    df['k_idx'] = df['target'].map(lambda x: target_to_cluster_idx(x, K))
    df = df.dropna(subset=['k_idx']).copy()
    df['k_idx'] = df['k_idx'].astype(int)
    df['i_idx'] = df['query'].map(id_to_idx)

    if df.empty:
        print("❌ 错误：target 未能映射到任何有效簇索引，请检查 hmmscan target 命名。")
        sys.exit(1)

    # 5) 每个 (i,k) 保留最大 bit_score
    df_grouped = df.groupby(['i_idx', 'k_idx'])['bit_score'].max().reset_index()

    # 6) 填充矩阵
    hmm_matrix = np.zeros((N, K), dtype=np.float32)
    i_indices = df_grouped['i_idx'].values.astype(int)
    k_indices = df_grouped['k_idx'].values.astype(int)
    scores = df_grouped['bit_score'].values.astype(np.float32)
    hmm_matrix[i_indices, k_indices] = scores
    print(f"✅ 成功提取并清洗了 {len(df_grouped)} 条唯一 HMM 比对得分对。")

    # 7) Min-Max 归一化（仅此方法）
    print("🧮 正在对 HMM 矩阵执行 Min-Max 归一化（按行）...")
    hmm_features = minmax_normalize_rows(hmm_matrix)

    # 8) 输出
    np.save(output_npy, hmm_features)
    print(f"💾 正在导出文本矩阵 {output_txt} ...")
    np.savetxt(output_txt, hmm_features, fmt='%.6f', delimiter='\t')

    print("\n" + "=" * 45)
    print("🎉 成功！进化流特征矩阵已生成（Min-Max）。")
    print(f"👉 矩阵维度：{hmm_features.shape}  (N={N}, K={K})")
    print(f"👉 NPY 矩阵已保存：{output_npy}")
    print(f"👉 TXT 矩阵已保存：{output_txt}")
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
