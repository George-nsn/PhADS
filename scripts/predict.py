import os
import argparse
import csv
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDHProtoNet(nn.Module):
    """从 train_sdh_protonet_new.py 内联的推理所需模型定义。"""
    def __init__(self, seq_dim=1024, evo_dim=195, latent_dim=512):
        super().__init__()
        self.seq_net = nn.Sequential(nn.Linear(seq_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())
        self.evo_net = nn.Sequential(nn.Linear(evo_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(latent_dim * 2, 1), nn.Sigmoid())
        self.projector = nn.Linear(latent_dim, latent_dim)

    def forward(self, x_seq, x_evo, return_gate=False):
        f_s, f_e = self.seq_net(x_seq), self.evo_net(x_evo)
        g = self.gate(torch.cat([f_s, f_e], dim=-1))
        fused = g * f_s + (1 - g) * f_e
        z = F.normalize(self.projector(fused), p=2, dim=1)
        if return_gate:
            return z, g.squeeze(-1)
        return z


def parse_args():
    p = argparse.ArgumentParser(description="SDH-ProtoNet 预测脚本（prototype/instance 双模式，含注释增强）")

    p.add_argument("--model-path", default="sdh_protonet_best.pth", help="模型权重路径")
    p.add_argument("--map-path", default="family_map.pth", help="地图文件路径（generate_prototypes.py 生成）")

    p.add_argument("--query-emb", required=True, help="待预测序列嵌入 .npy [N, seq_dim] 或 [seq_dim]")
    p.add_argument("--query-hmm", required=True, help="待预测 HMM 特征 .npy [N, evo_dim] 或 [evo_dim]")
    p.add_argument("--query-ids", default="", help="待预测样本ID文本（每行一个，可选）")

    p.add_argument("--mode", choices=["prototype", "instance"], default="prototype", help="预测模式")
    p.add_argument("--batch-size", type=int, default=1024, help="推理批大小")
    p.add_argument("--temperature", type=float, default=None, help="softmax 温度；默认用地图文件中的值")
    p.add_argument("--topk", type=int, default=5, help="输出前k个候选")

    p.add_argument("--cluster-annotation-txt", default="cluster_annotation.txt", help="簇注释TXT（TSV）")
    p.add_argument("--ads-function-txt", default="ADS_function.txt", help="ADS功能TXT（TSV）")

    p.add_argument("--output-tsv", default="prediction_results.tsv", help="主预测输出表")
    p.add_argument("--topk-tsv", default="prediction_topk.tsv", help="Top-k 明细输出表")
    p.add_argument("--print-topk", action="store_true", help="终端打印 top-k")

    return p.parse_args()


def load_query_ids(path: str, n: int) -> List[str]:
    if not path or (not os.path.exists(path)):
        return [f"query_{i}" for i in range(n)]
    with open(path, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if len(ids) != n:
        print(f"⚠️ query-ids 行数({len(ids)})与样本数({n})不一致，将使用默认 query_i")
        return [f"query_{i}" for i in range(n)]
    return ids


def ensure_2d(a: np.ndarray, name: str) -> np.ndarray:
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"{name} 必须是一维或二维数组，当前 shape={a.shape}")
    return a


def infer_latent_dim(state_dict: Dict[str, torch.Tensor], fallback: int = 512) -> int:
    for k, v in state_dict.items():
        if k.endswith("projector.weight") and v.ndim == 2:
            return int(v.shape[0])
    return fallback


def is_missing(v: str) -> bool:
    s = str(v).strip()
    return (not s) or s.lower() in {"nan", "none", "na", "-"}


def load_cluster_annotation(path: str):
    """读取 cluster_annotation.txt，返回 label->dict 与从第4列开始的列名列表。"""
    if not os.path.exists(path):
        print(f"⚠️ 未找到 cluster 注释文件: {path}")
        return {}, []

    out: Dict[int, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_idx = {c: i for i, c in enumerate(header)}

        if "label" not in col_idx:
            print(f"⚠️ cluster_annotation 缺少 label 列，当前列: {header}")
            return {}, []

        func_cols = header[3:] if len(header) > 3 else []  # 从第4列开始
        label_i = col_idx["label"]

        for line in f:
            parts = line.rstrip("\n").split("\t")
            try:
                lb = int(parts[label_i])
            except Exception:
                continue

            one = {}
            for c in func_cols:
                i = col_idx[c]
                val = parts[i].strip() if i < len(parts) else ""
                one[c] = "" if is_missing(val) else val
            out[lb] = one

    return out, func_cols


def load_ads_function(path: str) -> Dict[str, str]:
    """读取 ADS_function.txt (TSV)，返回 ADS_name -> 功能汇总字符串"""
    if not os.path.exists(path):
        print(f"⚠️ 未找到 ADS 功能文件: {path}")
        return {}

    name_to_vals: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_idx = {c: i for i, c in enumerate(header)}
        if "ADS_name" not in col_idx or "Against/Function" not in col_idx:
            print(f"⚠️ ADS_function 列不完整，当前列: {header}")
            return {}

        ni = col_idx["ADS_name"]
        fi = col_idx["Against/Function"]

        for line in f:
            parts = line.rstrip("\n").split("\t")
            name = parts[ni].strip() if ni < len(parts) else ""
            func = parts[fi].strip() if fi < len(parts) else ""
            if is_missing(name) or is_missing(func):
                continue

            name_to_vals.setdefault(name, [])
            if func not in name_to_vals[name]:
                name_to_vals[name].append(func)

    return {k: "; ".join(vs) for k, vs in name_to_vals.items()}


def lookup_ads_function(seq_id: str, ads_map: Dict[str, str]) -> str:
    """按序列ID查功能。支持完全匹配；若像 WP_xxx.1 也尝试 WP_xxx。"""
    if not seq_id:
        return ""
    if seq_id in ads_map:
        return ads_map[seq_id]

    if "." in seq_id:
        base = seq_id.rsplit(".", 1)[0]
        if base in ads_map:
            return ads_map[base]

    return ""


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for p in [args.model_path, args.map_path, args.query_emb, args.query_hmm]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到文件: {p}")

    payload = torch.load(args.map_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("地图文件格式错误")

    cluster_annotation_map, cluster_func_cols = load_cluster_annotation(args.cluster_annotation_txt)
    ads_function_map = load_ads_function(args.ads_function_txt)

    q_emb = ensure_2d(np.load(args.query_emb), "query_emb")
    q_hmm = ensure_2d(np.load(args.query_hmm), "query_hmm")

    if q_emb.shape[0] != q_hmm.shape[0]:
        raise ValueError(f"query_emb 与 query_hmm 样本数不一致: {q_emb.shape[0]} vs {q_hmm.shape[0]}")

    n = q_emb.shape[0]
    query_ids = load_query_ids(args.query_ids, n)

    seq_dim_expected = int(payload.get("seq_dim", q_emb.shape[1]))
    evo_dim_expected = int(payload.get("evo_dim", q_hmm.shape[1]))
    if q_emb.shape[1] != seq_dim_expected:
        raise ValueError(f"query_emb 维度不匹配: got {q_emb.shape[1]}, expect {seq_dim_expected}")
    if q_hmm.shape[1] != evo_dim_expected:
        raise ValueError(f"query_hmm 维度不匹配: got {q_hmm.shape[1]}, expect {evo_dim_expected}")

    temp = float(args.temperature) if args.temperature is not None else float(payload.get("temperature", 1.0))
    if temp <= 0:
        raise ValueError("temperature 必须 > 0")

    state = torch.load(args.model_path, map_location="cpu")
    latent_dim = infer_latent_dim(state, fallback=int(payload.get("latent_dim", 512)))
    model = SDHProtoNet(seq_dim=seq_dim_expected, evo_dim=evo_dim_expected, latent_dim=latent_dim)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    cluster_names: Dict[int, str] = {int(k): v for k, v in payload.get("cluster_names", {}).items()}
    cluster_reps: Dict[int, str] = {int(k): v for k, v in payload.get("cluster_representatives", {}).items()}

    topk = max(1, int(args.topk))
    results = []

    # mode-specific gallery
    if args.mode == "prototype":
        if "prototypes" not in payload:
            raise ValueError("地图文件中没有 prototypes，请重新用 --mode prototype/both 生成")

        gallery = payload["prototypes"].float().to(device)  # [C, D]
        gallery_labels = [int(x) for x in payload["labels"]]
        gallery_ids = [f"prototype_of_{lb}" for lb in gallery_labels]
        target_k = min(topk, len(gallery_labels))

    else:  # instance
        if "instance_features" not in payload:
            raise ValueError("地图文件中没有 instance_features，请重新用 --mode instance/both 生成")

        gallery = payload["instance_features"].float().to(device)  # [N, D]
        gallery_labels = [int(x) for x in payload["instance_labels"]]
        gallery_ids = payload["instance_ids"]
        target_k = min(topk, len(gallery_labels))

    has_instance_gallery = ("instance_features" in payload and "instance_labels" in payload and "instance_ids" in payload)
    if has_instance_gallery:
        instance_gallery = payload["instance_features"].float().to(device)
        instance_labels = [int(x) for x in payload["instance_labels"]]
        instance_ids = payload["instance_ids"]

    with torch.no_grad():
        for st in range(0, n, args.batch_size):
            ed = min(st + args.batch_size, n)
            x_seq = torch.from_numpy(q_emb[st:ed]).float().to(device)
            x_evo = torch.from_numpy(q_hmm[st:ed]).float().to(device)

            qz = model(x_seq, x_evo)  # [b, D]
            d2 = torch.cdist(qz, gallery, p=2).pow(2)  # [b, G]

            probs = F.softmax(-d2 / temp, dim=1)
            top_probs, top_idx = torch.topk(probs, k=target_k, dim=1)
            top_d2 = torch.gather(d2, 1, top_idx)

            if has_instance_gallery:
                d2_ins = torch.cdist(qz, instance_gallery, p=2).pow(2)  # [b, N]
                ins_idx = torch.argmin(d2_ins, dim=1)
                ins_d2 = d2_ins[torch.arange(d2_ins.shape[0]), ins_idx]
            else:
                ins_idx = None
                ins_d2 = None

            for i in range(ed - st):
                best_idx = int(top_idx[i, 0].item())
                best_prob = float(top_probs[i, 0].item())
                best_d2 = float(top_d2[i, 0].item())
                best_label = int(gallery_labels[best_idx])
                best_id = str(gallery_ids[best_idx])

                if args.mode == "instance":
                    nearest_seq_id = best_id
                    nearest_seq_label = best_label
                    nearest_seq_distance2 = best_d2
                elif has_instance_gallery:
                    nidx = int(ins_idx[i].item())
                    nearest_seq_id = str(instance_ids[nidx])
                    nearest_seq_label = int(instance_labels[nidx])
                    nearest_seq_distance2 = float(ins_d2[i].item())
                else:
                    nearest_seq_id = ""
                    nearest_seq_label = -1
                    nearest_seq_distance2 = float("nan")

                pred_cluster_label = best_label
                cluster_func_values = cluster_annotation_map.get(pred_cluster_label, {})
                nearest_seq_function_summary = lookup_ads_function(nearest_seq_id, ads_function_map)

                cand = []
                for j in range(target_k):
                    gidx = int(top_idx[i, j].item())
                    lb = int(gallery_labels[gidx])
                    cid = str(gallery_ids[gidx])
                    cfunc = lookup_ads_function(cid, ads_function_map) if args.mode == "instance" else ""
                    cand.append({
                        "rank": j + 1,
                        "id": cid,
                        "label": lb,
                        "cluster": cluster_names.get(lb, f"cluster_{lb}"),
                        "cluster_rep": cluster_reps.get(lb, "NA"),
                        "distance2": float(top_d2[i, j].item()),
                        "confidence": float(top_probs[i, j].item()),
                        "ads_function": cfunc,
                    })

                results.append({
                    "query_id": query_ids[st + i],
                    "mode": args.mode,
                    "pred_id": best_id,
                    "pred_label": pred_cluster_label,
                    "pred_cluster": cluster_names.get(pred_cluster_label, f"cluster_{pred_cluster_label}"),
                    "pred_cluster_rep": cluster_reps.get(pred_cluster_label, "NA"),
                    "pred_distance2": best_d2,
                    "confidence": best_prob,
                    "nearest_sequence_id": nearest_seq_id,
                    "nearest_sequence_label": nearest_seq_label,
                    "nearest_sequence_distance2": nearest_seq_distance2,
                    "nearest_sequence_function_summary": nearest_seq_function_summary,
                    "cluster_func_values": cluster_func_values,
                    "topk": cand,
                })

    # 1) 主输出：删除 pred_label 与 topk_summary，cluster_func_* 追加在最后
    cluster_func_out_cols = [f"cluster_func_{c}" for c in cluster_func_cols]

    with open(args.output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            "query_id", "mode", "pred_id", "pred_cluster", "pred_cluster_rep",
            "pred_distance2", "confidence",
            "nearest_sequence_id", "nearest_sequence_label", "nearest_sequence_distance2", "nearest_sequence_function_summary",
            *cluster_func_out_cols,
        ])

        for r in results:
            cluster_func_vals = [r["cluster_func_values"].get(c, "") for c in cluster_func_cols]
            writer.writerow([
                r["query_id"], r["mode"], r["pred_id"], r["pred_cluster"], r["pred_cluster_rep"],
                f"{r['pred_distance2']:.6f}", f"{r['confidence']:.6f}",
                r["nearest_sequence_id"], r["nearest_sequence_label"],
                f"{r['nearest_sequence_distance2']:.6f}" if r["nearest_sequence_id"] else "",
                r["nearest_sequence_function_summary"],
                *cluster_func_vals,
            ])

    # 2) Top-k 单独输出（美化展开，不含 label）
    with open(args.topk_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            "query_id", "mode", "rank", "candidate_id", "candidate_cluster", "candidate_rep",
            "candidate_distance2", "candidate_confidence", "candidate_ads_function"
        ])
        for r in results:
            for c in r["topk"]:
                writer.writerow([
                    r["query_id"], r["mode"], c["rank"], c["id"], c["cluster"], c["cluster_rep"],
                    f"{c['distance2']:.6f}", f"{c['confidence']:.6f}", c["ads_function"],
                ])

    # 终端输出
    print(f"\n🔮 ==== 预测结果（mode={args.mode}） ====")
    for r in results:
        print(
            f"{r['query_id']}: id={r['pred_id']} | {r['pred_cluster']} (rep={r['pred_cluster_rep']}) "
            f"| d²={r['pred_distance2']:.6f} | conf={r['confidence']:.2%}"
        )
        if r["nearest_sequence_id"]:
            print(
                f"   - nearest_sequence: {r['nearest_sequence_id']} (label={r['nearest_sequence_label']}, d²={r['nearest_sequence_distance2']:.6f})"
            )
            if r["nearest_sequence_function_summary"]:
                print(f"     function: {r['nearest_sequence_function_summary']}")
        if args.print_topk:
            for c in r["topk"]:
                print(
                    f"   - top{c['rank']}: id={c['id']} | {c['cluster']} (rep={c['cluster_rep']}) "
                    f"d²={c['distance2']:.6f}, p={c['confidence']:.2%}"
                )

    print(f"\n✅ 主预测表已保存: {args.output_tsv}")
    print(f"✅ Top-k 明细表已保存: {args.topk_tsv}")
    if args.mode == "prototype":
        print("📌 置信度公式(prototype): p(c|x)=exp(-d_c^2/T)/Σ_j exp(-d_j^2/T)")
    else:
        print("📌 置信度公式(instance): p(i|x)=exp(-d_i^2/T)/Σ_j exp(-d_j^2/T)")


if __name__ == "__main__":
    main()
