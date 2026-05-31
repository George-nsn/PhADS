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
    p = argparse.ArgumentParser(description="SDH-ProtoNet 预测脚本（prototype/instance 双模式，含动态阈值过滤与注释增强）")

    p.add_argument("--model-path", default="sdh_protonet_best.pth", help="模型权重路径")
    p.add_argument("--map-path", default="family_map.pth", help="地图文件路径（generate_prototypes.py 生成）")

    p.add_argument("--query-emb", required=True, help="待预测序列嵌入 .npy [N, seq_dim] 或 [seq_dim]")
    p.add_argument("--query-hmm", required=True, help="待预测 HMM 特征 .npy [N, evo_dim] 或 [evo_dim]")
    p.add_argument("--query-ids", default="", help="待预测样本ID文本（每行一个，可选）")

    p.add_argument("--mode", choices=["prototype", "instance"], default="prototype", help="预测模式")
    p.add_argument("--batch-size", type=int, default=1024, help="推理批大小")
    p.add_argument("--temperature", type=float, default=None, help="softmax 温度；默认用地图文件中的值")
    p.add_argument("--topk", type=int, default=5, help="输出前k个候选")

    # 🌟 新增：阈值过滤控制参数
    p.add_argument("--threshold-tsv", default="family_thresholds.tsv", help="三级阈值基准表格路径 (.tsv)")
    p.add_argument("--filter-mode", choices=["strict", "moderate", "loose", "none"], default="none", 
                   help="过滤控制模式：strict(高置信度), moderate(标准平衡), loose(远源挖掘), none(关闭过滤)")

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

    # 为了防止 exec 解释器转义冲突，统一采用基础安全读取逻辑
    ni = col_idx["ADS_name"]
    fi = col_idx["Against/Function"]

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline() # 跳过表头
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


def load_thresholds(path: str) -> Dict[str, Dict[str, float]]:
    """⚙️ 新增：解析 family_thresholds.tsv 基准矩阵"""
    if not os.path.exists(path):
        print(f"⚠️ 未找到三级阈值矩阵文件: {path}，将关闭在线拦截。")
        return {}
    
    thresh_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fid = row.get("family_id")
            if fid:
                thresh_map[fid] = {
                    "strict": float(row.get("threshold_strict", 0.0)),
                    "moderate": float(row.get("threshold_moderate", 0.0)),
                    "loose": float(row.get("threshold_loose", 0.0))
                }
    return thresh_map


def format_thresh(val: float) -> str:
    """辅助格式化输出数值"""
    if val == float("inf") or val == float("-inf") or np.isnan(val):
        return "NA"
    return f"{val:.4f}"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 基础依赖文件校验
    check_paths = [args.model_path, args.map_path, args.query_emb, args.query_hmm]
    if args.filter_mode != "none":
        check_paths.append(args.threshold_tsv)
        
    for p in check_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到文件: {p}")

    payload = torch.load(args.map_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("地图文件格式错误")

    # 2. 载入知识库与新增的阈值库
    cluster_annotation_map, cluster_func_cols = load_cluster_annotation(args.cluster_annotation_txt)
    ads_function_map = load_ads_function(args.ads_function_txt)
    thresholds_map = load_thresholds(args.threshold_tsv) if args.filter_mode != "none" else {}

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

    if args.mode == "prototype":
        if "prototypes" not in payload:
            raise ValueError("地图文件中没有 prototypes，请重新用 --mode prototype/both 生成")
        gallery = payload["prototypes"].float().to(device)
        gallery_labels = [int(x) for x in payload["labels"]]
        gallery_ids = [f"prototype_of_{lb}" for lb in gallery_labels]
        target_k = min(topk, len(gallery_labels))
    else:
        if "instance_features" not in payload:
            raise ValueError("地图文件中没有 instance_features，请重新用 --mode instance/both 生成")
        gallery = payload["instance_features"].float().to(device)
        gallery_labels = [int(x) for x in payload["instance_labels"]]
        gallery_ids = payload["instance_ids"]
        target_k = min(topk, len(gallery_labels))

    has_instance_gallery = ("instance_features" in payload and "instance_labels" in payload and "instance_ids" in payload)
    if has_instance_gallery:
        instance_gallery = payload["instance_features"].float().to(device)
        instance_labels = [int(x) for x in payload["instance_labels"]]
        instance_ids = payload["instance_ids"]

    # 3. 批推理与超球面度量
    with torch.no_grad():
        for st in range(0, n, args.batch_size):
            ed = min(st + args.batch_size, n)
            x_seq = torch.from_numpy(q_emb[st:ed]).float().to(device)
            x_evo = torch.from_numpy(q_hmm[st:ed]).float().to(device)

            qz = model(x_seq, x_evo)
            d2 = torch.cdist(qz, gallery, p=2).pow(2)

            probs = F.softmax(-d2 / temp, dim=1)
            top_probs, top_idx = torch.topk(probs, k=target_k, dim=1)
            top_d2 = torch.gather(d2, 1, top_idx)

            if has_instance_gallery:
                d2_ins = torch.cdist(qz, instance_gallery, p=2).pow(2)
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

                pred_cluster_name = cluster_names.get(best_label, f"cluster_{best_label}")
                cluster_func_values = cluster_annotation_map.get(best_label, {})
                nearest_seq_function_summary = lookup_ads_function(nearest_seq_id, ads_function_map)

                # 🛑 核心过滤拦截逻辑 (Top-1 核心预测检测)
                filter_status = "Pass"
                thresh_limit_val = float("inf")
                if args.filter_mode != "none" and thresholds_map:
                    if pred_cluster_name in thresholds_map:
                        thresh_limit_val = thresholds_map[pred_cluster_name][args.filter_mode]
                        filter_status = "Pass" if best_d2 <= thresh_limit_val else "Fail"
                    else:
                        filter_status = "Fail"  # 预测到了未包含在训练集阈值内的特殊簇，默认拦截

                # 对 Top-k 候选池中的每一项同样进行各自家族的独立阈值校验
                cand = []
                for j in range(target_k):
                    gidx = int(top_idx[i, j].item())
                    lb = int(gallery_labels[gidx])
                    cid = str(gallery_ids[gidx])
                    cfunc = lookup_ads_function(cid, ads_function_map) if args.mode == "instance" else ""
                    c_cluster_name = cluster_names.get(lb, f"cluster_{lb}")
                    
                    c_filter_status = "Pass"
                    c_thresh_limit_val = float("inf")
                    if args.filter_mode != "none" and thresholds_map:
                        if c_cluster_name in thresholds_map:
                            c_thresh_limit_val = thresholds_map[c_cluster_name][args.filter_mode]
                            c_filter_status = "Pass" if float(top_d2[i, j].item()) <= c_thresh_limit_val else "Fail"
                        else:
                            c_filter_status = "Fail"

                    cand.append({
                        "rank": j + 1,
                        "id": cid,
                        "label": lb,
                        "cluster": c_cluster_name,
                        "cluster_rep": cluster_reps.get(lb, "NA"),
                        "distance2": float(top_d2[i, j].item()),
                        "confidence": float(top_probs[i, j].item()),
                        "ads_function": cfunc,
                        "filter_status": c_filter_status,
                        "threshold_limit": c_thresh_limit_val
                    })

                results.append({
                    "query_id": query_ids[st + i],
                    "mode": args.mode,
                    "pred_id": best_id,
                    "pred_label": best_label,
                    "pred_cluster": pred_cluster_name,
                    "pred_cluster_rep": cluster_reps.get(best_label, "NA"),
                    "pred_distance2": best_d2,
                    "confidence": best_prob,
                    # 🌟 阈值控制元数据注入
                    "filter_mode": args.filter_mode,
                    "filter_status": filter_status,
                    "threshold_limit": thresh_limit_val,
                    "nearest_sequence_id": nearest_seq_id,
                    "nearest_sequence_label": nearest_seq_label,
                    "nearest_sequence_distance2": nearest_seq_distance2,
                    "nearest_sequence_function_summary": nearest_seq_function_summary,
                    "cluster_func_values": cluster_func_values,
                    "topk": cand,
                })

    # 4. 多轨格式化数据保存
    cluster_func_out_cols = [f"cluster_func_{c}" for c in cluster_func_cols]

    # 轨 1) 主输出（无缝嵌入 filter_mode, filter_status, threshold_limit）
    with open(args.output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            "query_id", "mode", "pred_id", "pred_cluster", "pred_cluster_rep",
            "pred_distance2", "confidence", "filter_mode", "filter_status", "threshold_limit",
            "nearest_sequence_id", "nearest_sequence_label", "nearest_sequence_distance2", "nearest_sequence_function_summary",
            *cluster_func_out_cols,
        ])

        for r in results:
            cluster_func_vals = [r["cluster_func_values"].get(c, "") for c in cluster_func_cols]
            writer.writerow([
                r["query_id"], r["mode"], r["pred_id"], r["pred_cluster"], r["pred_cluster_rep"],
                f"{r['pred_distance2']:.6f}", f"{r['confidence']:.6f}",
                r["filter_mode"], r["filter_status"], format_thresh(r["threshold_limit"]),
                r["nearest_sequence_id"], r["nearest_sequence_label"],
                f"{r['nearest_sequence_distance2']:.6f}" if r["nearest_sequence_id"] else "",
                r["nearest_sequence_function_summary"],
                *cluster_func_vals,
            ])

    # 轨 2) Top-k 明细输出（追加专属候选阈值校验结果）
    with open(args.topk_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            "query_id", "mode", "rank", "candidate_id", "candidate_cluster", "candidate_rep",
            "candidate_distance2", "candidate_confidence", "filter_mode", "filter_status", "threshold_limit", "candidate_ads_function"
        ])
        for r in results:
            for c in r["topk"]:
                writer.writerow([
                    r["query_id"], r["mode"], c["rank"], c["id"], c["cluster"], c["cluster_rep"],
                    f"{c['distance2']:.6f}", f"{c['confidence']:.6f}",
                    r["filter_mode"], c["filter_status"], format_thresh(c["threshold_limit"]),
                    c["ads_function"],
                ])

    # 5. 终端回显美化
    print(f"\n🔮 ==== 预测结果（mode={args.mode} | filter_mode={args.filter_mode}） ====")
    for r in results:
        status_tag = f" [{r['filter_status']}]" if args.filter_mode != "none" else ""
        print(
            f"{r['query_id']}: id={r['pred_id']} | {r['pred_cluster']} (rep={r['pred_cluster_rep']}){status_tag} "
            f"| d²={r['pred_distance2']:.6f} | conf={r['confidence']:.2%}"
        )
        if r["nearest_sequence_id"]:
            print(f"   - nearest_sequence: {r['nearest_sequence_id']} (label={r['nearest_sequence_label']}, d²={r['nearest_sequence_distance2']:.6f})")
            if r["nearest_sequence_function_summary"]:
                print(f"     function: {r['nearest_sequence_function_summary']}")
        
        if args.print_topk:
            for c in r["topk"]:
                c_status_tag = f" [{c['filter_status']}]" if args.filter_mode != "none" else ""
                print(f"   - top{c['rank']}: id={c['id']} | {c['cluster']} (rep={c['cluster_rep']}){c_status_tag} d²={c['distance2']:.6f}, p={c['confidence']:.2%}")

    print(f"\n✅ 主预测表（带阈值联动标签）已保存: {args.output_tsv}")
    print(f"✅ Top-k 明细表（带候选独立卡口标签）已保存: {args.topk_tsv}")


if __name__ == "__main__":
    main()