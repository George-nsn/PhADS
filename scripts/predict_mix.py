import os
import argparse
import csv
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from predict import (
    load_query_ids,
    ensure_2d,
    infer_latent_dim,
    load_cluster_annotation,
    load_ads_function,
    load_thresholds,
    lookup_ads_function,
    format_thresh,
)


class EmbeddingCrossAttnFusion(nn.Module):
    """ESM2 和 ProstT5 嵌入通过交叉注意力互相增强。"""
    def __init__(self, dim_esm2=1280, dim_prost=1024, dim_out=512, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj_esm2 = nn.Linear(dim_esm2, dim_out)
        self.proj_prost = nn.Linear(dim_prost, dim_out)
        self.cross_attn_s2p = nn.MultiheadAttention(embed_dim=dim_out, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_p2s = nn.MultiheadAttention(embed_dim=dim_out, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm_s2p = nn.LayerNorm(dim_out)
        self.norm_p2s = nn.LayerNorm(dim_out)
        self.norm_fused = nn.LayerNorm(dim_out)
        self.align_esm2 = nn.Sequential(nn.Linear(dim_esm2, 256), nn.ReLU(), nn.Linear(256, 128))
        self.align_prost = nn.Sequential(nn.Linear(dim_prost, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, e_esm2, e_prost):
        q_esm2 = self.proj_esm2(e_esm2).unsqueeze(1)
        q_prost = self.proj_prost(e_prost).unsqueeze(1)
        e_esm2_enhanced, _ = self.cross_attn_s2p(q_esm2, q_prost, q_prost)
        e_esm2_enhanced = self.norm_s2p(q_esm2 + e_esm2_enhanced)
        e_prost_enhanced, _ = self.cross_attn_p2s(q_prost, q_esm2, q_esm2)
        e_prost_enhanced = self.norm_p2s(q_prost + e_prost_enhanced)
        e_fused = self.norm_fused(e_esm2_enhanced.squeeze(1) + e_prost_enhanced.squeeze(1))
        align_esm2 = self.align_esm2(e_esm2)
        align_prost = self.align_prost(e_prost)
        return e_fused, align_esm2, align_prost


class CrossAttnSDHProtoNet(nn.Module):
    """交叉注意力 SDH-ProtoNet 推理模型。"""
    def __init__(self, dim_esm2=1280, dim_prost=1024, evo_dim=211, latent_dim=512, cross_n_heads=4, cross_dropout=0.1):
        super().__init__()
        self.fusion = EmbeddingCrossAttnFusion(dim_esm2=dim_esm2, dim_prost=dim_prost, dim_out=latent_dim,
                                                n_heads=cross_n_heads, dropout=cross_dropout)
        self.evo_net = nn.Sequential(nn.Linear(evo_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(latent_dim * 2, 1), nn.Sigmoid())
        self.projector = nn.Linear(latent_dim, latent_dim)

    def forward(self, e_esm2, e_prost, x_evo, return_gate=False):
        e_fused, _, _ = self.fusion(e_esm2, e_prost)
        f_e = self.evo_net(x_evo)
        g = self.gate(torch.cat([e_fused, f_e], dim=-1))
        fused = g * e_fused + (1 - g) * f_e
        z = F.normalize(self.projector(fused), p=2, dim=1)
        if return_gate:
            return z, g.squeeze(-1)
        return z


def parse_args():
    p = argparse.ArgumentParser(description="Cross-Attention SDH-ProtoNet 双模型预测脚本（prototype/instance 双模式）")

    p.add_argument("--model-path", default="sdh_protonet_crossattn_best.pth", help="交叉注意力模型权重路径")
    p.add_argument("--map-path", default="family_map.pth", help="地图文件路径")

    p.add_argument("--query-emb-esm2", required=True, help="待预测 ESM2 嵌入 .npy [N, 1280] 或 [1280]")
    p.add_argument("--query-emb-prost5", required=True, help="待预测 ProstT5 嵌入 .npy [N, 1024] 或 [1024]")
    p.add_argument("--query-hmm", required=True, help="待预测 HMM 特征 .npy [N, evo_dim] 或 [evo_dim]")
    p.add_argument("--query-ids", default="", help="待预测样本ID文本（每行一个，可选）")

    p.add_argument("--mode", choices=["prototype", "instance"], default="prototype", help="预测模式")
    p.add_argument("--batch-size", type=int, default=1024, help="推理批大小")
    p.add_argument("--temperature", type=float, default=None, help="softmax 温度；默认用地图文件中的值")
    p.add_argument("--topk", type=int, default=5, help="输出前k个候选")
    p.add_argument("--cross-n-heads", type=int, default=None, help="交叉注意力头数；默认使用 4")
    p.add_argument("--cross-dropout", type=float, default=0.1, help="交叉注意力 dropout；推理时仅用于构造模型")

    p.add_argument("--threshold-tsv", default="family_thresholds.tsv", help="三级阈值基准表格路径 (.tsv)")
    p.add_argument("--filter-mode", choices=["strict", "moderate", "loose", "none"], default="moderate",
                   help="过滤控制模式：strict(高置信度), moderate(标准平衡), loose(远源挖掘), none(关闭过滤)")

    p.add_argument("--cluster-annotation-txt", default="cluster_annotation.txt", help="簇注释TXT（TSV）")
    p.add_argument("--ads-function-txt", default="ADS_function.txt", help="ADS功能TXT（TSV）")

    p.add_argument("--output-tsv", default="prediction_results.tsv", help="主预测输出表")
    p.add_argument("--topk-tsv", default="prediction_topk.tsv", help="Top-k 明细输出表")
    p.add_argument("--print-topk", action="store_true", help="终端打印 top-k")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_paths = [args.model_path, args.map_path, args.query_emb_esm2, args.query_emb_prost5, args.query_hmm]
    for p in check_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到文件: {p}")

    payload = torch.load(args.map_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("地图文件格式错误")

    cluster_annotation_map, cluster_func_cols, annot_reps = load_cluster_annotation(args.cluster_annotation_txt)
    ads_function_map = load_ads_function(args.ads_function_txt)
    if args.filter_mode != "none" and os.path.exists(args.threshold_tsv):
        thresholds_map = load_thresholds(args.threshold_tsv)
    elif args.filter_mode != "none":
        print(f"⚠️ 未找到阈值文件: {args.threshold_tsv}，双模型预测将不执行距离过滤。")
        thresholds_map = {}
    else:
        thresholds_map = {}

    q_esm2 = ensure_2d(np.load(args.query_emb_esm2), "query_emb_esm2")
    q_prost = ensure_2d(np.load(args.query_emb_prost5), "query_emb_prost5")
    q_hmm = ensure_2d(np.load(args.query_hmm), "query_hmm")

    if q_esm2.shape[0] != q_prost.shape[0] or q_esm2.shape[0] != q_hmm.shape[0]:
        raise ValueError(f"query 样本数不一致: ESM2={q_esm2.shape[0]}, ProstT5={q_prost.shape[0]}, HMM={q_hmm.shape[0]}")

    n = q_esm2.shape[0]
    query_ids = load_query_ids(args.query_ids, n)

    dim_esm2_expected = int(payload.get("dim_esm2", q_esm2.shape[1]))
    dim_prost_expected = int(payload.get("dim_prost", payload.get("dim_prost5", q_prost.shape[1])))
    evo_dim_expected = int(payload.get("evo_dim", q_hmm.shape[1]))
    if q_esm2.shape[1] != dim_esm2_expected:
        raise ValueError(f"query_emb_esm2 维度不匹配: got {q_esm2.shape[1]}, expect {dim_esm2_expected}")
    if q_prost.shape[1] != dim_prost_expected:
        raise ValueError(f"query_emb_prost5 维度不匹配: got {q_prost.shape[1]}, expect {dim_prost_expected}")
    if q_hmm.shape[1] != evo_dim_expected:
        raise ValueError(f"query_hmm 维度不匹配: got {q_hmm.shape[1]}, expect {evo_dim_expected}")

    temp = float(args.temperature) if args.temperature is not None else float(payload.get("temperature", 1.0))
    if temp <= 0:
        raise ValueError("temperature 必须 > 0")

    state = torch.load(args.model_path, map_location="cpu")
    latent_dim = infer_latent_dim(state, fallback=int(payload.get("latent_dim", 512)))
    cross_n_heads = int(args.cross_n_heads or payload.get("cross_n_heads", 4))
    model = CrossAttnSDHProtoNet(
        dim_esm2=dim_esm2_expected,
        dim_prost=dim_prost_expected,
        evo_dim=evo_dim_expected,
        latent_dim=latent_dim,
        cross_n_heads=cross_n_heads,
        cross_dropout=float(args.cross_dropout),
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    cluster_names: Dict[int, str] = {int(k): v for k, v in payload.get("cluster_names", {}).items()}

    topk = max(1, int(args.topk))
    results = []

    if args.mode == "prototype":
        if "prototypes" not in payload:
            raise ValueError("地图文件中没有 prototypes，请重新用 prototype/both 生成")
        gallery = payload["prototypes"].float().to(device)
        gallery_labels = [int(x) for x in payload["labels"]]
        gallery_ids = [f"prototype_of_{lb}" for lb in gallery_labels]
        target_k = min(topk, len(gallery_labels))
    else:
        if "instance_features" not in payload:
            raise ValueError("地图文件中没有 instance_features，请重新用 instance/both 生成")
        gallery = payload["instance_features"].float().to(device)
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
            e_esm2 = torch.from_numpy(q_esm2[st:ed]).float().to(device)
            e_prost = torch.from_numpy(q_prost[st:ed]).float().to(device)
            x_evo = torch.from_numpy(q_hmm[st:ed]).float().to(device)

            qz = model(e_esm2, e_prost, x_evo)
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
                pred_cluster_rep = annot_reps.get(best_label, "NA")

                filter_status = "Pass"
                thresh_limit_val = float("inf")
                if args.filter_mode != "none" and thresholds_map:
                    if pred_cluster_name in thresholds_map:
                        thresh_limit_val = thresholds_map[pred_cluster_name][args.filter_mode]
                        filter_status = "Pass" if best_d2 <= thresh_limit_val else "Fail"
                    else:
                        filter_status = "Fail"

                unfiltered_cluster = pred_cluster_name

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
                        "id": cid if c_filter_status == "Pass" else "NA",
                        "label": lb if c_filter_status == "Pass" else -1,
                        "cluster": c_cluster_name if c_filter_status == "Pass" else "Unknown (Non-ADS)",
                        "unfiltered_cluster": c_cluster_name,
                        "cluster_rep": annot_reps.get(lb, "NA") if c_filter_status == "Pass" else "NA",
                        "distance2": float(top_d2[i, j].item()),
                        "confidence": float(top_probs[i, j].item()),
                        "ads_function": cfunc if c_filter_status == "Pass" else "",
                        "filter_status": c_filter_status,
                        "threshold_limit": c_thresh_limit_val,
                    })

                results.append({
                    "query_id": query_ids[st + i],
                    "mode": args.mode,
                    "pred_id": best_id,
                    "pred_label": best_label,
                    "pred_cluster": pred_cluster_name,
                    "unfiltered_cluster": unfiltered_cluster,
                    "pred_cluster_rep": pred_cluster_rep,
                    "pred_distance2": best_d2,
                    "confidence": best_prob,
                    "filter_mode": args.filter_mode,
                    "filter_status": filter_status,
                    "threshold_limit": thresh_limit_val,
                    "nearest_sequence_id": nearest_seq_id,
                    "nearest_sequence_label": nearest_seq_label,
                    "nearest_sequence_distance2": nearest_seq_distance2,
                    "nearest_sequence_function_summary": nearest_seq_function_summary,
                    "closest_member": nearest_seq_id,
                    "cluster_func_values": cluster_func_values,
                    "topk": cand,
                    "_cand_pool": cand,
                })

    for r in results:
        if r["filter_status"] == "Fail" and args.filter_mode != "none":
            best_pass = None
            for c in r["_cand_pool"]:
                if c["filter_status"] == "Pass":
                    if best_pass is None or c["distance2"] < best_pass["distance2"]:
                        best_pass = c
            if best_pass is not None:
                r["pred_id"] = best_pass["id"]
                r["pred_label"] = best_pass["label"]
                r["pred_cluster"] = best_pass["cluster"]
                r["pred_cluster_rep"] = best_pass["cluster_rep"]
                r["pred_distance2"] = best_pass["distance2"]
                r["confidence"] = best_pass["confidence"]
                r["filter_status"] = "Pass"
                r["threshold_limit"] = best_pass["threshold_limit"]
                r["cluster_func_values"] = cluster_annotation_map.get(best_pass["label"], {})
                r["nearest_sequence_id"] = best_pass["cluster_rep"]
                r["nearest_sequence_label"] = best_pass["label"]
                r["nearest_sequence_distance2"] = best_pass["distance2"]
                r["nearest_sequence_function_summary"] = lookup_ads_function(best_pass["cluster_rep"], ads_function_map)

    cluster_func_out_cols = [f"cluster_func_{c}" for c in cluster_func_cols]
    total_count = len(results)
    passed_count = 0
    intercepted_count = 0

    with open(args.output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            "query_id", "mode", "closest_member", "pred_cluster", "pred_cluster_rep",
            "pred_distance2", "confidence", "filter_mode", "filter_status", "threshold_limit",
            "nearest_sequence_id", "nearest_sequence_label", "nearest_sequence_distance2", "nearest_sequence_function_summary",
            *cluster_func_out_cols,
        ])

        for r in results:
            if r["filter_status"] == "Fail" and args.filter_mode != "none":
                intercepted_count += 1
                continue

            passed_count += 1
            cluster_func_vals = [r["cluster_func_values"].get(c, "") for c in cluster_func_cols]
            nearest_d2_str = ""
            if r["nearest_sequence_id"] and r["nearest_sequence_id"] != "NA":
                nearest_d2_str = f"{r['nearest_sequence_distance2']:.6f}"
            writer.writerow([
                r["query_id"], r["mode"], r["closest_member"], r["pred_cluster"], r["pred_cluster_rep"],
                f"{r['pred_distance2']:.6f}", f"{r['confidence']:.6f}",
                r["filter_mode"], r["filter_status"], format_thresh(r["threshold_limit"]),
                r["nearest_sequence_id"], r["nearest_sequence_label"],
                nearest_d2_str,
                r["nearest_sequence_function_summary"],
                *cluster_func_vals,
            ])

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

    print(f"\n🔮 ==== 双模型预测结果（mode={args.mode} | filter_mode={args.filter_mode}） ====")

    for r in results:
        if r["filter_status"] == "Fail" and args.filter_mode != "none":
            continue
        status_tag = f" [Pass]" if args.filter_mode != "none" else ""
        print(
            f"{r['query_id']}: {r['pred_cluster']} (rep={r['pred_cluster_rep']}){status_tag} "
            f"| d²={r['pred_distance2']:.6f} | conf={r['confidence']:.2%}"
        )
        if r["nearest_sequence_id"] and r["nearest_sequence_id"] != "NA":
            print(f"   - nearest_sequence: {r['nearest_sequence_id']} (label={r['nearest_sequence_label']}, d²={r['nearest_sequence_distance2']:.6f})")
            if r["nearest_sequence_function_summary"]:
                print(f"     function: {r['nearest_sequence_function_summary']}")

        if args.print_topk:
            for c in r["topk"]:
                c_status_tag = f" [{c['filter_status']}]" if args.filter_mode != "none" else ""
                print(f"   - top{c['rank']}: id={c['id']} | {c['cluster']}{c_status_tag} d²={c['distance2']:.6f}, p={c['confidence']:.2%}")

    if args.filter_mode != "none":
        for r in results:
            if r["filter_status"] == "Fail":
                limit_str = format_thresh(r["threshold_limit"])
                print(
                    f"🛑 [FILTERED] {r['query_id']}: matched {r['unfiltered_cluster']} "
                    f"| d²={r['pred_distance2']:.6f} > threshold={limit_str} | conf={r['confidence']:.2%}"
                )

    print("\n" + "=" * 55)
    print("📊 防火墙拦截系统统计报告：")
    print(f"   🔹 总预测序列数          : {total_count} 条")
    print(f"   ✅ 高置信通过并输出      : {passed_count} 条")
    print(f"   🛑 距离超标拦截丢弃      : {intercepted_count} 条")
    if total_count > 0:
        print(f"   📈 拦截率                : {intercepted_count / total_count * 100:.1f}%")
    print("=" * 55)

    print(f"\n✅ 主预测表（仅通过防火墙的行）: {args.output_tsv}")
    print(f"✅ Top-k 明细表: {args.topk_tsv}")


if __name__ == "__main__":
    main()
