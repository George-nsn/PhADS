#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pyrodigal-GV 基因預測與翻譯管線腳本
功能：自動預測巨型病毒/噬菌體基因，同步導出蛋白質 FASTA (.faa) 與標準 GFF3 (.gff) 文件。
"""

import argparse
import sys
from pathlib import Path
from Bio import SeqIO
import pyrodigal
import pyrodigal_gv


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict genes in giant viruses/phages using pyrodigal-gv, exporting proteins (.faa) and GFF3 (.gff).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-i", "--input-fasta", required=True, help="Input genomic FASTA file (supports multi-fasta).")
    p.add_argument("-a", "--output-faa", default=None, help="Output protein FASTA path. Defaults to <input_stem>.faa")
    p.add_argument("-g", "--output-gff", default=None, help="Output GFF3 path. Defaults to <input_stem>.gff")
    p.add_argument("-p", "--output-pos", default=None, help="Output gene position TSV path. Defaults to <input_stem>_pos.tsv")
    p.add_argument("--viral-only", action="store_true", help="Restrict gene calling to alternative viral models only.")
    return p.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input_fasta).resolve()
    if not input_path.exists():
        print(f"❌ 錯誤: 找不到輸入文件: {input_path}", file=sys.stderr)
        sys.exit(1)
        
    # 自動推導輸出路徑
    out_faa = Path(args.output_faa) if args.output_faa else input_path.with_suffix(".faa")
    out_gff = Path(args.output_gff) if args.output_gff else input_path.with_suffix(".gff")
    out_pos = Path(args.output_pos) if args.output_pos else input_path.with_suffix("").with_suffix("_pos.tsv")
    
    # 創建輸出目錄
    out_faa.parent.mkdir(parents=True, exist_ok=True)
    out_gff.parent.mkdir(parents=True, exist_ok=True)
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 初始化 Pyrodigal-GV (Meta模式, viral_only={args.viral_only})...")
    # 多级兼容初始化（脚本名已避开 pyrodigal_gv 包名冲突）
    orf_finder = None

    # 方式 1: 原始 ViralGeneFinder API（推荐，名称冲突已解决）
    try:
        orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True, viral_only=args.viral_only)
        print("   → 使用 ViralGeneFinder API")
    except AttributeError:
        pass

    # 方式 2: pyrodigal.GeneFinder + METAGENOMIC_BINS
    if orf_finder is None:
        try:
            bins = pyrodigal_gv.METAGENOMIC_BINS_VIRAL if args.viral_only else pyrodigal_gv.METAGENOMIC_BINS
            orf_finder = pyrodigal.GeneFinder(meta=True, metagenomic_bins=bins)
            print("   → 使用 GeneFinder + METAGENOMIC_BINS API")
        except AttributeError:
            pass

    # 方式 3: 纯 pyrodigal.GeneFinder meta 模式（最大兼容性回退）
    if orf_finder is None:
        print("   ⚠️ pyrodigal_gv 高级 API 不可用，回退到 pyrodigal.GeneFinder(meta=True)")
        orf_finder = pyrodigal.GeneFinder(meta=True)
    
    total_records = 0
    total_genes = 0
    
    print(f"📖 開始解析輸入序列: {input_path}")
    
    try:
        with open(out_faa, "w", encoding="utf-8") as faa_file, \
             open(out_gff, "w", encoding="utf-8") as gff_file, \
             open(out_pos, "w", encoding="utf-8") as pos_file:
            
            # 寫入基因位置文件表頭
            pos_file.write("gene_id\tgene_start\tgene_end\n")
            
            for idx, record in enumerate(SeqIO.parse(input_path, "fasta")):
                total_records += 1
                seq_bytes = bytes(record.seq)
                
                # 執行基因預測 (Memory-resident prediction)
                genes = orf_finder.find_genes(seq_bytes)
                gene_count = len(genes)
                total_genes += gene_count
                
                # 1) 高性能直寫蛋白翻譯序列 (.faa)
                genes.write_translations(faa_file, sequence_id=record.id)
                
                # 2) 高性能直寫標準 GFF3 注釋 (.gff)
                # 僅在第一條 contig 寫入全局 GFF3 文件頭，避免多序列時頭部信息重複冗餘
                genes.write_gff(gff_file, sequence_id=record.id, header=(idx == 0))
                
                # 3) 寫入基因位置映射 (.tsv)
                # gene_id 格式: {contig_id}_{gene_number}，與 FASTA 頭文件第一個 token 一致
                for i, gene in enumerate(genes, start=1):
                    gene_id = f"{record.id}_{i}"
                    pos_file.write(f"{gene_id}\t{gene.begin}\t{gene.end}\n")
                
                if idx < 3 or (idx + 1) % 100 == 0:
                    print(f"  - 序列: {record.id} | 預測到 {gene_count} 個編碼基因")
                    
    except Exception as e:
        print(f"❌ 運行中發生異常: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("🎉 預測與註釋導出順利完成！")
    print(f"- 處理 Contigs 總數 : {total_records}")
    print(f"- 預測基因總數量    : {total_genes}")
    print(f"- 蛋白質序列 (.faa) : {out_faa}")
    print(f"- 註釋文件 (.gff)   : {out_gff}")
    print(f"- 基因位置 (.tsv)   : {out_pos}")
    print("=" * 60)


if __name__ == "__main__":
    main()