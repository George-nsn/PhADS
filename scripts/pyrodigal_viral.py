#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict viral/phage genes with pyrodigal-gv and export protein FASTA and GFF3 files."""

import argparse
import sys
from pathlib import Path


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
    try:
        from Bio import SeqIO
        import pyrodigal
        import pyrodigal_gv
    except ImportError as exc:
        print(f"ERROR: required gene-prediction dependency is missing: {exc}", file=sys.stderr)
        sys.exit(1)
    
    input_path = Path(args.input_fasta).resolve()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
        
    out_faa = Path(args.output_faa) if args.output_faa else input_path.with_suffix(".faa")
    out_gff = Path(args.output_gff) if args.output_gff else input_path.with_suffix(".gff")
    out_pos = Path(args.output_pos) if args.output_pos else input_path.with_suffix("").with_suffix("_pos.tsv")
    
    out_faa.parent.mkdir(parents=True, exist_ok=True)
    out_gff.parent.mkdir(parents=True, exist_ok=True)
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing Pyrodigal-GV (meta mode, viral_only={args.viral_only})...")
    orf_finder = None

    try:
        orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True, viral_only=args.viral_only)
        print("   Using ViralGeneFinder API")
    except AttributeError:
        pass

    if orf_finder is None:
        try:
            bins = pyrodigal_gv.METAGENOMIC_BINS_VIRAL if args.viral_only else pyrodigal_gv.METAGENOMIC_BINS
            orf_finder = pyrodigal.GeneFinder(meta=True, metagenomic_bins=bins)
            print("   Using GeneFinder with METAGENOMIC_BINS API")
        except AttributeError:
            pass

    if orf_finder is None:
        print("   Advanced pyrodigal-gv API is unavailable; falling back to pyrodigal.GeneFinder(meta=True)")
        orf_finder = pyrodigal.GeneFinder(meta=True)
    
    total_records = 0
    total_genes = 0
    
    print(f"Parsing input sequences: {input_path}")
    
    try:
        with open(out_faa, "w", encoding="utf-8") as faa_file, \
             open(out_gff, "w", encoding="utf-8") as gff_file, \
             open(out_pos, "w", encoding="utf-8") as pos_file:
            
            pos_file.write("gene_id\tgene_start\tgene_end\n")
            
            for idx, record in enumerate(SeqIO.parse(input_path, "fasta")):
                total_records += 1
                seq_bytes = bytes(record.seq)
                
                genes = orf_finder.find_genes(seq_bytes)
                gene_count = len(genes)
                total_genes += gene_count
                
                genes.write_translations(faa_file, sequence_id=record.id)
                
                genes.write_gff(gff_file, sequence_id=record.id, header=(idx == 0))
                
                for i, gene in enumerate(genes, start=1):
                    gene_id = f"{record.id}_{i}"
                    pos_file.write(f"{gene_id}\t{gene.begin}\t{gene.end}\n")
                
                if idx < 3 or (idx + 1) % 100 == 0:
                    print(f"  - sequence: {record.id} | predicted coding genes: {gene_count}")
                    
    except Exception as e:
        print(f"ERROR: gene prediction failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("Gene prediction and annotation export completed successfully.")
    print(f"- Processed contigs: {total_records}")
    print(f"- Predicted genes: {total_genes}")
    print(f"- Protein FASTA (.faa): {out_faa}")
    print(f"- Annotation file (.gff): {out_gff}")
    print(f"- Gene position table (.tsv): {out_pos}")
    print("=" * 60)


if __name__ == "__main__":
    main()