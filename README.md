# PHADS Inference Pipeline User Guide

This directory provides a ready-to-run PHADS-style inference pipeline. The main entry point is `main.py`.

The pipeline supports four prediction modes:

- `prototype`: deep model prototype prediction (default)
- `instance`: deep model nearest-instance prediction
- `hmm`: HMM annotation only
- `foldseek`: structural similarity search only

Deep prediction modes support three model routes:

- `--model-type ESM2`: ESM2 residue features only
- `--model-type ProtT5`: ProtT5 residue features only
- `--model-type mix`: fused ESM2 + ProtT5 residue features (default)

## 1. Before You Start

### 1.1 Environment

Create a Conda environment with `environment.yml` or `prost.yml`:

```powershell
conda env create -f environment.yml
conda activate PhADS
```

The pipeline also depends on external executables:

- `hmmscan` from HMMER: required by `prototype`, `instance`, and `hmm` modes
- `foldseek`: required by `foldseek` mode

Automatic translation of DNA input also requires these Python packages:

- `pyrodigal`
- `pyrodigal-gv`
- `biopython`

### 1.2 Download Embedding Models

Before running deep prediction modes for the first time, download the ESM2 and ProtT5 ONNX embedding models:

```powershell
python main.py database --path D:\models\phads_embeddings
```

After downloading, the directory passed to `--db` should contain:

```text
esm2_t33_650M_UR50D/
prot-t5-xl-uniref50-enc-onnx/
```

## 2. Common Commands

### 2.1 Self-Check

```powershell
python main.py -v --model-type ESM2 --db D:\models\phads_embeddings
python main.py -v --model-type ProtT5 --db D:\models\phads_embeddings
python main.py -v --model-type mix --db D:\models\phads_embeddings
```

### 2.2 Deep Prediction for One FASTA File

Recommended mixed-model command:

```powershell
python main.py -i input.faa --db D:\models\phads_embeddings -o result_mix --model-type mix --predict-mode prototype --device cuda --filter-mode moderate --topk 5 --use-multiprototype --judge-mode heuristic
```

ESM2 only:

```powershell
python main.py -i input.faa --db D:\models\phads_embeddings -o result_esm2 --model-type ESM2 --predict-mode prototype --device cuda
```

ProtT5 only:

```powershell
python main.py -i input.faa --db D:\models\phads_embeddings -o result_prott5 --model-type ProtT5 --predict-mode prototype --device cuda
```

### 2.3 Batch Prediction for a Directory

```powershell
python main.py -d fasta_dir --db D:\models\phads_embeddings -o result_batch --model-type mix --predict-mode prototype --device cuda
```

### 2.4 HMM-Only and Foldseek-Only Modes

```powershell
python main.py -i input.faa --predict-mode hmm -o result_hmm
python main.py -i structures_dir --predict-mode foldseek -o result_foldseek
```

## 3. Frequently Used Arguments

Common `main.py` arguments:

- `-i`, `--input-fasta`: single input FASTA file, either protein or DNA
- `-d`, `--dir`: input directory for batch processing
- `-o`, `--work-dir`: output directory
- `-temp`, `--temp`: temporary directory, defaulting to `<work-dir>/temp`
- `--db`: root directory of the embedding models
- `--predict-mode`: `prototype`, `instance`, `hmm`, or `foldseek`
- `--model-type`: `ESM2`, `ProtT5`, or `mix`
- `--device`: `auto`, `cpu`, `cuda`, or `cuda:0`
- `--topk`: number of TopK candidates to report
- `--filter-mode`: `strict`, `moderate`, `loose`, or `none`
- `--use-multiprototype`: enable sub-prototypes
- `--judge-mode`: `off` or `heuristic`
- `--ads-detection-threshold-mode`: `auto`, `global`, or `cluster-guarded`

## 4. Output Files

Deep modes (`prototype` and `instance`) produce:

```text
prediction_results.tsv
prediction_topk.tsv
qc_report.txt
qc_report.json
```

Directory batch processing also produces:

```text
all_results.tsv
topk/all_prediction_topk.tsv
```

HMM-only mode produces:

```text
hmm_result.tsv
```

Foldseek-only mode produces:

```text
foldseek_result.tsv
```

### 4.1 Key Columns in `prediction_results.tsv`

- `query_id`
- `pred_cluster`, `pred_label`, `pred_cluster_rep`
- `pred_distance2`, `confidence`
- `hmm_gate`, `esm_gate`
- `hmm_score`, `judge_score`
- `pred_ads_function`
- `ads_detection_score`, `ads_detection_threshold`, `ads_detection_status`
- `ads_detection_threshold_mode`, `ads_detection_threshold_rule`, `ads_detection_threshold_cluster`
- `cluster_func_*` columns from `cluster_annotation.tsv`

### 4.2 ADS Detection Thresholds

Each model directory may provide:

- `ads_detection_threshold.tsv`: global threshold
- `ads_detection_cluster_thresholds.tsv`: cluster-specific guard thresholds, commonly used by the `mix` model

The `auto` threshold mode follows this strategy:

- If `model-type=mix` and the cluster threshold file exists, use `cluster-guarded`
- Otherwise, use `global`

## 5. Pipeline Overview

The internal order for `prototype` and `instance` modes is:

1. Read the FASTA input. If the input is DNA, translate it to protein with `pyrodigal_viral.py`.
2. Generate residue embeddings with ESM2, ProtT5, or both.
3. Run `hmmscan` and convert the result into fixed-length HMM features with `hmm_to_npy.py`.
4. In `mix` mode, align the two residue feature routes with `pair_residue_embeddings.py`.
5. Run `predict_residue_pool_gate.py` to produce predictions, TopK candidates, and ADS detection results.
6. Generate QC reports.

For a more detailed script-level workflow and mathematical description, see `description.md`.

## 6. Troubleshooting

1. `hmmscan` or `foldseek` is not found

   Make sure the required executable is installed and available in `PATH`.

2. Deep prediction reports that embedding model directories are missing

   Run the `database` subcommand first and make sure `--db` points to the directory containing both model subdirectories.

3. Pairing fails in `mix` mode

   Check `residue_pair_manifest.tsv` and `missing_residue_pairs.log` in the output directory.

4. DNA input produces empty results

   Check whether `pyrodigal`, `pyrodigal-gv`, and `biopython` are installed.

## 7. Version

- `APP_VERSION`: `0.6-best-ablation-triple-route`
- Default model type: `mix`
