# NegCLIP Agent Notes

Use `uv run python` to execute Python code.
Use `uv` for dependency management.

## Project role

This repository is the original NegCLIP implementation (ICLR 2023): an OpenCLIP fork with hard negative captions and images in the training loop. It is **not** a person ReID project. There is no CUHK-PEDES, ICFG-PEDES, or RSTPReid training pipeline here.

## Dataset location

Lab datasets are stored at one of:

- `/mnt/data/lab_datasets`
- `/data/jayn2u/lab_datasets`

These paths refer to the same storage. Use whichever exists on the current machine.

## Pretrained NegCLIP weights (Google Drive)

This repo does **not** ship pretrained NegCLIP checkpoints. For evaluation or inference, download the official ICLR 2023 NegCLIP weights from the authors' Google Drive (ViT-B/32, single OpenCLIP `.pt` file, ~1.7GB).

| Item | Value |
|------|-------|
| File | `negCLIP.pt` |
| Google Drive ID | `1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ` |
| Local path | `checkpoints/negCLIP.pt` |

### Download example

```bash
mkdir -p checkpoints
gdown 1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ -O checkpoints/negCLIP.pt
```

Or with `uv`:

```bash
mkdir -p checkpoints
uv run gdown 1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ -O checkpoints/negCLIP.pt
```

Load with OpenCLIP:

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="checkpoints/negCLIP.pt",
)
```

The checkpoint source is the authors' evaluation code ([`mertyg/vision-language-models-are-bows`](https://github.com/mertyg/vision-language-models-are-bows), `model_zoo`).

## Mined NegCLIP data (MSCOCO)

MSCOCO 2017 hard-negative TSV files produced by `src/data/build_mscoco_negclip_csv.py` live under `{DATASET_ROOT}/ms-coco/`:

| Split | TSV | Image-feature cache | Neighbor cache |
|-------|-----|---------------------|----------------|
| train2017 | `negclip_train2017.tsv` | `negclip_train2017.image_features.pt` | `negclip_train2017.image_neighbors.json` |
| val2017 | `negclip_val2017.tsv` | `negclip_val2017.image_features.pt` | `negclip_val2017.image_neighbors.json` |

On this machine:

```
/data/jayn2u/lab_datasets/ms-coco/
├── train2017/
├── val2017/
├── annotations/
├── negclip_train2017.tsv
├── negclip_train2017.image_features.pt
├── negclip_train2017.image_neighbors.json
├── negclip_val2017.tsv
├── negclip_val2017.image_features.pt
└── negclip_val2017.image_neighbors.json
```

TSV columns: `filepath`, `title`, `neg_caption`, `neg_image` (tab-separated). `filepath` points at images under `{DATASET_ROOT}/ms-coco/{split}/`.

## Mining

Entry point: `src/data/build_mscoco_negclip_csv.py` (run from repo root).

Prerequisites:

```bash
cd /data/jayn2u/neg_clip
uv sync
uv run python -m spacy download en_core_web_sm
```

Resolve `{DATASET_ROOT}` to `/mnt/data/lab_datasets` or `/data/jayn2u/lab_datasets` (whichever exists). Raw MSCOCO 2017 must be present at `{DATASET_ROOT}/ms-coco/` (`train2017/`, `val2017/`, `annotations/`).

Text mining uses spaCy phrase swap for `neg_caption`. Image mining uses OpenAI CLIP ViT-B/32 embeddings to pick visually similar neighbor images for `neg_image`. Caches are written next to the output TSV.

Train split (full):

```bash
cd /data/jayn2u/neg_clip

uv run python src/data/build_mscoco_negclip_csv.py \
  --coco-root /data/jayn2u/lab_datasets/ms-coco \
  --split train2017 \
  --output /data/jayn2u/lab_datasets/ms-coco/negclip_train2017.tsv \
  --device cuda \
  --model ViT-B-32 \
  --pretrained openai \
  --batch-size 256 \
  --workers 8
```

Validation split:

```bash
uv run python src/data/build_mscoco_negclip_csv.py \
  --coco-root /data/jayn2u/lab_datasets/ms-coco \
  --split val2017 \
  --output /data/jayn2u/lab_datasets/ms-coco/negclip_val2017.tsv \
  --device cuda \
  --model ViT-B-32 \
  --pretrained openai \
  --batch-size 256 \
  --workers 8
```

Smoke test (first 100 captions):

```bash
uv run python src/data/build_mscoco_negclip_csv.py \
  --coco-root /data/jayn2u/lab_datasets/ms-coco \
  --split train2017 \
  --output /tmp/negclip_debug.tsv \
  --limit 100 \
  --device cuda
```

Reuse image caches when regenerating TSV only:

```bash
uv run python src/data/build_mscoco_negclip_csv.py \
  --coco-root /data/jayn2u/lab_datasets/ms-coco \
  --split train2017 \
  --output /data/jayn2u/lab_datasets/ms-coco/negclip_train2017.tsv \
  --image-neighbors /data/jayn2u/lab_datasets/ms-coco/negclip_train2017.image_neighbors.json \
  --image-features /data/jayn2u/lab_datasets/ms-coco/negclip_train2017.image_features.pt \
  --device cuda
```

## Training

Entry point: `src/training/main.py` (run from `src/` with `PYTHONPATH` including `src`).

```bash
cd /data/jayn2u/neg_clip/src
export PYTHONPATH=/data/jayn2u/neg_clip/src:$PYTHONPATH

uv run python -m training.main \
  --train-data /data/jayn2u/lab_datasets/ms-coco/negclip_train2017.tsv \
  --val-data /data/jayn2u/lab_datasets/ms-coco/negclip_val2017.tsv \
  --dataset-type csv \
  --csv-separator $'\t' \
  --model ViT-B-32 \
  --pretrained openai \
  --batch-size 256 \
  --epochs 5 \
  --lr 1e-6 \
  --warmup 50
```

Single-GPU only; multi-GPU distributed training is not supported for NegCLIP in this repo.

## Config injection and dataset naming

This rule applies across all lab ReID repos; see each project's AGENTS.md for project-specific env vars and config prefixes.

Evaluation scripts do **not** pick a dataset or config file by default. Every run must inject the YAML path through an environment variable. If the variable is missing or empty, the script raises an error.

| Script | Environment variable | Config prefix |
|--------|---------------------|---------------|
| `sugarcrepe-pedes.py` | `SUGARCREPE_CONFIG` | `sugarcrepe` |
| `text-to-image-retrieval.py` | `RETRIEVAL_CONFIG` | `text_to_image_retrieval` |

Supported per-dataset config files (dataset slug as filename suffix):

- `configs/sugarcrepe_cuhk_pedes.yaml` / `configs/text_to_image_retrieval_cuhk_pedes.yaml` (`dataset: cuhk-pedes`)
- `configs/sugarcrepe_icfg_pedes.yaml` / `configs/text_to_image_retrieval_icfg_pedes.yaml` (`dataset: icfg-pedes`)
- `configs/sugarcrepe_rstpreid.yaml` / `configs/text_to_image_retrieval_rstpreid.yaml` (`dataset: rstpreid`)

Each YAML must set `dataset` explicitly. Do not rely on code defaults for dataset selection. Shell scripts under `shell/` export the matching pair of config paths before calling the Python entry points.

Example:

```bash
export SUGARCREPE_CONFIG=configs/sugarcrepe_icfg_pedes.yaml
export RETRIEVAL_CONFIG=configs/text_to_image_retrieval_icfg_pedes.yaml
uv run python text-to-image-retrieval.py
uv run python sugarcrepe-pedes.py
```

Or use `shell/eval_icfg_pedes.sh`, `shell/eval_cuhk_pedes.sh`, or `shell/eval_rstpreid.sh`.

CLI arguments are not supported for evaluation runs.

## Lab compositional evaluation

For SugarCrepe-style pedestrian compositional probes (`sugarcrepe-pedes`), set `checkpoint_dir: checkpoints/negCLIP.pt` in the per-dataset YAML configs under `configs/`. Do not use TripletCLIP Hugging Face checkpoints for this repo's evaluation pipeline.
