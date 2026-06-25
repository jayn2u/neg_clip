# MSCOCO NegCLIP Dataset

This repository's CSV dataloader expects a tab-separated file with these columns:

- `filepath`: absolute path to an image.
- `title`: original caption.
- `neg_caption`: Python-list literal of mined negative captions.
- `neg_image`: Python-list literal of row indices in the same TSV.

The script below builds that file from COCO 2017 captions.

Before running mining, record required Python packages in `pyproject.toml`
instead of installing them ad hoc during the run. The text mining step needs
spaCy, and the image mining step uses the project's existing CLIP, PyTorch,
torchvision, PIL, NumPy, pandas, and tqdm stack. After updating
`pyproject.toml`, sync the environment with `uv sync`.

```bash
uv run python -m spacy download en_core_web_sm

uv run python src/data/build_mscoco_negclip_csv.py \
  --coco-root /data/jayn2u/datasets/ms-coco \
  --split train2017 \
  --output /data/jayn2u/datasets/ms-coco/negclip_train2017.tsv \
  --device cuda \
  --model ViT-B-32 \
  --pretrained openai \
  --batch-size 256 \
  --workers 8
```

The image mining step computes CLIP image embeddings and then finds the nearest
images with blockwise similarity search. The full pairwise matrix is never
materialized. The script writes two caches next to the TSV by default:

- `negclip_train2017.image_features.pt`
- `negclip_train2017.image_neighbors.json`

Reuse those caches when rerunning the text side or changing TSV output paths:

```bash
uv run python src/data/build_mscoco_negclip_csv.py \
  --coco-root /data/jayn2u/datasets/ms-coco \
  --split train2017 \
  --output /data/jayn2u/datasets/ms-coco/negclip_train2017.tsv \
  --image-neighbors /data/jayn2u/datasets/ms-coco/negclip_train2017.image_neighbors.json \
  --image-features /data/jayn2u/datasets/ms-coco/negclip_train2017.image_features.pt
```

For a quick smoke test, use a small caption limit:

```bash
uv run python src/data/build_mscoco_negclip_csv.py \
  --coco-root /data/jayn2u/datasets/ms-coco \
  --split train2017 \
  --output /tmp/negclip_debug.tsv \
  --limit 100 \
  --device cuda
```

Train with the generated TSV:

```bash
uv run python -m training.main \
  --train-data /data/jayn2u/datasets/ms-coco/negclip_train2017.tsv \
  --dataset-type csv \
  --csv-separator $'\t' \
  --csv-img-key filepath \
  --csv-caption-key title \
  --csv-hard-captions-key neg_caption
```
