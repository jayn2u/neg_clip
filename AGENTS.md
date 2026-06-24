# NegCLIP Agent Notes

Use `uv run python` to execute Python code.

## Project role

This repository is the original NegCLIP implementation (ICLR 2023): an OpenCLIP fork with hard negative captions and images in the training loop. It is **not** a person ReID project. There is no CUHK-PEDES, ICFG-PEDES, or RSTPReid training pipeline here.

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
