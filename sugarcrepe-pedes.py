from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import open_clip
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

PerturbationTag = Literal["swap_att", "replace_att", "other"]

DATASET_SPECS = {
    "cuhk-pedes": ("CUHK-PEDES", "reid_raw.json", "file_path"),
    "rstpreid": ("RSTPReid", "data_captions.json", "img_path"),
    "icfg-pedes": ("ICFG-PEDES", "ICFG-PEDES.json", "file_path"),
}

_COLOR_WORDS = frozenset(
    {
        "black", "white", "red", "blue", "green", "yellow", "gray", "grey",
        "brown", "pink", "purple", "orange", "beige", "navy", "tan", "gold",
        "silver", "bright", "dark", "light", "neon",
    }
)
_GARMENT_WORDS = frozenset(
    {
        "shirt", "jacket", "coat", "pants", "jeans", "trousers", "shorts",
        "skirt", "dress", "shoes", "sneakers", "boots", "hat", "cap", "hoodie",
        "sweatshirt", "vest", "backpack", "bag", "top", "t-shirt", "tee",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)

SUGARCREPE_DEFAULTS = {
    "env_file": "env/.env",
    "dataset": "",
    "dataset_root": "",
    "negative_pedestrians_root": "/mnt/data/negative-pedestrians/outputs",
    "negative_model": "gemma4:e4b",
    "negative_prompt": "tripletclip_reid.yaml",
    "negative_annotation": "auto",
    "checkpoint_dir": "",
    "model_label": "",
    "test_split": "test",
    "img_height": 384,
    "img_width": 128,
    "batch_size": 128,
    "num_workers": 4,
    "device": "auto",
    "seed": 42,
    "max_probes": 0,
    "output_json": "",
    "no_amp": False,
}

RETRIEVAL_DEFAULTS = {
    "env_file": "env/.env",
    "dataset": "",
    "dataset_root": "",
    "checkpoint_dir": "",
    "model_label": "",
    "test_split": "test",
    "img_height": 384,
    "img_width": 128,
    "batch_size": 128,
    "num_workers": 4,
    "device": "auto",
    "seed": 42,
    "top_k": [1, 5, 10],
    "max_image_samples": 0,
    "max_text_samples": 0,
    "output_json": "",
    "no_amp": False,
}


@dataclass(frozen=True)
class CompositionalProbe:
    image_path: Path
    person_id: int
    positive_caption: str
    negative_caption: str
    caption_index: int
    perturbation: PerturbationTag


@dataclass(frozen=True)
class ReIDImageItem:
    image_path: Path
    person_id: int


@dataclass(frozen=True)
class ReIDTextItem:
    caption: str
    person_id: int


class HubReIDCLIP(torch.nn.Module):
    def __init__(
        self,
        checkpoint_dir: Path,
        device: torch.device,
        img_height: int,
        img_width: int,
    ) -> None:
        super().__init__()
        self.vision = AutoModel.from_pretrained(
            str(checkpoint_dir / "vision-encoder"),
            trust_remote_code=True,
        )
        self.text = AutoModel.from_pretrained(
            str(checkpoint_dir / "text-encoder"),
            trust_remote_code=True,
        )
        self.vision.to(device).eval()
        self.text.to(device).eval()
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            force_image_size=(img_height, img_width),
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision(
            {"pixel_values": images, "interpolate_pos_encoding": True}
        )

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        attention_mask = (text_tokens != 0).long()
        outputs = self.text(input_ids=text_tokens, attention_mask=attention_mask)
        return outputs.text_embeds


def normalize_dataset_name(dataset: str) -> str:
    normalized = dataset.strip().lower()
    if normalized not in DATASET_SPECS:
        supported = ", ".join(DATASET_SPECS)
        raise ValueError(f"Unsupported dataset {dataset!r}. Expected one of: {supported}.")
    return normalized


def _read_env_key(env_file: str | Path, key: str) -> str | None:
    value = os.environ.get(key)
    if value is not None:
        value = value.strip().strip('"').strip("'")
        return value or None
    env_path = Path(env_file)
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        env_key, env_value = line.split("=", 1)
        if env_key.strip() != key:
            continue
        env_value = env_value.strip().strip('"').strip("'")
        return env_value or None
    return None


def resolve_dataset_root(dataset_root: str, env_file: str) -> Path:
    if dataset_root.strip():
        path = Path(dataset_root).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"dataset_root does not exist: {path}")
    if env_file.strip():
        value = _read_env_key(env_file, "DATASET_ROOT")
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path
    for candidate in (Path("/mnt/data/lab_datasets"), Path("/data/jayn2u/lab_datasets")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve dataset root.")


def resolve_negative_pedestrians_root(config: SimpleNamespace) -> Path:
    if config.env_file.strip():
        value = _read_env_key(config.env_file, "NEGATIVE_REID_DATASET_PATH")
        if value:
            return Path(value).expanduser()
    return Path(config.negative_pedestrians_root).expanduser()


def resolve_negative_annotation_path(config: SimpleNamespace, dataset: str) -> Path:
    normalized = normalize_dataset_name(dataset)
    dataset_dir = resolve_negative_pedestrians_root(config) / normalized
    requested = str(config.negative_annotation).strip()
    if requested and requested.lower() != "auto":
        direct = Path(requested).expanduser()
        candidates = [direct]
        if not direct.is_absolute():
            candidates.append(dataset_dir / requested)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Could not find negative annotation {requested!r}.")
    if config.env_file.strip():
        model_tag = _read_env_key(config.env_file, "NEGATIVE_MODEL") or config.negative_model
        prompt = _read_env_key(config.env_file, "NEGATIVE_PROMPT") or config.negative_prompt
    else:
        model_tag = config.negative_model
        prompt = config.negative_prompt
    prompt_stem = Path(prompt).stem
    _, annotation_file, _ = DATASET_SPECS[normalized]
    negative_file = f"{Path(annotation_file).stem}_negative_{model_tag}_{prompt_stem}.json"
    candidate = dataset_dir / negative_file
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find negative captions file: {candidate}")


def tokenize_caption(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def tag_perturbation(positive_caption: str, negative_caption: str) -> PerturbationTag:
    pos_tokens = tokenize_caption(positive_caption)
    neg_tokens = tokenize_caption(negative_caption)
    if not pos_tokens or not neg_tokens:
        return "other"
    union = pos_tokens | neg_tokens
    overlap = len(pos_tokens & neg_tokens) / len(union)
    pos_colors = pos_tokens & _COLOR_WORDS
    neg_colors = neg_tokens & _COLOR_WORDS
    pos_garments = pos_tokens & _GARMENT_WORDS
    neg_garments = neg_tokens & _GARMENT_WORDS
    if overlap >= 0.65 and pos_colors and neg_colors and pos_garments and neg_garments:
        return "swap_att"
    if overlap >= 0.45:
        return "replace_att"
    return "other"


def load_compositional_probes(
    *,
    annotation_path: Path,
    dataset_root: Path,
    dataset: str,
    split: str,
    max_probes: int = 0,
) -> list[CompositionalProbe]:
    directory, _, image_key = DATASET_SPECS[normalize_dataset_name(dataset)]
    image_root = dataset_root / directory / "imgs"
    raw_records = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise TypeError(f"Expected a list in {annotation_path}.")
    probes: list[CompositionalProbe] = []
    for raw in raw_records:
        if not isinstance(raw, dict) or str(raw.get("split", "")) != split:
            continue
        pos_caps = [str(caption) for caption in raw.get("captions", []) if str(caption).strip()]
        neg_caps = [
            str(caption)
            for caption in (raw.get("negative_captions") or [])
            if str(caption).strip()
        ]
        if not pos_caps or not neg_caps:
            continue
        image_path = image_root / str(raw[image_key])
        if not image_path.exists():
            continue
        person_id = int(raw["id"])
        pair_count = min(len(pos_caps), len(neg_caps))
        for index in range(pair_count):
            probes.append(
                CompositionalProbe(
                    image_path=image_path,
                    person_id=person_id,
                    positive_caption=pos_caps[index],
                    negative_caption=neg_caps[index],
                    caption_index=index,
                    perturbation=tag_perturbation(pos_caps[index], neg_caps[index]),
                )
            )
            if max_probes and len(probes) >= max_probes:
                return probes
    return probes


def load_reid_test_items(
    *,
    dataset_root: Path,
    dataset: str,
    split: str,
) -> tuple[list[ReIDImageItem], list[ReIDTextItem]]:
    directory, annotation_file, image_key = DATASET_SPECS[normalize_dataset_name(dataset)]
    image_root = dataset_root / directory / "imgs"
    raw_records = json.loads(
        (dataset_root / directory / annotation_file).read_text(encoding="utf-8")
    )
    images: list[ReIDImageItem] = []
    texts: list[ReIDTextItem] = []
    for raw in raw_records:
        if not isinstance(raw, dict) or str(raw.get("split", "")) != split:
            continue
        image_path = image_root / str(raw[image_key])
        if not image_path.exists():
            continue
        person_id = int(raw["id"])
        images.append(ReIDImageItem(image_path=image_path, person_id=person_id))
        for caption in raw.get("captions", []):
            caption = str(caption).strip()
            if caption:
                texts.append(ReIDTextItem(caption=caption, person_id=person_id))
    return images, texts


def collect_probe_vocabulary(probes: list[CompositionalProbe]) -> tuple[list[Path], list[str]]:
    image_paths = sorted({probe.image_path for probe in probes})
    captions: set[str] = set()
    for probe in probes:
        captions.add(probe.positive_caption)
        captions.add(probe.negative_caption)
    return image_paths, sorted(captions)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() * right.float()).sum().item())


def _ratio_metrics(correct_flags: list[bool]) -> dict[str, float]:
    total = len(correct_flags)
    if total == 0:
        return {"count": 0.0, "discrimination_rate": 0.0, "random_chance": 0.0}
    correct = sum(1 for flag in correct_flags if flag)
    return {
        "count": float(total),
        "correct": float(correct),
        "discrimination_rate": correct / total,
    }


def evaluate_sugarcrepe_probes(
    probes: list[CompositionalProbe],
    *,
    image_features: dict[Path, torch.Tensor],
    text_features: dict[str, torch.Tensor],
) -> dict[str, object]:
    overall_correct: list[bool] = []
    margins: list[float] = []
    by_tag: dict[PerturbationTag, list[bool]] = {
        "swap_att": [],
        "replace_att": [],
        "other": [],
    }
    pos_sims: list[float] = []
    neg_sims: list[float] = []
    for probe in probes:
        image_feat = image_features[probe.image_path]
        pos_feat = text_features[probe.positive_caption]
        neg_feat = text_features[probe.negative_caption]
        pos_sim = _cosine(image_feat, pos_feat)
        neg_sim = _cosine(image_feat, neg_feat)
        correct = pos_sim > neg_sim
        overall_correct.append(correct)
        margins.append(pos_sim - neg_sim)
        by_tag[probe.perturbation].append(correct)
        pos_sims.append(pos_sim)
        neg_sims.append(neg_sim)
    return {
        "benchmark": "sugarcrepe",
        "description": (
            "Image-conditioned hard-caption discrimination. "
            "Given image I and captions (C+, C-), score 1 when sim(I, C+) > sim(I, C-)."
        ),
        "random_chance": 0.5,
        "overall": {
            **_ratio_metrics(overall_correct),
            "mean_margin": sum(margins) / len(margins) if margins else 0.0,
            "mean_positive_similarity": sum(pos_sims) / len(pos_sims) if pos_sims else 0.0,
            "mean_negative_similarity": sum(neg_sims) / len(neg_sims) if neg_sims else 0.0,
        },
        "by_perturbation": {tag: _ratio_metrics(flags) for tag, flags in by_tag.items()},
    }


def retrieval_metrics(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: torch.Tensor,
    gallery_ids: torch.Tensor,
    *,
    top_k: list[int],
) -> dict[str, float]:
    similarity = query_features @ gallery_features.T
    ordered = similarity.argsort(dim=1, descending=True)
    gallery_ids_list = gallery_ids.tolist()
    metrics: dict[str, float] = {}
    for k in top_k:
        hits = 0
        for row, query_id in enumerate(query_ids.tolist()):
            top_indices = ordered[row, :k].tolist()
            top_ids = {gallery_ids_list[index] for index in top_indices}
            if query_id in top_ids:
                hits += 1
        metrics[f"top{k}"] = hits / len(query_ids)
    average_precisions: list[float] = []
    for row, query_id in enumerate(query_ids.tolist()):
        ranked_ids = [gallery_ids_list[index] for index in ordered[row].tolist()]
        relevant = [1 if pid == query_id else 0 for pid in ranked_ids]
        total_rel = sum(relevant)
        if total_rel == 0:
            continue
        hit_count = 0
        precision_sum = 0.0
        for index, rel in enumerate(relevant, start=1):
            if rel:
                hit_count += 1
                precision_sum += hit_count / index
        average_precisions.append(precision_sum / total_rel)
    metrics["mAP"] = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    return metrics


def load_yaml_config(
    config_path: Path,
    *,
    defaults: dict[str, object],
    required: tuple[str, ...],
) -> SimpleNamespace:
    if len(sys.argv) > 1:
        raise ValueError("CLI arguments are not supported. Edit the YAML config file instead.")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file {config_path} must contain a mapping.")
    unknown_keys = set(data) - set(defaults)
    if unknown_keys:
        raise ValueError(f"Unknown keys in config {config_path}: {sorted(unknown_keys)}")
    config = dict(defaults)
    config.update(data)
    missing = [key for key in required if not str(config.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Missing required keys in config {config_path}: {missing}")
    return SimpleNamespace(**config)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return torch.device(device_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_results_date_dir(run_time: datetime | None = None) -> Path:
    run_time = run_time or datetime.now()
    return Path("results") / run_time.strftime("%m-%d")


def output_model_tag(checkpoint_dir: str) -> str:
    return Path(checkpoint_dir).as_posix().replace("/", "-").replace(":", "-")


def require_config_path(env_key: str, config_prefix: str) -> Path:
    override = os.environ.get(env_key, "").strip()
    if not override:
        supported = ", ".join(
            f"configs/{config_prefix}_{name}.yaml"
            for name in ("cuhk_pedes", "icfg_pedes", "rstpreid")
        )
        raise ValueError(f"{env_key} is required. Set it to one of: {supported}")
    path = Path(override)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def load_hub_model(config: SimpleNamespace, device: torch.device) -> HubReIDCLIP:
    checkpoint_dir = Path(config.checkpoint_dir).expanduser()
    if not (checkpoint_dir / "vision-encoder").is_dir():
        raise FileNotFoundError(f"Hub checkpoint not found: {checkpoint_dir}")
    model = HubReIDCLIP(
        checkpoint_dir,
        device,
        int(config.img_height),
        int(config.img_width),
    )
    return model


class _UniqueImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "index": index}


class _UniqueTextDataset(Dataset):
    def __init__(self, captions: list[str]) -> None:
        self.captions = captions

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, index: int) -> dict:
        return {"caption": self.captions[index], "index": index}


class _ReIDImageDataset(Dataset):
    def __init__(self, items: list[ReIDImageItem], transform) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        with Image.open(item.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "person_id": item.person_id}


class _ReIDTextDataset(Dataset):
    def __init__(self, items: list[ReIDTextItem]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        return {"caption": item.caption, "person_id": item.person_id}


def _collate_images(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


def _collate_reid_images(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "person_ids": torch.tensor([item["person_id"] for item in batch], dtype=torch.long),
    }


def _collate_texts(batch: list[dict], tokenizer) -> dict:
    captions = [item["caption"] for item in batch]
    return {
        "text_tokens": tokenizer(captions),
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


class _TextCollate:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        return _collate_texts(batch, self.tokenizer)


class _ReIDTextCollate:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        return _collate_reid_texts(batch, self.tokenizer)


def _collate_reid_texts(batch: list[dict], tokenizer) -> dict:
    captions = [item["caption"] for item in batch]
    return {
        "text_tokens": tokenizer(captions),
        "person_ids": torch.tensor([item["person_id"] for item in batch], dtype=torch.long),
    }


@torch.no_grad()
def encode_unique_images(
    model: HubReIDCLIP,
    image_paths: list[Path],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    dataset = _UniqueImageDataset(image_paths, model.preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_collate_images,
    )
    features = torch.empty((len(image_paths), 0), dtype=torch.float32)
    for batch in tqdm(loader, desc="encode probe images", dynamic_ncols=True):
        images = batch["images"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            batch_features = F.normalize(model.encode_image(images), dim=-1).cpu()
        if features.numel() == 0:
            features = torch.empty((len(image_paths), batch_features.shape[1]))
        features[batch["indices"]] = batch_features
    return features


@torch.no_grad()
def encode_unique_texts(
    model: HubReIDCLIP,
    captions: list[str],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    dataset = _UniqueTextDataset(captions)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_TextCollate(model.tokenizer),
    )
    features = torch.empty((len(captions), 0), dtype=torch.float32)
    for batch in tqdm(loader, desc="encode probe texts", dynamic_ncols=True):
        text_tokens = batch["text_tokens"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            batch_features = F.normalize(model.encode_text(text_tokens), dim=-1).cpu()
        if features.numel() == 0:
            features = torch.empty((len(captions), batch_features.shape[1]))
        features[batch["indices"]] = batch_features
    return features


@torch.no_grad()
def encode_reid_images(
    model: HubReIDCLIP,
    items: list[ReIDImageItem],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = _ReIDImageDataset(items, model.preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_collate_reid_images,
    )
    features: list[torch.Tensor] = []
    person_ids: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="encode images", dynamic_ncols=True):
        images = batch["images"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            batch_features = F.normalize(model.encode_image(images), dim=-1).cpu()
        features.append(batch_features)
        person_ids.append(batch["person_ids"].cpu())
    return torch.cat(features, dim=0), torch.cat(person_ids, dim=0)


@torch.no_grad()
def encode_reid_texts(
    model: HubReIDCLIP,
    items: list[ReIDTextItem],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = _ReIDTextDataset(items)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_ReIDTextCollate(model.tokenizer),
    )
    features: list[torch.Tensor] = []
    person_ids: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="encode texts", dynamic_ncols=True):
        text_tokens = batch["text_tokens"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            batch_features = F.normalize(model.encode_text(text_tokens), dim=-1).cpu()
        features.append(batch_features)
        person_ids.append(batch["person_ids"].cpu())
    return torch.cat(features, dim=0), torch.cat(person_ids, dim=0)


def print_sugarcrepe_metrics(metrics: dict[str, object]) -> None:
    overall = metrics["overall"]
    assert isinstance(overall, dict)
    print("SugarCrepe-style compositional discrimination")
    print(f"overall discrimination_rate: {overall['discrimination_rate']:.4f}")
    print(f"overall mean_margin: {overall['mean_margin']:.4f}")
    by_perturbation = metrics["by_perturbation"]
    assert isinstance(by_perturbation, dict)
    for tag, payload in by_perturbation.items():
        assert isinstance(payload, dict)
        print(
            f"{tag}: discrimination_rate={payload['discrimination_rate']:.4f} "
            f"(n={int(payload['count'])})"
        )


def run_sugarcrepe(config_path: Path) -> None:
    config = load_yaml_config(
        config_path,
        defaults=SUGARCREPE_DEFAULTS,
        required=("dataset", "checkpoint_dir", "model_label"),
    )
    seed_everything(config.seed)
    device = get_device(config.device)
    dataset_name = normalize_dataset_name(config.dataset)
    dataset_root = resolve_dataset_root(config.dataset_root, config.env_file)
    annotation_path = resolve_negative_annotation_path(config, dataset_name)
    probes = load_compositional_probes(
        annotation_path=annotation_path,
        dataset_root=dataset_root,
        dataset=dataset_name,
        split=config.test_split,
        max_probes=config.max_probes,
    )
    if not probes:
        raise RuntimeError(f"No compositional probes found for split={config.test_split!r}.")
    image_paths, captions = collect_probe_vocabulary(probes)
    model = load_hub_model(config, device)
    use_amp = device.type == "cuda" and not config.no_amp
    image_features_tensor = encode_unique_images(
        model,
        image_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        use_amp=use_amp,
    )
    text_features_tensor = encode_unique_texts(
        model,
        captions,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        use_amp=use_amp,
    )
    image_feature_map = {path: image_features_tensor[index] for index, path in enumerate(image_paths)}
    text_feature_map = {
        caption: text_features_tensor[index] for index, caption in enumerate(captions)
    }
    metrics = evaluate_sugarcrepe_probes(
        probes,
        image_features=image_feature_map,
        text_features=text_feature_map,
    )
    print_sugarcrepe_metrics(metrics)
    run_time = datetime.now()
    timestamp = run_time.strftime("%Y%m%d_%H%M%S")
    results_dir = build_results_date_dir(run_time)
    model_tag = output_model_tag(str(config.checkpoint_dir))
    output = {
        "benchmark": "sugarcrepe",
        "model": config.model_label,
        "dataset": dataset_name,
        "domain": "person-reid-compositional-probe",
        "checkpoint": str(config.checkpoint_dir),
        "split": config.test_split,
        "negative_annotation": str(annotation_path),
        "probes": len(probes),
        "unique_images": len(image_paths),
        "unique_captions": len(captions),
        "metrics": metrics,
    }
    default_output_path = (
        results_dir / f"{dataset_name}_sugarcrepe_compositional_{model_tag}_{timestamp}.json"
    )
    output_path = Path(config.output_json) if config.output_json else default_output_path
    if not output_path.is_absolute() and config.output_json:
        output_path = results_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {output_path}")


def run_text_to_image_retrieval(config_path: Path) -> None:
    config = load_yaml_config(
        config_path,
        defaults=RETRIEVAL_DEFAULTS,
        required=("dataset", "checkpoint_dir", "model_label"),
    )
    seed_everything(config.seed)
    device = get_device(config.device)
    dataset_name = normalize_dataset_name(config.dataset)
    dataset_root = resolve_dataset_root(config.dataset_root, config.env_file)
    image_items, text_items = load_reid_test_items(
        dataset_root=dataset_root,
        dataset=dataset_name,
        split=config.test_split,
    )
    if config.max_image_samples:
        image_items = image_items[: int(config.max_image_samples)]
    if config.max_text_samples:
        text_items = text_items[: int(config.max_text_samples)]
    if not image_items or not text_items:
        raise RuntimeError(f"No records found for split={config.test_split!r}.")
    model = load_hub_model(config, device)
    use_amp = device.type == "cuda" and not config.no_amp
    image_features, image_ids = encode_reid_images(
        model,
        image_items,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        use_amp=use_amp,
    )
    text_features, text_ids = encode_reid_texts(
        model,
        text_items,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        use_amp=use_amp,
    )
    metrics = retrieval_metrics(
        query_features=text_features,
        gallery_features=image_features,
        query_ids=text_ids,
        gallery_ids=image_ids,
        top_k=list(config.top_k),
    )
    print("text-to-image retrieval")
    for key in [f"top{k}" for k in config.top_k] + ["mAP"]:
        print(f"{key}: {metrics[key]:.4f}")
    run_time = datetime.now()
    timestamp = run_time.strftime("%Y%m%d_%H%M%S")
    results_dir = build_results_date_dir(run_time)
    model_tag = output_model_tag(str(config.checkpoint_dir))
    output = {
        "model": config.model_label,
        "dataset": dataset_name,
        "domain": "person-reid",
        "direction": "text-to-image",
        "checkpoint": str(config.checkpoint_dir),
        "model_name": "ViT-B-32",
        "split": config.test_split,
        "queries": len(text_items),
        "gallery": len(image_items),
        "metrics": metrics,
    }
    default_output_path = (
        results_dir / f"{dataset_name}_text_to_image_{model_tag}_{timestamp}.json"
    )
    output_path = Path(config.output_json) if config.output_json else default_output_path
    if not output_path.is_absolute() and config.output_json:
        output_path = results_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_sugarcrepe(require_config_path("SUGARCREPE_CONFIG", "sugarcrepe"))
