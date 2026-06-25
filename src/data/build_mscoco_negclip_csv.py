#!/usr/bin/env python3
"""Build a NegCLIP-compatible MSCOCO TSV.

The training dataloader in this repository expects a tab-separated file with:

    filepath, title, neg_caption, neg_image

`neg_caption` must be a Python-list literal of hard negative captions.
`neg_image` must be a Python-list literal of row indices in the same TSV.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    label: str


def clean_caption(caption: str) -> str:
    return " ".join(str(caption).strip().split())


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    for punct in [".", ",", "!", "?", ";", ":", "%", ")", "]", "}"]:
        text = text.replace(f" {punct}", punct)
    for punct in ["(", "[", "{"]:
        text = text.replace(f"{punct} ", punct)
    for suffix in ["n't", "'m", "'re", "'ve", "'ll", "'d", "'s"]:
        text = text.replace(f" {suffix}", suffix)
    text = text.replace("$ ", "$")
    return clean_caption(text)


def trim_span(doc, start: int, end: int) -> tuple[int, int]:
    while start < end and doc[start].is_punct:
        start += 1
    while end > start and doc[end - 1].is_punct:
        end -= 1
    return start, end


def extract_candidates(doc, min_np_tokens: int) -> list[SpanCandidate]:
    candidates: list[SpanCandidate] = []
    seen: set[tuple[int, int, str]] = set()

    def add(start: int, end: int, label: str) -> None:
        start, end = trim_span(doc, start, end)
        if start >= end:
            return
        if not any(token.is_alpha for token in doc[start:end]):
            return
        key = (start, end, label)
        if key in seen:
            return
        seen.add(key)
        candidates.append(SpanCandidate(start, end, label))

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            add(token.i, token.i + 1, "noun")
        elif token.pos_ == "ADJ":
            add(token.i, token.i + 1, "adjective")
        elif token.pos_ == "ADV":
            add(token.i, token.i + 1, "adverb")

    for chunk in doc.noun_chunks:
        if chunk.end - chunk.start >= min_np_tokens:
            add(chunk.start, chunk.end, "noun_phrase")

    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        if token.pos_ == "AUX" and token.head.pos_ == "VERB":
            continue

        phrase_indices = {token.i}
        for child in token.children:
            if child.dep_ in {"aux", "auxpass", "neg", "prt", "advmod"}:
                phrase_indices.add(child.i)

        start = min(phrase_indices)
        end = max(phrase_indices) + 1
        if phrase_indices == set(range(start, end)):
            add(start, end, "verb_phrase")
        else:
            add(token.i, token.i + 1, "verb_phrase")

    return candidates


def spans_overlap(left: SpanCandidate, right: SpanCandidate) -> bool:
    return left.start < right.end and right.start < left.end


def swap_spans(doc, left: SpanCandidate, right: SpanCandidate) -> str:
    if right.start < left.start:
        left, right = right, left
    tokens = [token.text for token in doc]
    swapped = (
        tokens[: left.start]
        + tokens[right.start : right.end]
        + tokens[left.end : right.start]
        + tokens[left.start : left.end]
        + tokens[right.end :]
    )
    return detokenize(swapped)


def mine_negative_captions(
    doc,
    rng: random.Random,
    max_negatives: int,
    min_np_tokens: int,
) -> list[str]:
    original = clean_caption(doc.text)
    candidates = extract_candidates(doc, min_np_tokens=min_np_tokens)
    pairs = [
        (left, right)
        for index, left in enumerate(candidates)
        for right in candidates[index + 1 :]
        if not spans_overlap(left, right)
    ]
    rng.shuffle(pairs)

    negatives: list[str] = []
    seen = {original}
    for left, right in pairs:
        negative = swap_spans(doc, left, right)
        if negative in seen:
            continue
        seen.add(negative)
        negatives.append(negative)
        if len(negatives) >= max_negatives:
            break
    return negatives


def load_spacy_model(model_name: str):
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit(
            "spaCy is required for negative text mining. Install it with:\n"
            "  uv pip install spacy\n"
            f"  uv run python -m spacy download {model_name}"
        ) from exc

    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' is not installed. Install it with:\n"
            f"  uv run python -m spacy download {model_name}"
        ) from exc


def annotation_paths(coco_root: Path, split: str) -> tuple[Path, Path]:
    captions_json = coco_root / "annotations" / f"captions_{split}.json"
    image_dir = coco_root / split
    if not captions_json.exists():
        raise FileNotFoundError(f"Missing captions annotation: {captions_json}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    return captions_json, image_dir


def load_coco(coco_root: Path, split: str, limit: int | None) -> tuple[list[dict], list[dict], Path]:
    captions_json, image_dir = annotation_paths(coco_root, split)
    with captions_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    annotations = data["annotations"]
    if limit is not None:
        annotations = annotations[:limit]
        keep_image_ids = {ann["image_id"] for ann in annotations}
        images = [image for image in data["images"] if image["id"] in keep_image_ids]
    else:
        images = data["images"]
    return images, annotations, image_dir


def build_text_rows(
    annotations: list[dict],
    images_by_id: dict[int, dict],
    image_dir: Path,
    spacy_model: str,
    max_neg_captions: int,
    min_np_tokens: int,
    seed: int,
    spacy_batch_size: int,
) -> list[dict]:
    from tqdm import tqdm

    nlp = load_spacy_model(spacy_model)
    rng = random.Random(seed)
    captions = [clean_caption(annotation["caption"]) for annotation in annotations]
    rows: list[dict] = []

    docs = nlp.pipe(captions, batch_size=spacy_batch_size)
    iterator = zip(annotations, captions, docs)
    for annotation, caption, doc in tqdm(iterator, total=len(annotations), desc="Mining text negatives"):
        image = images_by_id.get(annotation["image_id"])
        if image is None:
            continue
        filepath = image_dir / image["file_name"]
        if not filepath.exists():
            continue

        neg_captions = mine_negative_captions(
            doc,
            rng=rng,
            max_negatives=max_neg_captions,
            min_np_tokens=min_np_tokens,
        )
        if not neg_captions:
            continue

        rows.append(
            {
                "filepath": str(filepath),
                "title": caption,
                "neg_caption": neg_captions,
                "image_id": int(annotation["image_id"]),
                "caption_id": int(annotation["id"]),
                "file_name": image["file_name"],
            }
        )
    return rows


def import_open_clip():
    repo_src = Path(__file__).resolve().parents[1]
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    import open_clip

    return open_clip


def compute_image_features(
    image_records: list[dict],
    model_name: str,
    pretrained: str,
    batch_size: int,
    workers: int,
    device_name: str,
):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    open_clip = import_open_clip()
    device = torch.device(device_name)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()

    class ImageDataset(Dataset):
        def __init__(self, records: list[dict]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            record = self.records[index]
            image = Image.open(record["filepath"]).convert("RGB")
            return preprocess(image), index

    loader = DataLoader(
        ImageDataset(image_records),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    features = []
    image_ids = [int(record["image_id"]) for record in image_records]
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Encoding images"):
            images = images.to(device=device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features.float(), dim=-1)
            features.append(image_features.cpu())

    return image_ids, torch.cat(features, dim=0)


def load_or_compute_image_features(
    image_records: list[dict],
    feature_cache: Path,
    force_recompute: bool,
    model_name: str,
    pretrained: str,
    batch_size: int,
    workers: int,
    device: str,
):
    import torch

    expected_ids = [int(record["image_id"]) for record in image_records]
    if feature_cache.exists() and not force_recompute:
        payload = torch.load(feature_cache, map_location="cpu")
        if payload.get("image_ids") == expected_ids:
            return payload["image_ids"], payload["features"].float()

    image_ids, features = compute_image_features(
        image_records=image_records,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        workers=workers,
        device_name=device,
    )
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_ids": image_ids, "features": features.half()}, feature_cache)
    return image_ids, features


def compute_image_neighbors(
    image_ids: list[int],
    features,
    candidate_neg_images: int,
    block_size: int,
    device_name: str,
    dtype: str,
) -> dict[str, list[int]]:
    import torch
    from tqdm import tqdm

    device = torch.device(device_name)
    compute_dtype = torch.float16 if dtype == "float16" and device.type == "cuda" else torch.float32
    features = features.to(device=device, dtype=compute_dtype)
    neighbor_count = min(candidate_neg_images, max(len(image_ids) - 1, 1))
    neighbors: dict[str, list[int]] = {}

    for start in tqdm(range(0, len(image_ids), block_size), desc="Mining image negatives"):
        end = min(start + block_size, len(image_ids))
        similarities = features[start:end] @ features.t()
        row_indices = torch.arange(end - start, device=device)
        col_indices = torch.arange(start, end, device=device)
        similarities[row_indices, col_indices] = -float("inf")
        _, top_indices = torch.topk(similarities, k=neighbor_count, dim=1)

        top_indices = top_indices.cpu().tolist()
        for offset, indices in enumerate(top_indices):
            neighbors[str(image_ids[start + offset])] = [int(image_ids[index]) for index in indices]

    return neighbors


def load_or_compute_image_neighbors(
    image_records: list[dict],
    neighbors_path: Path,
    feature_cache: Path,
    force_recompute_neighbors: bool,
    force_recompute_features: bool,
    model_name: str,
    pretrained: str,
    batch_size: int,
    workers: int,
    device: str,
    candidate_neg_images: int,
    block_size: int,
    dtype: str,
) -> dict[str, list[int]]:
    if neighbors_path.exists() and not force_recompute_neighbors:
        with neighbors_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    image_ids, features = load_or_compute_image_features(
        image_records=image_records,
        feature_cache=feature_cache,
        force_recompute=force_recompute_features,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        workers=workers,
        device=device,
    )
    neighbors = compute_image_neighbors(
        image_ids=image_ids,
        features=features,
        candidate_neg_images=candidate_neg_images,
        block_size=block_size,
        device_name=device,
        dtype=dtype,
    )
    neighbors_path.parent.mkdir(parents=True, exist_ok=True)
    with neighbors_path.open("w", encoding="utf-8") as handle:
        json.dump(neighbors, handle)
    return neighbors


def image_records_for_mining(images: list[dict], image_dir: Path) -> list[dict]:
    records = []
    for image in images:
        filepath = image_dir / image["file_name"]
        if filepath.exists():
            records.append({"image_id": int(image["id"]), "filepath": str(filepath)})
    return records


def find_eligible_image_ids(
    rows_by_image: dict[int, list[int]],
    image_neighbors: dict[str, list[int]],
    min_neg_images: int,
) -> set[int]:
    eligible = set(rows_by_image)
    while True:
        next_eligible = {
            image_id
            for image_id in eligible
            if sum(1 for neighbor_id in image_neighbors.get(str(image_id), []) if neighbor_id in eligible)
            >= min_neg_images
        }
        if next_eligible == eligible:
            return eligible
        eligible = next_eligible


def attach_negative_image_rows(
    rows: list[dict],
    image_neighbors: dict[str, list[int]],
    neg_images: int,
    min_neg_images: int,
    row_neighbor_mode: str,
) -> list[dict]:
    rows_by_image: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        rows_by_image[int(row["image_id"])].append(index)

    eligible_image_ids = find_eligible_image_ids(
        rows_by_image=rows_by_image,
        image_neighbors=image_neighbors,
        min_neg_images=min_neg_images,
    )
    final_rows = [dict(row) for row in rows if int(row["image_id"]) in eligible_image_ids]

    final_rows_by_image: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(final_rows):
        final_rows_by_image[int(row["image_id"])].append(index)

    output_rows: list[dict] = []
    for row in final_rows:
        neighbor_image_ids = [
            neighbor_id
            for neighbor_id in image_neighbors.get(str(row["image_id"]), [])
            if neighbor_id in final_rows_by_image and neighbor_id != row["image_id"]
        ][:neg_images]

        neg_row_indices: list[int] = []
        for neighbor_id in neighbor_image_ids:
            if row_neighbor_mode == "all":
                neg_row_indices.extend(final_rows_by_image[neighbor_id])
            else:
                neg_row_indices.append(final_rows_by_image[neighbor_id][0])

        if len(set(neighbor_image_ids)) < min_neg_images or not neg_row_indices:
            continue
        row["neg_image"] = neg_row_indices
        output_rows.append(row)

    return output_rows


def write_tsv(rows: Iterable[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "title", "neg_caption", "neg_image", "image_id", "caption_id", "file_name"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filepath": row["filepath"],
                    "title": row["title"],
                    "neg_caption": repr(row["neg_caption"]),
                    "neg_image": repr(row["neg_image"]),
                    "image_id": row["image_id"],
                    "caption_id": row["caption_id"],
                    "file_name": row["file_name"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=Path("/data/jayn2u/datasets/ms-coco"))
    parser.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Debug with the first N captions.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--spacy-batch-size", type=int, default=512)
    parser.add_argument("--max-neg-captions", type=int, default=5)
    parser.add_argument("--min-np-tokens", type=int, default=3)

    parser.add_argument("--neg-images", type=int, default=3)
    parser.add_argument("--min-neg-images", type=int, default=1)
    parser.add_argument("--candidate-neg-images", type=int, default=20)
    parser.add_argument("--row-neighbor-mode", choices=["first", "all"], default="first")
    parser.add_argument("--image-neighbors", type=Path, default=None)
    parser.add_argument("--image-features", type=Path, default=None)
    parser.add_argument("--force-recompute-image-neighbors", action="store_true")
    parser.add_argument("--force-recompute-image-features", action="store_true")

    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--similarity-block-size", type=int, default=1024)
    parser.add_argument("--similarity-dtype", choices=["float32", "float16"], default="float16")
    return parser.parse_args()


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    args = parse_args()
    device = args.device or default_device()

    images, annotations, image_dir = load_coco(args.coco_root, args.split, args.limit)
    images_by_id = {int(image["id"]): image for image in images}

    rows = build_text_rows(
        annotations=annotations,
        images_by_id=images_by_id,
        image_dir=image_dir,
        spacy_model=args.spacy_model,
        max_neg_captions=args.max_neg_captions,
        min_np_tokens=args.min_np_tokens,
        seed=args.seed,
        spacy_batch_size=args.spacy_batch_size,
    )
    if not rows:
        raise SystemExit("No rows survived negative text mining.")

    neighbors_path = args.image_neighbors or args.output.with_suffix(".image_neighbors.json")
    feature_cache = args.image_features or args.output.with_suffix(".image_features.pt")
    image_records = image_records_for_mining(images, image_dir)
    image_neighbors = load_or_compute_image_neighbors(
        image_records=image_records,
        neighbors_path=neighbors_path,
        feature_cache=feature_cache,
        force_recompute_neighbors=args.force_recompute_image_neighbors,
        force_recompute_features=args.force_recompute_image_features,
        model_name=args.model,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        candidate_neg_images=max(args.candidate_neg_images, args.neg_images),
        block_size=args.similarity_block_size,
        dtype=args.similarity_dtype,
    )

    rows = attach_negative_image_rows(
        rows=rows,
        image_neighbors=image_neighbors,
        neg_images=args.neg_images,
        min_neg_images=args.min_neg_images,
        row_neighbor_mode=args.row_neighbor_mode,
    )
    if not rows:
        raise SystemExit("No rows survived negative image mining.")

    write_tsv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Image-neighbor cache: {neighbors_path}")
    print(f"Image-feature cache: {feature_cache}")


if __name__ == "__main__":
    main()
