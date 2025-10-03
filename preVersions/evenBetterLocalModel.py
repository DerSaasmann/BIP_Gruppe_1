#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Image Describer for Museum Objects

- Generates up to 3 English caption variants per image using InstructBLIP (fallback to BLIP-large).
- Translates each variant to German (Helsinki-NLP/opus-mt-en-de).
- Selects a "best" caption via a simple domain-aware scoring heuristic.
- Exports a CSV with all variants (EN/DE) and the selected best candidates.

Usage:
    python batch_describe_images.py --imgdir /path/to/images --out captions.csv
    # or a single image:
    python batch_describe_images.py --image /path/to/img.jpg --out captions.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import pandas as pd
from PIL import Image
import torch
from transformers import (
    pipeline,
    BlipProcessor, BlipForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
)

# ----------------------------- Configuration -------------------------------- #

# Image file extensions we will process
IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# English prompt for museum-style captions (kept concise but specific)
PROMPT_EN: str = (
    "Describe this object in detail as if it were a museum catalog entry. "
    "Mention object type, materials, color, shape, function, and notable visual features. "
    "Avoid guessing brand names unless they are clearly visible."
)

# Generation parameters (sampling and beams).
# Note: sampling yields diversity; beams yield stability. We combine both.
GEN_KWARGS_SAMPLING = dict(
    max_length=128,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    no_repeat_ngram_size=3,
    repetition_penalty=1.15,
    num_return_sequences=3,
    early_stopping=True,
)

GEN_KWARGS_BEAM = dict(
    max_length=128,
    num_beams=8,
    no_repeat_ngram_size=3,
    repetition_penalty=1.15,
    length_penalty=1.05,
    early_stopping=True,
    num_return_sequences=1,
)

# Phrases to penalize (too generic); tokens to reward (domain-relevant)
GENERIC_PHRASES: Tuple[str, ...] = (
    "a photo of", "a picture of", "on a table", "in the room",
    "in the background", "in the foreground",
)
MUSEUM_TOKENS: Tuple[str, ...] = (
    "object", "device", "instrument", "apparatus", "mechanical", "metal",
    "wood", "plastic", "glass", "rubber", "textile",
    "typewriter", "camera", "telephone", "calculator", "radio", "recorder",
    "keyboard", "dial", "lever", "switch", "handle", "label", "plaque"
)


# ----------------------------- Utilities ------------------------------------ #

def pick_device() -> str:
    """Choose the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = pick_device()
print(f"[INFO] Device: {DEVICE}")


def all_images_in_dir(imgdir: Path) -> List[Path]:
    """Recursively list all image files under `imgdir` using the allowed extensions."""
    return sorted([p for p in imgdir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def dedup_preserve_order(seq: Iterable[str]) -> List[str]:
    """De-duplicate strings while preserving order (case-insensitive check)."""
    seen, out = set(), []
    for s in seq:
        s_norm = s.strip().lower()
        if s_norm and s_norm not in seen:
            seen.add(s_norm)
            out.append(s.strip())
    return out


# ----------------------------- Model Loading -------------------------------- #

class VisionModel:
    """
    Small wrapper holding either an InstructBLIP or BLIP model, plus its processor and mode tag.
    """
    def __init__(self, mode: str, processor, model):
        self.mode = mode  # "instructblip" or "blip"
        self.processor = processor
        self.model = model


def load_vision_model() -> VisionModel:
    """
    Try to load InstructBLIP (higher quality out-of-the-box). If it fails,
    gracefully fall back to BLIP-large.
    """
    try:
        model_id = "Salesforce/instructblip-flan-t5-xl"
        print(f"[INFO] Loading InstructBLIP: {model_id}")
        proc = InstructBlipProcessor.from_pretrained(model_id)
        mdl = InstructBlipForConditionalGeneration.from_pretrained(model_id)
        mdl.to(DEVICE)
        mdl.eval()
        return VisionModel("instructblip", proc, mdl)
    except Exception as e:
        print(f"[WARN] Could not load InstructBLIP: {e}")
        print("[INFO] Falling back to BLIP-large …")
        model_id = "Salesforce/blip-image-captioning-large"
        proc = BlipProcessor.from_pretrained(model_id)
        mdl = BlipForConditionalGeneration.from_pretrained(model_id)
        mdl.to(DEVICE)
        mdl.eval()
        return VisionModel("blip", proc, mdl)


VISION: VisionModel = load_vision_model()

# Translation pipeline (can run on CPU; we keep it optional)
try:
    TRANSLATOR = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de", device=-1)
except Exception as e:
    print(f"[WARN] Translation model unavailable: {e}")
    TRANSLATOR = None


# ----------------------------- Captioning ----------------------------------- #

def _decode_sequences(proc, sequences) -> List[str]:
    """Decode a batch of generated token sequences to strings, strip and prettify."""
    texts: List[str] = []
    for ids in sequences:
        t = proc.decode(ids, skip_special_tokens=True).strip()
        # Cosmetic cleanup
        t = re.sub(r"\s+", " ", t).strip()
        # Remove accidental echoes like "Describe …:"
        t = re.sub(r"^(describe|beschreibe).{0,60}:", "", t, flags=re.I).strip()
        if t and t[0].islower():
            t = t[0].upper() + t[1:]
        if t and not t.endswith("."):
            t += "."
        texts.append(t)
    return texts


def generate_captions(image_path: Path) -> List[str]:
    """
    Generate up to 3 museum-style English captions for a single image.
    1) Prefer diverse sampling (3 sequences).
    2) If diversity fails (e.g., short/duplicated outputs), add 1 beam-searched caption.
    """
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs depending on model mode
    if VISION.mode == "instructblip":
        inputs = VISION.processor(images=image, text=PROMPT_EN, return_tensors="pt")
    else:
        inputs = VISION.processor(images=image, return_tensors="pt")

    inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # 1) Sampling for diversity
    with torch.no_grad():
        out = VISION.model.generate(**inputs, **GEN_KWARGS_SAMPLING)
    captions = _decode_sequences(VISION.processor, out)

    # 2) Fallback: add one beam-searched caption if diversity is insufficient
    if len(dedup_preserve_order(captions)) < 2:
        with torch.no_grad():
            out_beam = VISION.model.generate(**inputs, **GEN_KWARGS_BEAM)
        beam_text = _decode_sequences(VISION.processor, out_beam)[0]
        captions = dedup_preserve_order([*captions, beam_text])

    return dedup_preserve_order(captions)


# ----------------------------- Heuristic Scoring ---------------------------- #

def score_caption(text: str) -> float:
    """
    Simple domain-aware score:
    - base length (longer tends to be more informative),
    - bonus for museum/technical tokens,
    - penalty for generic filler phrases or artifacts.
    """
    t = text.lower()
    score = float(len(text))
    score += sum(1.5 for w in MUSEUM_TOKENS if w in t)          # domain bonus
    score -= sum(1.0 for g in GENERIC_PHRASES if g in t)        # generic penalty
    score -= 0.5 * t.count("  ")                                # formatting penalty
    score -= 1.0 * t.count("…")                                 # ellipsis penalty
    return score


def pick_best(captions: List[str]) -> str:
    """Choose the caption with the highest heuristic score."""
    return max(captions, key=score_caption) if captions else ""


# ----------------------------- Translation ---------------------------------- #

def translate_to_german(text: str) -> str:
    """
    Translate a single English caption to German.
    If the translation model is not available, return the original English text.
    """
    if not text:
        return ""
    if TRANSLATOR is None:
        return text
    try:
        out = TRANSLATOR(text, max_length=256)
        return out[0]["translation_text"]
    except Exception as e:
        print(f"[WARN] Translation failed: {e}")
        return text


# ----------------------------- Processing ----------------------------------- #

def process_one_image(img_path: Path) -> Dict[str, str]:
    """
    Generate 3 English captions (with fallback), select the best, and translate to German.
    Returns a flat dict suitable for CSV export.
    """
    captions_en = generate_captions(img_path)

    # Ensure we have up to 3 unique candidates
    caps_en = (captions_en + ["", "", ""])[:3]
    best_en = pick_best(captions_en)

    # Translate variants
    caps_de = [translate_to_german(c) if c else "" for c in caps_en]
    best_de = translate_to_german(best_en) if best_en else ""

    return {
        "image_path": str(img_path),
        "caption_en_1": caps_en[0],
        "caption_en_2": caps_en[1],
        "caption_en_3": caps_en[2],
        "best_en": best_en,
        "caption_de_1": caps_de[0],
        "caption_de_2": caps_de[1],
        "caption_de_3": caps_de[2],
        "best_de": best_de,
    }


def collect_images(args: argparse.Namespace) -> List[Path]:
    """Resolve CLI arguments to a list of image paths."""
    if args.image:
        p = Path(args.image).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Unsupported file extension: {p.suffix}")
        return [p]

    if args.imgdir:
        d = Path(args.imgdir).expanduser()
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")
        images = all_images_in_dir(d)
        if not images:
            raise SystemExit("No matching image files found in directory.")
        return images

    raise SystemExit("Please provide either --image or --imgdir.")


# ----------------------------- CLI / Main ----------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Batch Image Describer (InstructBLIP → BLIP fallback, EN→DE translation)."
    )
    ap.add_argument("--imgdir", type=str, help="Directory containing images (searched recursively).")
    ap.add_argument("--image", type=str, help="Single image file to process.")
    ap.add_argument("--out", type=str, default="captions.csv", help="Output CSV file.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve images from CLI parameters
    images = collect_images(args)
    print(f"[INFO] Found {len(images)} image(s).")

    rows: List[Dict[str, str]] = []
    for i, img in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {img}")
        try:
            row = process_one_image(img)
        except Exception as e:
            # Ensure a row is written, even if captioning failed for this file
            print(f"[ERROR] Failed on {img}: {e}")
            row = {"image_path": str(img), "best_en": "", "best_de": ""}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] Done → {out_path}")


if __name__ == "__main__":
    main()
