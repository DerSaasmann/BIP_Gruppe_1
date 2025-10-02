#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Image Captioning (EN/DE) with ViT-GPT2 + Translation

- Loads a pretrained VisionEncoderDecoder model ("nlpconnect/vit-gpt2-image-captioning").
- Generates a caption in English using beam search.
- Optionally translates the caption into German using Helsinki-NLP/opus-mt-en-de.
- Designed for single-image usage but can be extended for batch processing.

Dependencies:
    pip install torch transformers pillow

Usage Example:
    python caption_image.py
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    pipeline
)

# ----------------------------- Device Selection ----------------------------- #
def pick_device() -> str:
    """Choose best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = pick_device()
print(f"[INFO] Using device: {DEVICE}")

# ----------------------------- Model Loading -------------------------------- #
# Vision-Language Captioning model
CAPTION_MODEL = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
).to(DEVICE)
CAPTION_PROCESSOR = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
CAPTION_TOKENIZER = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Translation model (optional)
try:
    TRANSLATOR = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-de",
        device=0 if DEVICE == "cuda" else -1
    )
except Exception as e:
    print(f"[WARN] Could not load translator: {e}")
    TRANSLATOR = None

# ----------------------------- Captioning Function -------------------------- #
def describe_image(img_path: str | Path, lang: str = "en", max_length: int = 20) -> str:
    """
    Generate a description for an image (English by default, optionally German).

    Args:
        img_path (str | Path): Path to the image file.
        lang (str): "en" for English, "de" for German translation.
        max_length (int): Maximum length of the generated caption.

    Returns:
        str: Caption in the requested language.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    pixel_values = CAPTION_PROCESSOR(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    # Generate English caption
    with torch.no_grad():
        output_ids = CAPTION_MODEL.generate(
            pixel_values,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    caption_en: str = CAPTION_TOKENIZER.decode(output_ids[0], skip_special_tokens=True).strip()
    if caption_en and caption_en[0].islower():
        caption_en = caption_en[0].upper() + caption_en[1:]
    if caption_en and not caption_en.endswith("."):
        caption_en += "."

    # If German requested, run translation (if available)
    if lang == "de":
        if TRANSLATOR is None:
            print("[WARN] Translator not available, returning English caption.")
            return caption_en
        out = TRANSLATOR(caption_en, max_length=200)
        return out[0]["translation_text"]

    return caption_en


# ----------------------------- Example Run ---------------------------------- #
if __name__ == "__main__":
    example_img = "1-2025-0025-000-000.JPG"
    try:
        caption = describe_image(example_img, lang="de")
        print("Beschreibung:", caption)
    except Exception as e:
        print(f"[ERROR] Could not describe {example_img}: {e}")
