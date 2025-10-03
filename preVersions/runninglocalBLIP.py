#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Captioning with BLIP (Salesforce/blip-image-captioning-large)

- Generates captions for an input image using BLIP (large variant).
- Optionally uses a short English prompt for more controlled captions.
- Supports optional German translation using Helsinki-NLP/opus-mt-en-de.
- Designed for single-image usage but can be extended to batch processing.

Dependencies:
    pip install torch transformers pillow

Usage Example:
    python blip_caption.py
"""

from pathlib import Path
from typing import Union, Optional

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# ----------------------------- Device Selection ----------------------------- #
def pick_device() -> str:
    """Choose the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = pick_device()
print(f"[INFO] Using device: {DEVICE}")

# ----------------------------- Model Loading -------------------------------- #
MODEL_ID = "Salesforce/blip-image-captioning-large"  # Larger = better quality
PROCESSOR = BlipProcessor.from_pretrained(MODEL_ID)
MODEL = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
MODEL.eval()  # important for inference mode

# Translation model (optional, runs on CPU)
try:
    TRANSLATOR = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-de",
        device=-1  # force CPU for compatibility
    )
except Exception as e:
    print(f"[WARN] Translator not available: {e}")
    TRANSLATOR = None

# ----------------------------- Captioning Function -------------------------- #
def describe_image(img_path: Union[str, Path], lang: str = "de", use_prompt: bool = False) -> str:
    """
    Generate an image caption in English (default) or German.

    Args:
        img_path (str | Path): Path to the image file.
        lang (str): Output language ("en" or "de").
        use_prompt (bool): If True, adds a short English prompt for generation.

    Returns:
        str: Generated caption.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Input preparation: with or without prompt
    if use_prompt:
        prompt = "Describe the object in the photo."
        inputs = PROCESSOR(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    else:
        inputs = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

    # Caption generation
    with torch.no_grad():
        output_ids = MODEL.generate(
            **inputs,
            max_length=96,
            num_beams=8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.05,
            early_stopping=True
        )

    # Decode and clean English caption
    caption_en: str = PROCESSOR.decode(output_ids[0], skip_special_tokens=True).strip()
    if caption_en and caption_en[0].islower():
        caption_en = caption_en[0].upper() + caption_en[1:]
    if caption_en and not caption_en.endswith("."):
        caption_en += "."

    # Optional German translation
    if lang == "de" and TRANSLATOR is not None:
        return TRANSLATOR(caption_en, max_length=256)[0]["translation_text"]

    return caption_en


# ----------------------------- Example Run ---------------------------------- #
if __name__ == "__main__":
    example_img = "1-2025-0025-000-000.JPG"
    try:
        caption = describe_image(example_img, lang="de", use_prompt=False)
        print("Beschreibung:", caption)
    except Exception as e:
        print(f"[ERROR] Could not describe {example_img}: {e}")
