#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Image Describer for Museum Objects
- erzeugt pro Bild 3 Caption-Varianten (EN) mit InstructBLIP (fallback BLIP-large)
- übersetzt nach DE (Helsinki-NLP/opus-mt-en-de)
- wählt die "beste" Variante heuristisch
- exportiert CSV mit EN/DE-Varianten + Auswahl

Nutzung:
  python batch_describe_images.py --imgdir /path/to/images --out captions.csv
  # oder einzelnes Bild:
  python batch_describe_images.py --image /path/to/img.jpg --out captions.csv
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
import re
import pandas as pd
from PIL import Image
import torch

from transformers import (
    pipeline,
    BlipProcessor, BlipForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
)

# ------------------ Gerätewahl ------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
print("Device:", DEVICE)

# ------------------ Modelle laden ------------------
def load_models():
    """
    Versucht zuerst InstructBLIP (deutlich besser), sonst BLIP-large.
    Gibt (mode, processor, model) zurück, wobei mode in {"instructblip", "blip"} liegt.
    """
    try:
        model_id = "Salesforce/instructblip-flan-t5-xl"
        print("Lade InstructBLIP:", model_id)
        proc = InstructBlipProcessor.from_pretrained(model_id)
        mdl = InstructBlipForConditionalGeneration.from_pretrained(model_id)
        mdl.to(DEVICE); mdl.eval()
        return "instructblip", proc, mdl
    except Exception as e:
        print("[Hinweis] Konnte InstructBLIP nicht laden:", e)
        print("Falle zurück auf BLIP-large…")
        model_id = "Salesforce/blip-image-captioning-large"
        proc = BlipProcessor.from_pretrained(model_id)
        mdl = BlipForConditionalGeneration.from_pretrained(model_id)
        mdl.to(DEVICE); mdl.eval()
        return "blip", proc, mdl

MODE, VISION_PROC, VISION_MODEL = load_models()

# Übersetzungs-Pipeline (CPU ist ok)
try:
    TRANSLATOR = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de", device=-1)
except Exception as e:
    print("[Hinweis] Übersetzungsmodell nicht verfügbar:", e)
    TRANSLATOR = None

# ------------------ Captioning ------------------
PROMPT_EN = (
    "Describe this object in detail as if it were a museum catalog entry. "
    "Mention object type, materials, color, shape, function, and notable visual features. "
    "Avoid guessing brand names unless clearly visible."
)

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

def dedup_preserve_order(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        k = s.strip().lower()
        if k not in seen and s.strip():
            seen.add(k); out.append(s)
    return out

def generate_captions(img_path: Path) -> List[str]:
    image = Image.open(img_path).convert("RGB")

    if MODE == "instructblip":
        # InstructBLIP erwartet text + image
        inputs = VISION_PROC(images=image, text=PROMPT_EN, return_tensors="pt")
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = VISION_MODEL.generate(**inputs, **GEN_KWARGS_SAMPLING)
        texts = [VISION_PROC.decode(ids, skip_special_tokens=True).strip() for ids in out]
    else:
        # BLIP: optional mit kurzem Prompt; hier ohne Prompt stabiler
        inputs = VISION_PROC(images=image, return_tensors="pt")
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = VISION_MODEL.generate(**inputs, **GEN_KWARGS_SAMPLING)
        texts = [VISION_PROC.decode(ids, skip_special_tokens=True).strip() for ids in out]

    # Kosmetik & harte Echos des Prompts entfernen
    cleaned = []
    for t in texts:
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"^(describe|beschreibe).{0,60}:", "", t, flags=re.I).strip()
        if t and t[0].islower(): t = t[0].upper() + t[1:]
        if t and not t.endswith("."): t += "."
        cleaned.append(t)
    return dedup_preserve_order(cleaned)

# ------------------ Heuristische Auswahl der "besten" Caption ------------------
GENERIC_PHRASES = [
    "a photo of", "a picture of", "on a table", "in the room",
    "in the background", "in the foreground",
]

MUSEUM_TOKENS = [
    "object", "device", "instrument", "apparatus", "mechanical", "metal",
    "wood", "plastic", "glass", "rubber", "textile",
    "typewriter", "camera", "telephone", "calculator", "radio", "recorder",
    "keyboard", "dial", "lever", "switch", "handle", "label", "plaque"
]

def score_caption(txt: str) -> float:
    t = txt.lower()
    score = len(txt)  # Länge als Basis
    score += sum(1.5 for w in MUSEUM_TOKENS if w in t)  # Bonus für objektrelevante Wörter
    score -= sum(1.0 for g in GENERIC_PHRASES if g in t)  # Malus für Floskeln
    score -= 0.5 * t.count("  ")  # Malus für doppelte Leerzeichen
    score -= 1.0 * t.count("…")
    return score

def pick_best(captions: List[str]) -> str:
    if not captions:
        return ""
    return max(captions, key=score_caption)

# ------------------ Übersetzung ------------------
def to_german(text: str) -> str:
    if not text:
        return ""
    if TRANSLATOR is None:
        return text  # Fallback: Englisch zurückgeben
    out = TRANSLATOR(text, max_length=256)
    return out[0]["translation_text"]

# ------------------ Batch/Single Runner ------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def all_images_in_dir(imgdir: Path) -> List[Path]:
    return sorted([p for p in imgdir.rglob("*") if p.suffix.lower() in IMG_EXTS])

def process_one(img_path: Path) -> Dict[str, str]:
    caps_en = generate_captions(img_path)
    # falls Sampling zu viel Duplikate oder leer -> Via Beam einen Zusatz holen
    if len(caps_en) < 2:
        image = Image.open(img_path).convert("RGB")
        if MODE == "instructblip":
            inputs = VISION_PROC(images=image, text=PROMPT_EN, return_tensors="pt")
        else:
            inputs = VISION_PROC(images=image, return_tensors="pt")
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = VISION_MODEL.generate(**inputs, **GEN_KWARGS_BEAM)
        beam_txt = (VISION_PROC.decode(out[0], skip_special_tokens=True).strip() + ".").replace("..",".")
        caps_en = dedup_preserve_order(caps_en + [beam_txt])

    best_en = pick_best(caps_en)
    caps_de = [to_german(c) for c in caps_en]
    best_de = to_german(best_en)

    # Ergebnis-Row
    row = {
        "image_path": str(img_path),
        "caption_en_1": caps_en[0] if len(caps_en) >= 1 else "",
        "caption_en_2": caps_en[1] if len(caps_en) >= 2 else "",
        "caption_en_3": caps_en[2] if len(caps_en) >= 3 else "",
        "best_en": best_en,
        "caption_de_1": caps_de[0] if len(caps_de) >= 1 else "",
        "caption_de_2": caps_de[1] if len(caps_de) >= 2 else "",
        "caption_de_3": caps_de[2] if len(caps_de) >= 3 else "",
        "best_de": best_de,
    }
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", type=str, help="Ordner mit Bildern (rekursiv)")
    ap.add_argument("--image", type=str, help="Einzelnes Bild")
    ap.add_argument("--out", type=str, default="captions.csv", help="Ausgabe-CSV")
    args = ap.parse_args()

    imgs: List[Path] = []
    if args.image:
        p = Path(args.image)
        if not p.exists():
            raise FileNotFoundError(p)
        imgs = [p]
    elif args.imgdir:
        d = Path(args.imgdir)
        if not d.exists():
            raise FileNotFoundError(d)
        imgs = all_images_in_dir(d)
    else:
        raise SystemExit("Bitte --image oder --imgdir angeben.")

    if not imgs:
        raise SystemExit("Keine passenden Bilddateien gefunden.")

    rows = []
    for i, img in enumerate(imgs, 1):
        print(f"[{i}/{len(imgs)}] {img}")
        try:
            row = process_one(img)
        except Exception as e:
            print("Fehler bei", img, "->", e)
            row = {"image_path": str(img), "best_en": "", "best_de": ""}
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print("Fertig:", args.out)

if __name__ == "__main__":
    main()
