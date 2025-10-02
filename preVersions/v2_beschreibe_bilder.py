# -*- coding: utf-8 -*-
import base64
import csv
import json
import mimetypes
import re
from pathlib import Path

from openai import OpenAI

# -------------------- Settings --------------------
DEST_DIR = Path("/Users/davidassmann/Desktop/GuI/GefilterteteBilder")  # enthält alle Ansichten EINES Objekts
OUT_CSV_LONG = Path("descriptions_long.csv")
OUT_TXT      = Path("catalog_descriptions.txt")

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_IMAGES = 6                     # max. Anzahl Bilder, die in EINER Anfrage mitgeschickt werden
MODEL = "gpt-4o-mini"

# Optional: Objekt-Nr. manuell setzen; wenn None, wird sie aus Dateinamen abgeleitet (ersten 3 Segmente)
OBJECT_NUMBER = None
KEYWORDS = "typewriter, communication"

# OpenAI Client (OPENAI_API_KEY muss als Umgebungsvariable gesetzt sein)
client = OpenAI()

# -------------------- Felder & Utilities --------------------
FIELD_ORDER = [
    ("brand_logo", "Brand/logo"),
    ("keyboard_type", "Keyboard"),
    ("housing_material", "Housing material"),
    ("color", "Color"),
    ("form", "Form"),
    ("mechanism", "Mechanism"),
    ("form_factor", "Form factor"),
    ("controls", "Controls"),
    ("connectors", "Connectors"),
    ("labels_or_scales", "Labels/scales"),
    ("material_specific", "Material (specific)"),
]

def normalize_space(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def clean_value(v: str) -> str:
    if not isinstance(v, str):
        return v
    s = normalize_space(v)
    s = re.sub(r"\s*/\s*not\s+available\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\bor\s+not\s+available\b\s*$", "", s, flags=re.I)
    if s.lower() in {"n/a", "na", "none"}:
        return "not available"
    return normalize_space(s)

def sanitize_entry(entry: dict) -> dict:
    if not isinstance(entry, dict):
        return entry
    cls = entry.get("classification", "")
    if isinstance(cls, str) and "|" in cls:
        cls = cls.split("|", 1)[0].strip()
        entry["classification"] = cls
    desc = entry.get("description", {})
    if isinstance(desc, dict):
        for k, v in list(desc.items()):
            desc[k] = clean_value(v) if isinstance(v, str) else v
        entry["description"] = desc
    if isinstance(entry.get("description_text"), str) and entry["description_text"].strip().lower() == "not available":
        entry["description_text"] = ""
    return entry

def ensure_description_text(entry: dict) -> None:
    text = normalize_space(entry.get("description_text", ""))
    if text:
        return
    desc = entry.get("description", {}) if isinstance(entry.get("description"), dict) else {}
    cls = entry.get("classification", "Object")
    brand = normalize_space(desc.get("brand_logo", ""))
    color = normalize_space(desc.get("color", ""))
    housing = normalize_space(desc.get("housing_material", ""))
    form = normalize_space(desc.get("form", ""))
    mech = normalize_space(desc.get("mechanism", "")) or normalize_space(desc.get("form_factor", ""))
    keys = normalize_space(desc.get("keyboard_type", "")) or normalize_space(desc.get("controls", ""))
    labels = normalize_space(desc.get("labels_or_scales", ""))

    parts = [f"This {cls.lower()}"]
    if brand and brand.lower() not in {"not available", "not applicable"}:
        parts.append(f"bearing the marking '{brand}'")
    if color and color.lower() not in {"not available", "not applicable"}:
        parts.append(f"in {color}")
    if housing and housing.lower() not in {"not available", "not applicable"}:
        parts.append(f"with a {housing} housing")
    if form and form.lower() not in {"not available", "not applicable"}:
        article = "an" if form[:1].lower() in "aeiou" else "a"
        parts.append(f"and {article} {form} form")
    sentence1 = " ".join(parts) + "."

    sentence2_bits = []
    if keys and keys.lower() not in {"not available", "not applicable"}:
        sentence2_bits.append(keys)
    if mech and mech.lower() not in {"not available", "not applicable"}:
        sentence2_bits.append(mech)
    sentence2 = ("It features " + " and ".join(sentence2_bits) + ".") if sentence2_bits else ""

    sentence3 = f"Visible labels or scales: {labels}." if labels and labels.lower() not in {"not available", "not applicable"} else ""
    entry["description_text"] = " ".join(x for x in [sentence1, sentence2, sentence3] if x)

def build_prompt(object_number: str, keywords: str) -> str:
    return f"""
You are a curator at the Technisches Museum Berlin.
You will receive MULTIPLE images of the SAME object. Consider ALL images together before answering.

Rules to avoid hallucinations:
- Choose exactly ONE classification: TYPEWRITER or COMMUNICATION TOOL.
  - If slightly uncertain, append " (probably)" directly to the chosen class.
- For each field, return EITHER a single concise value OR exactly "not available" OR exactly "not applicable".
  - NEVER use slashes ("/") or "A/B" in any value.
- Fill ONLY the relevant field group:
  - TYPEWRITER: fill keyboard_type, housing_material, color, form, brand_logo (if legible), mechanism.
    Set form_factor, controls, connectors, labels_or_scales, material_specific to "not applicable".
  - COMMUNICATION TOOL: fill form_factor, controls, connectors, labels_or_scales, material_specific.
    Set keyboard_type, housing_material, color, form, brand_logo, mechanism to "not applicable".
- If a feature is not visible, use "not available".
- Museum catalogue style: objective and factual. Avoid "maybe" or "seems".
- Include "confidence" (0.0–1.0) if >90% is based on visible evidence.
- Always include "generated_by": "AI".

Return exactly ONE JSON object with this schema:
{{
  "object_number": "{object_number}",
  "classification": "TYPEWRITER" or "COMMUNICATION TOOL" (optionally with " (probably)"),
  "description": {{
    "keyboard_type": "... or not available or not applicable",
    "housing_material": "... or not available or not applicable",
    "color": "... or not available or not applicable",
    "form": "... or not available or not applicable",
    "brand_logo": "... or not available or not applicable",
    "mechanism": "... or not available or not applicable",
    "form_factor": "... or not available or not applicable",
    "controls": "... or not available or not applicable",
    "connectors": "... or not available or not applicable",
    "labels_or_scales": "... or not available or not applicable",
    "material_specific": "... or not available or not applicable"
  }},
  "description_text": "Write 2–4 sentences in English. Do NOT write 'not available' in this paragraph.",
  "confidence": 0.0,
  "generated_by": "AI",
  "keywords": "{keywords}"
}}

If the images are unusable, return exactly: "Image not usable"
"""

def make_text_block(entry: dict) -> str:
    obj = entry.get("object_number", "not available")
    cls = entry.get("classification", "not available")
    conf = entry.get("confidence", "not available")
    desc = entry.get("description", {}) if isinstance(entry.get("description"), dict) else {}
    lines = [f"Object number: {obj}", f"Classification: {cls}"]
    for key, label in FIELD_ORDER:
        val = normalize_space(desc.get(key, ""))
        if val and val.lower() not in {"not available", "not applicable"}:
            lines.append(f"{label}: {val}")
    lines.append(f"Confidence: {conf if conf not in (None, '') else 'not available'}")
    lines.append("Generated by: AI")
    lines.append("Description: " + (entry.get("description_text") or ""))
    return "\n".join(lines)

def explode_to_long_rows(entry: dict):
    obj = entry.get("object_number", "")
    d = entry.get("description", {}) if isinstance(entry.get("description"), dict) else {}
    rows = [{"object_number": obj, "field": "classification", "value": entry.get("classification","")}]
    rows.append({"object_number": "", "field": "confidence", "value": entry.get("confidence","")})
    for k,_ in FIELD_ORDER:
        rows.append({"object_number": "", "field": k, "value": d.get(k, "")})
    rows.append({"object_number": "", "field": "description_text", "value": entry.get("description_text","")})
    rows.append({"object_number": "", "field": "keywords", "value": entry.get("keywords","")})
    rows.append({"object_number": "", "field": "generated_by", "value": entry.get("generated_by","AI")})
    if entry.get("error"):
        rows.append({"object_number": "", "field": "error", "value": entry.get("error","")})
    for r in rows:
        for k, v in r.items():
            if isinstance(v, str):
                r[k] = normalize_space(v)
    return rows

# -------------------- Bilder sammeln & Objekt-ID bestimmen --------------------
def list_images_in_dir(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    # Größte zuerst (oft bessere Qualität)
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files

def derive_object_number_from_files(files: list[Path]) -> str:
    if OBJECT_NUMBER:
        return OBJECT_NUMBER
    if not files:
        return "unknown-object"
    # Heuristik: erste Datei nehmen und die ersten 3 Segmente vor dem ersten Bildsuffix
    stem = files[0].stem
    parts = stem.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return stem

# -------------------- Ein einziger Multi-Image-Call --------------------
def ai_describe_multi(object_number: str, keywords: str, image_paths: list[Path]) -> dict:
    # nur die größten MAX_IMAGES verwenden
    image_paths = image_paths[:MAX_IMAGES]
    content = [{"type": "text", "text": build_prompt(object_number, keywords)}]

    for p in image_paths:
        mime, _ = mimetypes.guess_type(p)
        if mime is None:
            mime = "image/jpeg"
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    messages = [
        {"role": "system", "content": "Return exactly one JSON object as specified. Choose ONE classification only. No slashes in values. Use 'not applicable' for the irrelevant field group."},
        {"role": "user", "content": content},
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = resp.choices[0].message.content.strip()

    if raw == "Image not usable":
        return {
            "object_number": object_number,
            "classification": "not available",
            "description": "Image not usable",
            "description_text": "Image not usable",
            "confidence": "",
            "generated_by": "AI",
            "keywords": keywords,
        }

    if raw.startswith("["):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and arr:
                raw = json.dumps(arr[0])
        except Exception:
            pass

    data = json.loads(raw)
    data.setdefault("object_number", object_number)
    data.setdefault("generated_by", "AI")
    data.setdefault("keywords", keywords)

    data = sanitize_entry(data)
    ensure_description_text(data)
    return data

# -------------------- Main --------------------
def main():
    images = list_images_in_dir(DEST_DIR)
    if not images:
        print(f"No images found in {DEST_DIR}")
        return

    obj_num = derive_object_number_from_files(images)
    print(f"Using {len(images[:MAX_IMAGES])} image(s) for object: {obj_num}")

    try:
        entry = ai_describe_multi(obj_num, KEYWORDS, images)
    except Exception as e:
        entry = {"object_number": obj_num, "error": str(e), "generated_by": "AI"}

    # Terminal-Ausgabe (ein einziger Block)
    if "error" in entry:
        print(f"[ERROR] {entry['error']}")
    print("——— Result ———")
    print(make_text_block(entry))
    print()

    # TXT schreiben (ein Datensatz)
    with open(OUT_TXT, "w", encoding="utf-8") as ftxt:
        ftxt.write(make_text_block(entry))
        if "error" in entry:
            ftxt.write(f"\nError: {entry['error']}")
        ftxt.write("\n")

    # Long-CSV (ein Objekt => mehrere Zeilen)
    long_cols = ["object_number","field","value"]
    with open(OUT_CSV_LONG, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_cols, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        for row in explode_to_long_rows(entry):
            writer.writerow(row)

    print("Done.")
    print(f"- Text descriptions: {OUT_TXT.resolve()}")
    print(f"- CSV (long):       {OUT_CSV_LONG.resolve()}")

if __name__ == "__main__":
    main()
