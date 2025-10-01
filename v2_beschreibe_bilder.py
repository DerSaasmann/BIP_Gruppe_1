
# -*- coding: utf-8 -*-
import base64
import csv
import json
import mimetypes
import re
import shutil
from pathlib import Path

from openai import OpenAI

# -------------------- Settings --------------------
ROOT_DIR = Path("/Users/davidassmann/Desktop/GuI/Objektbilder")
PREFIXES = ["1-1997-0007", "1-1997-0011"]  # adjust as needed
DEST_DIR = Path("/Users/davidassmann/Desktop/GuI/GefilterteteBilder")

OUT_CSV_LONG = Path("descriptions_long.csv")
OUT_TXT      = Path("catalog_descriptions.txt")

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CASE_INSENSITIVE = True
ONLY_PREFIX_START = True

# OpenAI Client (requires OPENAI_API_KEY in env)
client = OpenAI()
MODEL = "gpt-4o-mini"

# -------------------- Fields --------------------
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

# -------------------- Helpers --------------------
def find_images_by_prefix(root: Path, prefixes, exts, case_insensitive=True, only_prefix_start=True):
    """Return image file Paths under root that match any of the prefixes."""
    prefixes_norm = [p.lower().strip() if case_insensitive else p.strip() for p in prefixes]
    results = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        key = p.name.lower() if case_insensitive else p.name
        for pref in prefixes_norm:
            if (only_prefix_start and key.startswith(pref)) or (not only_prefix_start and pref in key):
                results.append(p)
                break
    return results

def normalize_space(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

# ---------- Prompt (strict, no slashes; one class only; irrelevant group = not applicable)
def build_prompt(object_number: str, keywords: str) -> str:
    """
    Stricter prompt:
    - One class only (TYPEWRITER or COMMUNICATION TOOL; optional ' (probably)' suffix).
    - For each field: either a single value string OR exactly "not available" OR "not applicable" (no slashes).
    - Fill only the relevant field group; set the other group to "not applicable".
    - description_text must be 2–4 sentences and must NOT contain 'not available'.
    """
    return f"""
You are a curator at the Technisches Museum Berlin.
Create a catalogue entry for the given object (typewriters and communication technology) from the provided image.

Rules to avoid hallucinations:
- Choose exactly ONE classification: TYPEWRITER or COMMUNICATION TOOL.
  - If slightly uncertain, append " (probably)" directly to the chosen class, e.g. "TYPEWRITER (probably)".
  - Do NOT output both classes or use a pipe character.
- For each field, return EITHER a single concise value OR exactly "not available" OR exactly "not applicable".
  - NEVER use slashes like "value / not available" or "A/B".
  - Use "(probably) ..." when the feature is visible but not fully clear.
- Fill ONLY the field group that matches your classification:
  - If TYPEWRITER: fill keyboard_type, housing_material, color, form, brand_logo (only if legible), mechanism.
    Set form_factor, controls, connectors, labels_or_scales, material_specific to "not applicable".
  - If COMMUNICATION TOOL: fill form_factor, controls, connectors, labels_or_scales, material_specific.
    Set keyboard_type, housing_material, color, form, brand_logo, mechanism to "not applicable".
- If a feature is truly not visible, use "not available".
- Museum catalogue style, objective and factual. Avoid "maybe" or "seems".
- If >90% of the description is based on visible evidence, include "confidence" as a float 0.0–1.0.
- Always include "generated_by": "AI".

Return exactly ONE JSON object (not an array) with this schema:
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
  "description_text": "Write 2–4 complete sentences in English. Do NOT write 'not available' in this paragraph. Summarize only what is visible (use '(probably)' where needed).",
  "confidence": 0.0,
  "generated_by": "AI",
  "keywords": "{keywords}"
}}

If the image is unusable (too blurry/incomplete/unclear), output exactly: "Image not usable"
"""

# ---------- Output cleaners
def clean_value(v: str) -> str:
    """Remove patterns like 'value / not available', normalize whitespace."""
    if not isinstance(v, str):
        return v
    s = normalize_space(v)
    # Remove ' / not available' tails (any case)
    s = re.sub(r"\s*/\s*not\s+available\s*$", "", s, flags=re.I)
    # Remove ' or not available' tails (any case)
    s = re.sub(r"\s*\bor\s+not\s+available\b\s*$", "", s, flags=re.I)
    # Normalize common NA tokens
    if s.lower() in {"n/a", "na", "none"}:
        return "not available"
    return normalize_space(s)

def sanitize_entry(entry: dict) -> dict:
    """Force single class; clean 'value / not available' artifacts; tidy description fields."""
    if not isinstance(entry, dict):
        return entry

    # Fix classification like "TYPEWRITER | COMMUNICATION TOOL" -> take first token before '|'
    cls = entry.get("classification", "")
    if isinstance(cls, str) and "|" in cls:
        cls = cls.split("|", 1)[0].strip()
        entry["classification"] = cls

    # Clean description fields
    desc = entry.get("description", {})
    if isinstance(desc, dict):
        for k, v in list(desc.items()):
            desc[k] = clean_value(v) if isinstance(v, str) else v
        entry["description"] = desc

    # Avoid literal 'not available' as the paragraph
    if isinstance(entry.get("description_text"), str) and entry["description_text"].strip().lower() == "not available":
        entry["description_text"] = ""

    return entry

def ensure_description_text(entry: dict) -> None:
    """
    If the model did not supply a good 'description_text', build a concise, factual paragraph from fields.
    Never inserts 'not available' into the paragraph.
    """
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

# ---------- Model call
def ai_describe(object_number: str, keywords: str, image_path: Path) -> dict:
    """Send image + prompt to the model, return parsed dict or an 'Image not usable' record."""
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    content = [
        {"type": "text", "text": build_prompt(object_number, keywords)},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    messages = [
        {"role": "system", "content": "Return exactly one JSON object as specified. Choose ONE classification only. Do not use slashes in field values. Use 'not applicable' for the irrelevant field group."},
        {"role": "user", "content": content},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,                # a bit more text but still stable
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = response.choices[0].message.content.strip()

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

    # Rarely, models return a list; take first element
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

    # sanitize + ensure a paragraph
    data = sanitize_entry(data)
    ensure_description_text(data)
    return data

# ---------- Rendering & CSV ----------
def make_text_block(entry: dict) -> str:
    """Structured field list + free-text description for terminal and TXT."""
    obj = entry.get("object_number", "not available")
    cls = entry.get("classification", "not available")
    conf = entry.get("confidence", "not available")
    desc = entry.get("description", {}) if isinstance(entry.get("description"), dict) else {}

    lines = []
    lines.append(f"Object number: {obj}")
    lines.append(f"Classification: {cls}")
    for key, label in FIELD_ORDER:
        val = normalize_space(desc.get(key, ""))
        # hide both 'not available' and 'not applicable' from human-readable output
        if val and val.lower() not in {"not available", "not applicable"}:
            lines.append(f"{label}: {val}")
    lines.append(f"Confidence: {conf if conf not in (None, '') else 'not available'}")
    lines.append("Generated by: AI")
    lines.append("Description: " + (entry.get("description_text") or ""))
    return "\n".join(lines)

def explode_to_long_rows(entry: dict):
    """
    Create long CSV rows where the object_number is present only on the first row per object.
    Order:
      classification, confidence, description fields (FIELD_ORDER), description_text, keywords, generated_by, error
    """
    obj = entry.get("object_number", "")
    d = entry.get("description", {}) if isinstance(entry.get("description"), dict) else {}
    rows = []
    rows.append({"object_number": obj, "field": "classification", "value": entry.get("classification","")})
    rows.append({"object_number": "", "field": "confidence", "value": entry.get("confidence","")})
    for k,_ in FIELD_ORDER:
        rows.append({"object_number": "", "field": k, "value": d.get(k, "")})
    rows.append({"object_number": "", "field": "description_text", "value": entry.get("description_text","")})
    rows.append({"object_number": "", "field": "keywords", "value": entry.get("keywords","")})
    rows.append({"object_number": "", "field": "generated_by", "value": entry.get("generated_by","AI")})
    if entry.get("error"):
        rows.append({"object_number": "", "field": "error", "value": entry.get("error","")})
    # normalize whitespace across rows
    for r in rows:
        for k, v in r.items():
            if isinstance(v, str):
                r[k] = normalize_space(v)
    return rows

# -------------------- Main --------------------
def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    hits = find_images_by_prefix(
        ROOT_DIR, PREFIXES, EXTS,
        case_insensitive=CASE_INSENSITIVE,
        only_prefix_start=ONLY_PREFIX_START
    )
    print(f"Found {len(hits)} file(s)\n")

    results = []
    for i, src in enumerate(hits, start=1):
        # copy image to DEST_DIR (deduplicate name if needed)
        dst = DEST_DIR / src.name
        j, dst_candidate = 1, dst
        while dst_candidate.exists():
            dst_candidate = DEST_DIR / f"{src.stem}__dup{j}{src.suffix}"
            j += 1
        shutil.copy2(src, dst_candidate)

        obj_num = src.stem
        keywords = "typewriter, communication"
        try:
            entry = ai_describe(obj_num, keywords, src)
        except Exception as e:
            entry = {"object_number": obj_num, "error": str(e), "generated_by": "AI"}
        results.append(entry)

        # Terminal block
        print(f"——— Record {i}/{len(hits)} ———")
        print(make_text_block(entry))
        print()

    # TXT: all records
    with open(OUT_TXT, "w", encoding="utf-8") as ftxt:
        for entry in results:
            ftxt.write(make_text_block(entry))
            ftxt.write("\n\n")

    # LONG CSV: object_number only on first row per object
    long_cols = ["object_number","field","value"]
    with open(OUT_CSV_LONG, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_cols, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        for entry in results:
            for row in explode_to_long_rows(entry):
                writer.writerow(row)

    print("Done.")
    print(f"- Text descriptions: {OUT_TXT.resolve()}")
    print(f"- CSV (long):       {OUT_CSV_LONG.resolve()}")

if __name__ == "__main__":
    main()
