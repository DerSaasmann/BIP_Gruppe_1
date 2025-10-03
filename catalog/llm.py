"""
Anbindung an den Textdienst in unserem Fall an OpenAi und für die Aufbereitung der Ergebnisse.

Funktionen:
- `describe_object(object_number, image_paths, lang, max_images, keywords)`: 
  * Schickt pro Objekt alle ausgewählten Ansichten in einem Auftrag.
  * Erwartet ein einzelnes JSON-Objekt als Antwort.
  * Ergänzt Standardfelder (Objektnummer, Produzentenlabel, verwendete Bildanzahl).
  * Säubert Werte (z. B. entfernt " / not available") und stellt die Klassifikation eindeutig.
  * Erzeugt bei Bedarf einen kurzen, sachlichen Absatz (falls kein Text geliefert wurde).

Interne Helfer:
- Normalisierung von Leerzeichen.
- Standardisierung von Feldwerten.
- Erzeugen eines kompakten Beschreibungstextes im Katalogstil.

Ziel:
- Ein Auftrag pro Objekt, robuste Auswertung der Antwort, konsistente Felder für nachgelagerte Formatierer.
"""

import json, re, os
from pathlib import Path
from typing import List, Literal, Dict, Any
from openai import OpenAI
from .config import MODEL, DEFAULT_KEYWORDS
from .files import encode_image_to_data_url
from catalog.prompts import build_prompt, SYSTEM_INSTRUCTION
from .labels import de_localize_value, localize_classification_for_display

client = OpenAI()

def _normalize_space(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else s

def _clean_value(v: str) -> str:
    if not isinstance(v, str): return v
    s = _normalize_space(v)
    s = re.sub(r"\s*/\s*not\s+available\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\bor\s+not\s+available\b\s*$", "", s, flags=re.I)
    if s.lower() in {"n/a","na","none"}: return "not available"
    return s

def _sanitize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict): return entry
    cls = entry.get("classification","")
    if isinstance(cls, str) and "|" in cls:
        entry["classification"] = cls.split("|",1)[0].strip()
    desc = entry.get("description",{})
    if isinstance(desc, dict):
        for k,v in list(desc.items()):
            desc[k] = _clean_value(v) if isinstance(v, str) else v
        entry["description"] = desc
    if isinstance(entry.get("description_text"), str) and entry["description_text"].strip().lower()=="not available":
        entry["description_text"] = ""
    return entry

def _ensure_description_text(entry: Dict[str, Any], lang: Literal["de","en"]) -> None:
    text = _normalize_space(entry.get("description_text",""))
    if text: return
    d = entry.get("description",{}) if isinstance(entry.get("description"), dict) else {}
    cls = entry.get("classification","Object")
    brand = _normalize_space(d.get("brand_logo",""))
    color = _normalize_space(d.get("color",""))
    housing = _normalize_space(d.get("housing_material",""))
    form = _normalize_space(d.get("form",""))
    mech = _normalize_space(d.get("mechanism","")) or _normalize_space(d.get("form_factor",""))
    keys = _normalize_space(d.get("keyboard_type","")) or _normalize_space(d.get("controls",""))
    labels = _normalize_space(d.get("labels_or_scales",""))

    if lang == "de":
        parts = [f"Diese {localize_classification_for_display(cls, 'de').lower()}"]
        if brand and brand.lower() not in {"not available","not applicable"}: parts.append(f"mit der Markierung „{brand}“")
        if color and color.lower() not in {"not available","not applicable"}: parts.append(f"in {de_localize_value(color)}")
        if housing and housing.lower() not in {"not available","not applicable"}: parts.append(f"mit einem Gehäuse aus {de_localize_value(housing)}")
        if form and form.lower() not in {"not available","not applicable"}:
            ftxt = de_localize_value(form)
            parts.append(f"und {ftxt}er Form" if not ftxt.endswith("form") else f"und {ftxt}")
        s1 = " ".join(parts) + "."
        s2_bits = []
        if keys and keys.lower() not in {"not available","not applicable"}: s2_bits.append(keys)
        if mech and mech.lower() not in {"not available","not applicable"}: s2_bits.append(de_localize_value(mech))
        s2 = ("Sie verfügt über " + " und ".join(s2_bits) + ".") if s2_bits else ""
        s3 = f"Sichtbare Beschriftungen/Skalen: {de_localize_value(labels)}." if labels and labels.lower() not in {"not available","not applicable"} else ""
        entry["description_text"] = " ".join(x for x in [s1,s2,s3] if x)
    else:
        parts = [f"This {cls.lower()}"]
        if brand and brand.lower() not in {"not available","not applicable"}: parts.append(f"bearing the marking '{brand}'")
        if color and color.lower() not in {"not available","not applicable"}: parts.append(f"in {color}")
        if housing and housing.lower() not in {"not available","not applicable"}: parts.append(f"with a {housing} housing")
        if form and form.lower() not in {"not available","not applicable"}:
            article = "an" if form[:1].lower() in "aeiou" else "a"
            parts.append(f"and {article} {form} form")
        s1 = " ".join(parts) + "."
        s2_bits = []
        if keys and keys.lower() not in {"not available","not applicable"}: s2_bits.append(keys)
        if mech and mech.lower() not in {"not available","not applicable"}: s2_bits.append(mech)
        s2 = ("It features " + " and ".join(s2_bits) + ".") if s2_bits else ""
        s3 = f"Visible labels or scales: {labels}." if labels and labels.lower() not in {"not available","not applicable"} else ""
        entry["description_text"] = " ".join(x for x in [s1,s2,s3] if x)

def describe_object(object_number: str, image_paths: List[Path], lang: Literal["de","en"], max_images: int, keywords: str = DEFAULT_KEYWORDS) -> Dict[str, Any]:
    selected = sorted(image_paths, key=lambda x: x.name.lower())[:max_images]
    content = [{"type": "text", "text": build_prompt(object_number, keywords, lang)}]
    for p in selected:
        content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(p)}})

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
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
            "_images_used": len(selected),
        }

    # Falls ausnahmsweise ein Array kommt → erstes Element nehmen
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
    data["_images_used"] = len(selected)

    data = _sanitize_entry(data)
    _ensure_description_text(data, lang)
    return data
