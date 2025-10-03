"""
Output formatting for text (Terminal/TXT) and CSV format.

Functions:
- make_text_block(entry, lang):
  Creates a compact, human-readable text block.
  Hides "not available"/"not applicable" in the readable output and uses localized field names.

- explode_to_long_rows(entry, lang):
  Converts a dataset into multiple CSV rows (one row per field),
  suitable for flexible further processing in spreadsheet software.

Goal:
- Factual text output in catalogue style.
- CSV structure that can be easily filtered, sorted, and further processed.
"""


from typing import Dict, Any, List, Literal
from .labels import HEADINGS, FIELD_LABELS, FIELD_ORDER, de_localize_value, localize_classification_for_display

def _norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else s

def make_text_block(entry: Dict[str, Any], lang: Literal["de","en"]) -> str:
    H = HEADINGS[lang]
    FL = FIELD_LABELS[lang]
    obj = entry.get("object_number", "not available")
    cls = localize_classification_for_display(entry.get("classification", "not available"), lang)
    conf = entry.get("confidence", "not available")
    desc = entry.get("description", {}) if isinstance(entry.get("description"), dict) else {}
    used = entry.get("_images_used")

    lines = [f"{H['objnum']}: {obj}", f"{H['classif']}: {cls}"]
    if isinstance(used, int):
        lines.append(f"{H['images_used']}: {used}")
    for key in FIELD_ORDER:
        val = _norm(desc.get(key, ""))
        if not val or val.lower() in {"not available","not applicable"}:
            continue
        if lang == "de":
            val = de_localize_value(val)
        lines.append(f"{FL[key]}: {val}")
    lines.append(f"{H['confidence']}: {conf if conf not in (None,'') else 'not available'}")
    lines.append(f"{H['gen_by']}: AI")
    lines.append(f"{H['desc']}: " + (entry.get("description_text") or ""))
    return "\n".join(lines)

def explode_to_long_rows(entry: Dict[str, Any], lang: Literal["de","en"]) -> List[Dict[str, str]]:
    FL = FIELD_LABELS[lang]
    H  = HEADINGS[lang]
    obj = entry.get("object_number","")
    d = entry.get("description",{}) if isinstance(entry.get("description"), dict) else {}

    def L(val: str) -> str:
        if lang == "de" and isinstance(val, str):
            return de_localize_value(val)
        return val

    rows = [
        {"object_number": obj, "field": H["classif"],   "value": localize_classification_for_display(entry.get("classification",""), lang)},
        {"object_number": "",  "field": H["confidence"], "value": entry.get("confidence","")},
    ]
    for key in FIELD_ORDER:
        val = d.get(key,"")
        rows.append({"object_number": "", "field": FL[key], "value": L(val) if isinstance(val, str) else val})
    rows += [
        {"object_number": "", "field": H["desc"],      "value": entry.get("description_text","")},
        {"object_number": "", "field": "keywords",     "value": entry.get("keywords","")},
        {"object_number": "", "field": H["gen_by"],    "value": entry.get("generated_by","AI")},
    ]
    # Whitespace normieren
    for r in rows:
        for k,v in r.items():
            if isinstance(v, str): r[k] = _norm(v)
    return rows
