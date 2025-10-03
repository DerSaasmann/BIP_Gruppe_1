"""
Erzeugt die Prompts mit klaren Regeln gegen Spekulation.

Enthält:
- build_prompt(object_number, keywords, lang):
  Baut den Eingabetext für das Modell (DE/EN) mit festen Vorgaben:
  • Genau eine Klassifikation: TYPEWRITER oder COMMUNICATION TOOL (optional „(probably)”).
  • Nur sichtbare Merkmale benennen; sonst „not available“.
  • Nicht relevante Felder auf „not applicable“ setzen.
  • Sachlicher Katalogstil, kurze und überprüfbare Formulierungen.

- SYSTEM_INSTRUCTION:
  Zusätzliche Anweisung: genau EIN JSON-Objekt, keine Schrägstriche in Feldern,
  klare Werte („not available“ / „not applicable“).

Ziel:
Reproduzierbare, prüfbare Katalogeinträge ohne Mutmaßungen.

"""

from typing import Literal

def build_prompt(object_number: str, keywords: str, lang: Literal["de","en"]) -> str:
    if lang == "de":
        desc_line = ('Schreibe 2–4 vollständige Sätze auf Deutsch im Katalogstil. '
                     'Kein "not available" in diesem Absatz. Fasse nur klar Sichtbares zusammen '
                     '(verwende "(probably)" bei Bedarf).')
        value_lang_rule = "Use German terms for generic values (materials, colours, mechanism, form). Keep brand names and keyboard layouts (e.g., QWERTY) as-is."
    else:
        desc_line = ("Write 2–4 complete sentences in English, catalogue style. "
                     "Do NOT include 'not available' in this paragraph. Summarize only what is visible "
                     "(use '(probably)' where needed).")
        value_lang_rule = "Use English terms for generic values."

    return f"""
You are a curator at the Technisches Museum Berlin.
Create ONE catalogue entry using ALL provided images of the SAME object (multiple views/angles).

Rules to avoid hallucinations:
- Choose exactly ONE classification: TYPEWRITER or COMMUNICATION TOOL.
  - If slightly uncertain, append " (probably)" to the chosen class, e.g. "TYPEWRITER (probably)".
  - Do NOT output both classes or use a pipe character.
- For each field, return EITHER a single concise value OR exactly "not available" OR exactly "not applicable".
  - NEVER use slashes like "value / not available" or "A/B".
  - Use "(probably) ..." when the feature is visible but not fully clear.
- Fill ONLY the field group that matches your classification:
  - TYPEWRITER: keyboard_type, housing_material, color, form, brand_logo (only if legible), mechanism.
    Set form_factor, controls, connectors, labels_or_scales, material_specific to "not applicable".
  - COMMUNICATION TOOL: form_factor, controls, connectors, labels_or_scales, material_specific.
    Set keyboard_type, housing_material, color, form, brand_logo, mechanism to "not applicable".
- If a feature is truly not visible, use "not available".
- Museum catalogue style, objective and factual.
- {value_lang_rule}
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
  "description_text": "{desc_line}",
  "confidence": 0.0,
  "generated_by": "AI",
  "keywords": "{keywords}"
}}

If ALL images are unusable (too blurry/incomplete/unclear), output exactly: "Image not usable"
"""

SYSTEM_INSTRUCTION = (
    "Return exactly one JSON object as specified. Choose ONE classification only. "
    "No slashes in field values. Use 'not applicable' for the irrelevant field group."
)
