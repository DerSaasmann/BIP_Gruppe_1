from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# --- Device-Handling (CPU / CUDA / Apple Silicon MPS) ---
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# --- BLIP (base) laden ---
# Modelle: https://huggingface.co/Salesforce/blip-image-captioning-base
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(device)

# --- Optional: EN->DE Übersetzer (offline) ---
try:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de",
                          device=0 if device == "cuda" else -1)
except Exception as e:
    print("[Hinweis] Übersetzungsmodell nicht verfügbar:", e)
    translator = None

def describe_image(img_path: str, lang="en", detailed: bool = True,
                   max_length: int = 50, num_beams: int = 8, num_return_sequences: int = 1):
    """
    Erzeugt eine Bildbeschreibung mit BLIP (base).
    - lang: 'en' oder 'de'
    - detailed: True -> promptet BLIP für ausführlichere Beschreibungen
    - max_length / num_beams: Qualität vs. Geschwindigkeit
    - num_return_sequences: mehrere Varianten zurückgeben (hier liefern wir nur die beste)
    """
    image = Image.open(img_path).convert("RGB")

    # Prompt für ausführlichere Captions (kann DE oder EN sein; BLIP versteht beides)
    prompt = "Beschreibe das Objekt auf dem Foto detailliert:" if detailed else ""

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = blip_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2
        )

    # BLIP gibt direkt Text zurück
    caption_en = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    # kleine Kosmetik
    if caption_en and caption_en[0].islower():
        caption_en = caption_en[0].upper() + caption_en[1:]
    if caption_en and not caption_en.endswith("."):
        caption_en += "."

    if lang == "de":
        if translator is None:
            print("[Hinweis] Fallback auf Englisch, da Übersetzung nicht verfügbar ist.")
            return caption_en
        out = translator(caption_en, max_length=200)
        return out[0]["translation_text"]

    return caption_en


# --- Beispielaufruf ---
if __name__ == "__main__":
    bildpfad = "1-2025-0025-000-000.JPG"
    beschreibung_de = describe_image(bildpfad, lang="de", detailed=True, max_length=64, num_beams=8)
    print("Beschreibung (DE):", beschreibung_de)
