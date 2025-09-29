from pathlib import Path
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline

# --- Modelle laden (einmal pro Session) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de",
                      device=0 if device == "cuda" else -1)

def describe_image(img_path: str, lang="en"):
    """Liest ein Bild ein und erzeugt eine Beschreibung (EN oder DE)."""
    image = Image.open(img_path).convert("RGB")
    pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output_ids = caption_model.generate(pixel_values, max_length=20, num_beams=4)

    caption_en = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().capitalize()
    if not caption_en.endswith("."):
        caption_en += "."

    if lang == "de":
        out = translator(caption_en, max_length=200)
        return out[0]["translation_text"]
    return caption_en


# --- Beispielaufruf ---
beschreibung = describe_image("1-2025-0025-000-000.JPG", lang="de")
print("Beschreibung:", beschreibung)
