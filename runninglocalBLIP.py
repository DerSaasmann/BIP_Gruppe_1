from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Device
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Device:", device)

MODEL = "Salesforce/blip-image-captioning-large"  # besser als base
processor = BlipProcessor.from_pretrained(MODEL)
model = BlipForConditionalGeneration.from_pretrained(MODEL).to(device)
model.eval()  # wichtig

# Übersetzer optional (auf CPU lassen; mps wird von pipeline nicht genutzt)
try:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de", device=-1)
except Exception as e:
    print("[Hinweis] Übersetzer nicht verfügbar:", e)
    translator = None

def describe_image(img_path, lang="de", use_prompt=False):
    image = Image.open(img_path).convert("RGB")

    # 1) Stabilste Variante: OHNE Prompt
    if not use_prompt:
        inputs = processor(images=image, return_tensors="pt").to(device)
    else:
        # 2) Falls Prompt: KURZ & ENGLISCH
        prompt = "Describe the object in the photo."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=96,
            num_beams=8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.05,
            early_stopping=True
        )

    caption_en = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    if caption_en and caption_en[0].islower():
        caption_en = caption_en[0].upper() + caption_en[1:]
    if caption_en and not caption_en.endswith("."):
        caption_en += "."

    if lang == "de" and translator is not None:
        return translator(caption_en, max_length=256)[0]["translation_text"]
    return caption_en

# Beispiel
pfad = "1-2025-0025-000-000.JPG"
print("Beschreibung:", describe_image(pfad, lang="de", use_prompt=False))
