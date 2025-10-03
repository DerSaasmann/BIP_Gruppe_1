

import os, base64, csv, mimetypes, glob
from openai import OpenAI

# 1) Client aus Umgebungsvariable OPENAI_API_KEY initialisieren
client = OpenAI()  # nutzt automatisch $OPENAI_API_KEY

def bild_zu_caption(pfad: str, sprache: str = "de") -> str:
    """
    Schickt ein einzelnes Bild an das ChatGPT-Vision-Modell und
    gibt eine kurze, sachliche Beschreibung in der gewünschten Sprache zurück.
    """
    # MIME-Typ (image/jpeg, image/png ...) ermitteln
    mime, _ = mimetypes.guess_type(pfad)
    if mime is None:
        # Fallback (für z. B. .jpg/.jpeg/.png klappt guess_type normalerweise)
        mime = "image/jpeg"

    # Bild als Base64-Daten-URL vorbereiten
    with open(pfad, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    prompt = (
        f"Beschreibe dieses Bild auf {sprache}. "
        "Bleib sachlich und knapp (1–2 Sätze). "
        "Keine Spekulationen, keine Personenerkennung. "
        "Wenn Text im Bild steht, fasse ihn kurz zusammen."
    )

    # 2) Chat-Completions mit Bild-Eingabe
    # (Offizieller Weg für Vision in der Chat-API)  # docs: chat + vision
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # versteht Bilder, ist schnell & günstig
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    )

    text = response.choices[0].message.content.strip()
    return text

def haupt():
    # 3) Alle Bilddateien im Unterordner ./bilder einsammeln
    ordner = os.path.join(os.getcwd(), "bilder")
    muster = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    dateien = []
    for m in muster:
        dateien.extend(glob.glob(os.path.join(ordner, m)))

    if not dateien:
        print("Keine Bilder gefunden. Lege Dateien in den Ordner ./bilder.")
        return

    # 4) CSV vorbereiten
    ausgabe_csv = os.path.join(os.getcwd(), "captions.csv")
    with open(ausgabe_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["datei", "beschreibung"])  # Header

        # 5) Bilder der Reihe nach beschreiben
        for i, pfad in enumerate(sorted(dateien), start=1):
            try:
                caption = bild_zu_caption(pfad, sprache="de")
            except Exception as e:
                caption = f"FEHLER: {e}"
            relname = os.path.relpath(pfad, ordner)
            print(f"[{i}/{len(dateien)}] {relname}\n→ {caption}\n")
            print(f"[{i}/{len(dateien)}] {relname}: {caption}")
            writer.writerow([relname, caption])

    print("\nFertig! Datei gespeichert als captions.csv")

if __name__ == "__main__":
    haupt()
