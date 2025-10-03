"""
Einstiegspunkt für den Museum Catalogue Builder.

Aufgabe:
- Liest Kommandozeilenargumente (Bildwurzelordner, Zielordner, Präfixe, Sprache, Limits).
- Übernimmt keine Logik zur Bildverarbeitung oder Textgenerierung.
- Delegiert den gesamten Ablauf an `catalog.pipeline.run_pipeline(...)`.

Eingaben in Komandozeile:
--root          Pfad mit den Objektbildern
--dest          Zielordner für Kopien der gefundenen Bilder
--prefix        Eine oder mehrere Inventar-Präfixe (Dateinamen müssen mit "1" beginnen)
--lang          Ausgabesprache: "de" oder "en" wenn keine auswahl automatisch englisch
--max-images    Maximalzahl Bilder pro Objekt, die an das Modell übergeben werden
--out-txt       Pfad für die txt datei
--out-csv       Pfad für die Long-CSV 

Ausgabe:
- Startet die Pipeline, schreibt TXT und CSV und zeigt eine knappe Zusammenfassung im Terminal.
"""
import argparse
from pathlib import Path
from catalog.pipeline import run_pipeline



def build_parser():
    p = argparse.ArgumentParser(description="Museum Catalogue Builder (Typewriters & Communication)")
    p.add_argument("--root", required=True, help="Root folder with images")
    p.add_argument("--dest", required=True, help="Destination folder for copied matched images")
    p.add_argument("--prefix", nargs="+", required=True, help="Inventory prefixes (filenames must start with these)")
    p.add_argument("--lang", choices=["de","en"], default="en", help="Output language")
    p.add_argument("--max-images", type=int, default=12, help="Max images per object included in the prompt")
    p.add_argument("--out-txt", default="catalog_descriptions.txt", help="TXT output path")
    p.add_argument("--out-csv", default="descriptions_long.csv", help="CSV (long) output path")
    return p

def main():
    args = build_parser().parse_args()
    run_pipeline(
        root_dir=Path(args.root),
        dest_dir=Path(args.dest),
        prefixes=args.prefix,
        out_txt=Path(args.out_txt),
        out_csv=Path(args.out_csv),
        lang=args.lang,
        max_images=args.max_images,
    )

if __name__ == "__main__":
    main()
