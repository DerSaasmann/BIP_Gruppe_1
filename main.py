"""
Entry point for the Museum Catalogue Builder.

Purpose:
- Reads command-line arguments (image root folder, destination folder, prefixes, language, limits).
- Does not contain any image processing or text generation logic.
- Delegates the entire workflow to `catalog.pipeline.run_pipeline(...)`.

Command-line arguments:
--root          Path to the folder containing the object images
--dest          Destination folder for copies of the matched images
--prefix        One or more object number prefixes (filenames must start with "1")
--lang          Output language: "de" or "en" (defaults to English if not specified)
--max-images    Maximum number of images per object to send to the model
--out-txt       Path for the TXT output file
--out-csv       Path for the long-format CSV file

Output:
- Launches the pipeline, writes TXT and CSV outputs, and displays a concise summary in the terminal.
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
