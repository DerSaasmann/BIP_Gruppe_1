"""
Overall processing workflow.

Steps:
1) Identify image files for each inventory prefix and group them into objects.
2) Copy found files to the destination folder (with duplicate protection).
3) For each object, create a request with all associated images and evaluate the result.
4) Output results: terminal (short overview), TXT (readable description), and CSV.

Function:
- run_pipeline(root_dir, dest_dir, prefixes, out_txt, out_csv, lang, max_images)

Goal:
- Clear, reproducible workflow without duplicated logic.
- Clean separation of file access, model calls, and output formatting.
"""

from pathlib import Path
from typing import List, Literal
import csv

from catalog.files import collect_matches_by_prefix, copy_matched
from catalog.llm import describe_object
from catalog.formatter import make_text_block, explode_to_long_rows
from catalog.labels import HEADINGS

def run_pipeline(
    root_dir: Path,
    dest_dir: Path,
    prefixes: List[str],
    out_txt: Path,
    out_csv: Path,
    lang: Literal["de","en"],
    max_images: int
):
    groups, pref_map = collect_matches_by_prefix(root_dir, prefixes)
    total_images = sum(len(v) for v in groups.values())
    H = HEADINGS[lang]
    print(H["matched"].format(files=total_images, objs=len(groups)) + "\n")

    copy_matched(groups, dest_dir)

    results = []
    for i, (pref_norm, paths) in enumerate(groups.items(), start=1):
        obj_no = pref_map[pref_norm]
        try:
            entry = describe_object(obj_no, paths, lang=lang, max_images=max_images)
        except Exception as e:
            entry = {"object_number": obj_no, "error": str(e), "generated_by": "AI"}
        results.append(entry)

        print(H["record"].format(i=i, n=len(groups)))
        print(make_text_block(entry, lang))
        print()

    # TXT
    with open(out_txt, "w", encoding="utf-8") as ftxt:
        for e in results:
            ftxt.write(make_text_block(e, lang)); ftxt.write("\n\n")

    # CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object_number","field","value"], quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        for e in results:
            for r in explode_to_long_rows(e, lang):
                writer.writerow(r)

    print(H["done"])
    print(H["txt_saved"].format(p=out_txt.resolve()))
    print(H["csv_saved"].format(p=out_csv.resolve()))
    print(H["copied"].format(p=dest_dir.resolve()))
