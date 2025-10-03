from pathlib import Path
from catalog.pipeline import run_pipeline

run_pipeline(
    root_dir=Path("/Users/davidassmann/Desktop/GuI/Objektbilder"),
    dest_dir=Path("/Users/davidassmann/Desktop/GuI/GefilterteteBilder"),
    prefixes=["1-1997-0007", "1-1997-0011", "1-2024-0062"],
    out_txt=Path("catalog_descriptions.txt"),
    out_csv=Path("descriptions_long.csv"),
    lang="de",
    max_images=8,
)