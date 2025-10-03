"""

Dateisystem-Helfer: Bildsuche, Gruppierung und Kopieren.

Funktionen:
- `iter_images(root)`: Findet Bilddateien unterhalb eines Wurzelpfads.
- `collect_matches_by_prefix(root, prefixes)`: Ordnet Bilder nach Inventar-Präfix (Dateinamen müssen mit Präfix beginnen).
- `copy_matched(groups, dest)`: Kopiert gefundene Dateien in den Zielordner (mit Duplikatschutz).
- `encode_image_to_data_url(path)`: Wandelt ein Bild in eine Data-URL (Base64) für die Übergabe an den Dienst.

Ziel:
- Saubere Gruppierung: ein Objekt pro Präfix.
- Nachvollziehbarkeit: Kopien der verwendeten Bilder liegen zentral vor.

"""


from pathlib import Path
from typing import Dict, List, Tuple
import base64, mimetypes, shutil
from .config import EXTS, CASE_INSENSITIVE

def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p

def collect_matches_by_prefix(root: Path, prefixes: List[str]) -> Tuple[Dict[str, List[Path]], Dict[str, str]]:
    pref_norm = { (p.lower().strip() if CASE_INSENSITIVE else p.strip()): p for p in prefixes }
    groups = {k: [] for k in pref_norm}
    for p in iter_images(root):
        name = p.name.lower() if CASE_INSENSITIVE else p.name
        for k in pref_norm:
            if name.startswith(k):
                groups[k].append(p)
                break
    groups = {k: sorted(v, key=lambda x: x.name.lower()) for k, v in groups.items() if v}
    return groups, pref_norm

def encode_image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None: mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def copy_matched(groups: Dict[str, List[Path]], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for paths in groups.values():
        for src in paths:
            dst = dest / src.name
            j, cand = 1, dst
            while cand.exists():
                cand = dest / f"{src.stem}__dup{j}{src.suffix}"
                j += 1
            shutil.copy2(src, cand)
