"""

Zentrale Einstellungen für das Projekt 
(Dateitypen, Modell, Standard-Wörter, Produzentenlabel).

"""

from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CASE_INSENSITIVE = True

# OpenAI Modell
MODEL = "gpt-4o-mini"

# Schlüsselwörter, die wir an jede Anfrage hängen
DEFAULT_KEYWORDS = "typewriter, communication"
