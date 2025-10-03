"""
Central settings for the project
(file types, model, default keywords, producer label).
"""

from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CASE_INSENSITIVE = True

# OpenAI model
MODEL = "gpt-4o-mini"

# Keywords that we append to every request
DEFAULT_KEYWORDS = "typewriter, communication"
