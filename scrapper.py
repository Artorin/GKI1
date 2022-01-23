from jmd_imagescraper.core import * # dont't worry, it's designed to work with import *
from pathlib import Path

root = Path().cwd()/"images"

duckduckgo_search(root, "taylor swift", "taylor swift", max_results=200)