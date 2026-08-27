"""Fetches the "Attention Is All You Need" paper from arXiv and renders it
to one PIL.Image per page -- used for the sample doc in scratch.py instead
of committing page images to the repo."""
import os
import urllib.request

import pymupdf
from PIL import Image

_ARXIV_PDF = "https://arxiv.org/pdf/1706.03762"
_CACHE_PATH = os.path.expanduser("~/.cache/find_and_interpret/attention_is_all_you_need.pdf")


def get_example_pages():
    """Returns [(name, PIL.Image), ...] for every page of the paper, sorted."""
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    if not os.path.exists(_CACHE_PATH):
        urllib.request.urlretrieve(_ARXIV_PDF, _CACHE_PATH)

    doc = pymupdf.open(_CACHE_PATH)
    pages = []
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append((f"{i:02d}.png", img))
    return pages
