from .ocr_client import recognize
from .normalizer import normalize_legacy
from .consumer import to_json


def process(path):
    raw = recognize(path)
    lines = normalize_legacy(raw)
    return to_json(lines)
