"""Submitted example is statically reviewed, never executed by the service."""
from paddleocr import PaddleOCR


def recognize(path):
    engine = PaddleOCR(lang='ch')
    return engine.ocr(path)
