"""Original student adapter that assumes the 2.9.1 nested OCR result shape."""
from paddleocr import PaddleOCR as TextEngine


def normalize_text(scan_path: str) -> list[dict]:
    engine = TextEngine(lang="ch")
    result = engine.ocr(scan_path)
    normalized = []
    for page in result:
        for line in page:
            normalized.append({"text": line[1][0], "confidence": line[1][1]})
    return normalized
