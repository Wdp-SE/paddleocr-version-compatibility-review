"""Original student call surface; result normalization still needs runtime tests."""
import paddleocr as document_ocr


def extract(scan_path: str):
    engine = document_ocr.PaddleOCR(lang="ch")
    result = engine.ocr(scan_path)
    return result
