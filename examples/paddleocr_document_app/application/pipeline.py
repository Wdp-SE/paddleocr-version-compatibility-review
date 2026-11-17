from .ocr_client import recognize, recognize_configured
from .normalizer import normalize_legacy, normalize_target
from .consumer import to_json


def process(path):
    raw = recognize(path)
    lines = normalize_legacy(raw)
    return to_json(lines, page_count=len(raw))


def process_configured(path, *, config_path=None, drop_score=None, engine_factory=None, provenance=None):
    raw = recognize_configured(path, config_path=config_path, drop_score=drop_score,
                               engine_factory=engine_factory, provenance=provenance)
    lines = normalize_legacy(raw)
    return to_json(lines, page_count=len(raw))


def process_target(path):
    """Explicit PaddleOCR 3.0 result migration; the review baseline stays legacy."""
    raw = recognize(path)
    lines = normalize_target(raw)
    return to_json(lines, page_count=len(raw))


def process_target_configured(path, *, config_path=None, drop_score=None, engine_factory=None, provenance=None):
    """Run the target adapter through the original, controlled engine factory."""
    raw = recognize_configured(path, config_path=config_path, drop_score=drop_score,
                               engine_factory=engine_factory, provenance=provenance)
    lines = normalize_target(raw)
    return to_json(lines, page_count=len(raw))
