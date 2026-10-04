"""Original student document adapter: existing 2.9.1 layout API to review."""
from paddleocr import PPStructure as LayoutEngine


def normalize_regions(scan_path: str) -> list[dict]:
    engine = LayoutEngine()
    regions = engine(scan_path)
    return [
        {"kind": region["type"], "bounds": region["bbox"], "content": region["res"]}
        for region in regions
    ]
