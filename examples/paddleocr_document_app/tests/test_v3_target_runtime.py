"""Explicit target application entry, separate from legacy review baseline."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application import pipeline


def test_unconfigured_target_entry_uses_original_factory_and_target_adapter(monkeypatch):
    from application import ocr_client

    class Engine:
        def ocr(self, image):
            return [{'rec_texts': ['default-target'], 'rec_scores': [0.88]}]

    monkeypatch.setattr(ocr_client, 'create_engine', lambda: Engine())
    assert pipeline.process_target('author-owned-fixture') == {
        'schema_version': 1, 'pages': [{'page_index': 0, 'text': 'default-target'}]}


def test_target_entry_preserves_page_order_empty_page_and_consumer_contract():
    vendor_pages = [
        {'rec_texts': ['first'], 'rec_scores': [0.91]},
        {'rec_texts': [], 'rec_scores': []},
        {'rec_texts': ['last'], 'rec_scores': [0.82]},
    ]

    class Engine:
        def ocr(self, image):
            assert image == 'author-owned-fixture'
            return vendor_pages

    def factory(**parameters):
        assert parameters == {'lang': 'ch'}
        return Engine()

    target = getattr(pipeline, 'process_target_configured', None)
    assert callable(target), 'the explicit PaddleOCR3 target application entry is missing'
    output = target('author-owned-fixture', engine_factory=factory)
    assert output == {'schema_version': 1, 'pages': [
        {'page_index': 0, 'text': 'first'}, {'page_index': 1, 'text': ''},
        {'page_index': 2, 'text': 'last'}]}
    with pytest.raises(ValueError, match='invalid v2 page'):
        pipeline.process_configured('author-owned-fixture', engine_factory=factory)


def test_target_entry_rejects_mismatched_vendor_text_score_pairs():
    class Engine:
        def ocr(self, image):
            return [{'rec_texts': ['first', 'unpaired'], 'rec_scores': [0.9]}]

    target = getattr(pipeline, 'process_target_configured', None)
    assert callable(target), 'the explicit PaddleOCR3 target application entry is missing'
    with pytest.raises(ValueError, match='length mismatch'):
        target('author-owned-fixture', engine_factory=lambda **parameters: Engine())
