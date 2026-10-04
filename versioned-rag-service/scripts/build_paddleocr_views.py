"""Build deterministic view sidecar; neural experiments rebuild with their tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
from src.paddleocr_corpus import SERVICE_ROOT, write_json, validate_paddleocr_manifest
from src.paddleocr_retrieval_views import build_views, validate_views


def build(root: Path, *, token_count=None, tokenizer_id='unicode_character_conservative', max_tokens=384):
    chunks=json.loads((root/'chunks.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'corpus_manifest.json').read_text(encoding='utf-8'))
    validate_paddleocr_manifest(root,manifest,chunks)
    # Character budget is a conservative fallback, explicitly not neural token counts.
    views=build_views(chunks,token_count=token_count or len,max_tokens=max_tokens)
    validate_views(chunks,views)
    payload={'schema_version':1,'tokenizer_id':tokenizer_id,'max_tokens':max_tokens,
             'chunks_sha256':hashlib.sha256((root/'chunks.json').read_bytes()).hexdigest(), 'views':views}
    write_json(root/'retrieval_views.json',payload,compact=True)
    return {'views':len(views),'parents':len(chunks),'tokenizer_id':tokenizer_id}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=SERVICE_ROOT/'public_corpus_paddleocr')
    print(build(parser.parse_args().root))
