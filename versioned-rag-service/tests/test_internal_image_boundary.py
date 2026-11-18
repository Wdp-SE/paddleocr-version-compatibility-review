"""Guard the Docker COPY boundary: local sessions/SQLite and secrets are not source."""
from pathlib import Path
import shlex


def test_image_copy_uses_runtime_source_allowlist():
    root=Path(__file__).resolve().parents[2]
    dockerfile=root/'deploy/internal/Dockerfile'
    allowed={'deploy/internal/requirements.lock','versioned-rag-service/src/*.py',
        'versioned-rag-service/config/*.json','demo-ui/*.py','demo-ui/components/*.py',
        'demo-ui/services/*.py','demo-ui/assets/*.svg','change-review-agent/app',
        'change-review-agent/config/*.json','config/internal_applications.json','deploy/internal/entrypoint.py'}
    for line in dockerfile.read_text().splitlines():
        if line.startswith('COPY '):
            source=shlex.split(line)[1]
            assert source in allowed,f'Unbounded COPY can include sessions, databases or secrets: {source}'
