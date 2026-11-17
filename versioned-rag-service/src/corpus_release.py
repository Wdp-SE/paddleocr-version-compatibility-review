"""Administrator-only immutable corpus releases. A passed gate is an operator artifact,
not an assertion that static tests establish answer accuracy. Services reload on rollout.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from src.internal_tasks import canonical, fingerprint


def tree_identity(root):
    root=Path(root).resolve();files={};size=0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Release cannot contain symbolic links')
        if not path.is_file():continue
        if len(files)>=10000:raise ValueError('Release file budget exceeded')
        size+=path.stat().st_size
        if size>200_000_000:raise ValueError('Release byte budget exceeded')
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:raise ValueError('Empty release')
    return fingerprint(files)


def validate_corpus(root):
    from src.public_knowledge import PublicKnowledgeIndex
    from src.public_retrieval_runtime import PublicRetrievalRuntime
    from src.figure_sidecar_integrity import validate_reviewed_sidecar
    root=Path(root)
    index=PublicKnowledgeIndex(root=root)
    if not index.ready or index.manifest.get('workspace_id')!='paddleocr':raise ValueError('Unavailable corpus')
    validate_reviewed_sidecar(sidecar_path=root/'figure_evidence_reviewed.json',
        lock_path=root/'figure_evidence_reviewed.lock.json',manifest_path=root/'corpus_manifest.json')
    PublicRetrievalRuntime(index,config_path=root/'public_retrieval_runtime.json')


def _gate(root,gate,code_identity):
    if (gate.get('passed') is not True or gate.get('tree_sha256')!=tree_identity(root)
            or gate.get('service_code_fingerprint')!=code_identity
            or not re.fullmatch('[a-f0-9]{64}',str(gate.get('evaluation_sha256','')))):
        raise ValueError('Evaluation gate does not bind this corpus and service build')


def stage(source,releases,gate,*,code_identity,validate=validate_corpus):
    source=Path(source).resolve();releases=Path(releases).resolve();releases.mkdir(parents=True,exist_ok=True)
    if source==releases or source in releases.parents:raise ValueError('Release directory must be outside corpus')
    _gate(source,gate,code_identity);validate(source)
    name=fingerprint(gate);target=releases/name
    if target.exists():
        _gate(target/'corpus',gate,code_identity);return name
    temporary=releases/('.stage-'+uuid.uuid4().hex);temporary.mkdir()
    # Keep failed stages for administrator diagnosis. Never remove an existing release.
    shutil.copytree(source,temporary/'corpus',symlinks=True)
    _gate(temporary/'corpus',gate,code_identity);validate(temporary/'corpus')
    (temporary/'gate.json').write_text(canonical(gate),encoding='utf8')
    temporary.rename(target)
    return name


@contextmanager
def _lock(pointer):
    path=pointer.with_suffix('.lock')
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise ValueError('Release activation already in progress') from None
    try:yield
    finally:os.close(fd);path.unlink()


def _release(root,name,code_identity,validate):
    if not isinstance(name,str) or not re.fullmatch('[a-f0-9]{64}',name):raise ValueError('Invalid release identity')
    root=Path(root).resolve();folder=root/name
    if folder.is_symlink() or (folder/'corpus').is_symlink():raise ValueError('Release path changed')
    gate=json.loads((folder/'gate.json').read_text(encoding='utf8'))
    if fingerprint(gate)!=name:raise ValueError('Gate identity changed')
    _gate(folder/'corpus',gate,code_identity);validate(folder/'corpus')
    return folder/'corpus'


def activate(root,pointer,name,*,expected,code_identity,validate=validate_corpus):
    root=Path(root).resolve();pointer=Path(pointer).absolute()
    if pointer.parent.resolve()!=root or pointer.is_symlink():raise ValueError('Pointer outside release directory')
    _release(root,name,code_identity,validate)
    with _lock(pointer):
        current=json.loads(pointer.read_text(encoding='utf8')).get('release') if pointer.exists() else None
        if current!=expected:raise ValueError('Active release changed; inspect before retrying')
        temporary=root/('.pointer-'+uuid.uuid4().hex)
        with temporary.open('x',encoding='utf8') as stream:
            stream.write(canonical({'release':name}));stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,pointer)


def resolve(pointer,*,code_identity,validate=validate_corpus):
    pointer=Path(pointer).absolute()
    if pointer.is_symlink():raise ValueError('Invalid active pointer')
    name=json.loads(pointer.read_text(encoding='utf8'))['release']
    return _release(pointer.parent,name,code_identity,validate)
