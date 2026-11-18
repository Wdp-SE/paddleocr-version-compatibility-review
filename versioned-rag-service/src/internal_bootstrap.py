"""Opt-in internal deployment wiring; public demo has no enterprise identity claims."""
import json
import os
from pathlib import Path
import sys

from src.internal_api import install_internal
from src.internal_security import InternalConfig,ReloadingPolicy,TokenVerifier
from src.internal_tasks import TaskStore


def configure(app):
    config=InternalConfig.from_env(os.environ)
    policy=ReloadingPolicy(config.policy_path)
    verifier=TokenVerifier(config.issuer,config.audience,jwks_url=config.jwks_url)
    store=TaskStore(config.database_url,initialize=config.development)
    root=Path(__file__).resolve().parents[2]
    sys.path.append(str(root/'demo-ui'))
    from services.registered_application import load_registry,collect_snapshot
    registry_path=Path(os.environ.get('INTERNAL_APPLICATION_REGISTRY',str(root/'config/internal_applications.json')))
    def identity():
        value=app.state.public_build_identity
        return {k:value.get(k) for k in ('corpus_fingerprint','service_code_fingerprint','retrieval_config_fingerprint','review_code_fingerprint')}
    ci=None
    if os.environ.get('INTERNAL_CI_AUDIENCE'):
        ci=TokenVerifier('https://token.actions.githubusercontent.com',os.environ['INTERNAL_CI_AUDIENCE'],
                        jwks_url='https://token.actions.githubusercontent.com/.well-known/jwks')
    install_internal(app,store=store,policy=policy,verifier=verifier,registry_loader=lambda:load_registry(registry_path),
                     collector=collect_snapshot,build_identity=identity,
                     provider=lambda:getattr(app.state.public_generator,'provider',None),ci_verifier=ci,
                     retrieval_ready=lambda:bool(getattr(app.state.public_base_knowledge_index,'ready',False)))
