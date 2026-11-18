import json
import time
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from src.internal_security import AccessPolicy, TokenVerifier, denied_egress, InternalConfig


@pytest.fixture
def keys():
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    return key,key.public_key()


def token(key,**overrides):
    return jwt.encode({'iss':'https://identity.example','aud':'ocr-review','sub':'alice',
        'exp':int(time.time())+60,'iat':int(time.time()),**overrides},key,algorithm='RS256',headers={'kid':'test'})


def test_token_requires_signature_issuer_audience_expiry_and_subject(keys):
    private,public=keys;verify=TokenVerifier('https://identity.example','ocr-review',key_resolver=lambda t:public)
    assert verify.verify(token(private)).subject=='alice'
    for fields in ({'iss':'https://evil.example'},{'aud':'other'},{'exp':1},{'sub':''}):
        with pytest.raises(PermissionError):verify.verify(token(private,**fields))
    with pytest.raises(PermissionError):verify.verify(jwt.encode({'sub':'alice'},'x'*32,algorithm='HS256'))


def test_acl_is_server_owned_and_missing_permissions_fail_closed():
    policy=AccessPolicy({'principals':{'alice':{'document-ocr':['read','submit']}},
                         'applications':{'document-ocr':{'classification':'internal','external_models':[]}}})
    policy.require('alice','document-ocr','submit')
    for args in [('bob','document-ocr','read'),('alice','other','read'),('alice','document-ocr','review')]:
        with pytest.raises(PermissionError):policy.require(*args)


def test_external_data_policy_and_secret_content_are_independent_controls():
    app={'classification':'internal','external_models':['deepseek']}
    assert denied_egress(app,'deepseek','ordinary source') is None
    assert denied_egress(app,'other','ordinary source')=='PROVIDER_NOT_APPROVED'
    assert denied_egress(app,'deepseek','api_key="sk-12345678901234567890"')=='POSSIBLE_SECRET'
    assert denied_egress({'classification':'restricted','external_models':['deepseek']},'deepseek','x')=='DATA_CLASSIFICATION_DENIED'


def test_production_config_requires_real_identity_policy_and_postgres(tmp_path):
    policy=tmp_path/'acl.json';policy.write_text(json.dumps({'principals':{},'applications':{}}))
    env={'INTERNAL_OIDC_ISSUER':'https://identity.example','INTERNAL_OIDC_AUDIENCE':'ocr-review',
         'INTERNAL_OIDC_JWKS_URL':'https://identity.example/jwks','INTERNAL_ACCESS_POLICY':str(policy),
         'INTERNAL_DATABASE_URL':'sqlite:///'+str(tmp_path/'jobs.db')}
    with pytest.raises(ValueError):InternalConfig.from_env(env)
    env['INTERNAL_ENVIRONMENT']='development'
    assert InternalConfig.from_env(env).development
    del env['INTERNAL_OIDC_AUDIENCE']
    with pytest.raises(ValueError):InternalConfig.from_env(env)


def test_actual_model_outlet_rejects_secret_evidence_not_just_request_body(monkeypatch):
    from src.internal_security import model_scope
    from src.answer_generation import StructuredAnswerGenerator
    policy=AccessPolicy({'principals':{},'applications':{'document-ocr':{'classification':'internal','external_models':['deepseek']}}})
    monkeypatch.setenv('DEEPSEEK_API_KEY','test-key-not-used')
    generator=StructuredAnswerGenerator(provider='deepseek',model='deepseek-chat')
    calls=[]
    monkeypatch.setattr(generator,'_generate_deepseek',lambda m:(calls.append(m) or ('ok',{})))
    token=model_scope.set((policy,'document-ocr'))
    try:
        with pytest.raises(PermissionError):generator._complete_with_diagnostics([{'content':'password=12345678901234567890'}])
        assert not calls
        assert generator._complete_with_diagnostics([{'content':'public evidence'}])[0]=='ok'
    finally:model_scope.reset(token)


def test_removed_jwks_key_is_rejected_after_key_set_refresh(keys,monkeypatch):
    private,public=keys
    key=jwt.algorithms.RSAAlgorithm.to_jwk(public,as_dict=True)
    key.update(kid='test',use='sig',alg='RS256')
    current={'keys':[key]};clients=[]
    original=jwt.PyJWKClient
    def client(*args,**kwargs):
        value=original(*args,**kwargs)
        def fetch():
            value.jwk_set_cache.put(current.copy())
            return current
        monkeypatch.setattr(value,'fetch_data',fetch)
        clients.append(value)
        return value
    monkeypatch.setattr(jwt,'PyJWKClient',client)
    verifier=TokenVerifier('https://identity.example','ocr-review',jwks_url='https://identity.example/jwks')
    encoded=token(private)
    assert verifier.verify(encoded).subject=='alice'
    current['keys']=[]
    clients[0].jwk_set_cache.put(None)
    with pytest.raises(PermissionError):verifier.verify(encoded)
