"""Server-owned identity, application authorization and data-egress boundaries."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Principal:
    subject: str
    claims: dict


class TokenVerifier:
    def __init__(self,issuer,audience,*,jwks_url=None,key_resolver=None):
        self.issuer,self.audience=issuer,audience
        if key_resolver is None:
            import jwt
            # Cache the JWKS set for a bounded time, not individual keys forever.
            client=jwt.PyJWKClient(jwks_url,cache_keys=False,lifespan=300,timeout=5)
            key_resolver=lambda token:client.get_signing_key_from_jwt(token).key
        self.key_resolver=key_resolver

    def verify(self,token):
        import jwt
        try:
            if not isinstance(token,str) or not 1<=len(token)<=16000:raise ValueError()
            if jwt.get_unverified_header(token).get('alg')!='RS256':raise ValueError()
            value=jwt.decode(token,self.key_resolver(token),algorithms=['RS256'],issuer=self.issuer,
                audience=self.audience,options={'require':['exp','iat','iss','aud','sub']},leeway=5)
            if not isinstance(value['sub'],str) or not 1<=len(value['sub'])<=256:raise ValueError()
            return Principal(value['sub'],value)
        except Exception:
            raise PermissionError('Invalid identity token') from None


class AccessPolicy:
    actions={'read','submit','review','worker','admin'}

    def __init__(self,value):
        if not isinstance(value,dict) or set(value)!={'principals','applications'}:raise ValueError('Invalid access policy')
        self.value=value
        if not all(isinstance(value[k],dict) for k in value):raise ValueError('Invalid access policy')
        for subject,applications in value['principals'].items():
            if not isinstance(subject,str) or not 1<=len(subject)<=256 or not isinstance(applications,dict):raise ValueError('Invalid principal')
            for application,actions in applications.items():
                if application not in value['applications'] or not isinstance(actions,list) or not set(actions)<=self.actions:
                    raise ValueError('Invalid application permission')
        for application,settings in value['applications'].items():
            if (not re.fullmatch('[a-z0-9-]{1,64}',application) or not isinstance(settings,dict)
                    or settings.get('classification') not in ('public','internal','restricted')
                    or not isinstance(settings.get('external_models',[]),list)):
                raise ValueError('Invalid application data policy')

    def require(self,subject,application,action):
        actions=self.value['principals'].get(subject,{}).get(application,[])
        if action not in actions and 'admin' not in actions:raise PermissionError('Resource unavailable')

    def applications(self,subject,action):
        return [app for app,actions in self.value['principals'].get(subject,{}).items()
                if action in actions or 'admin' in actions]

    def application(self,application):
        return self.value['applications'].get(application,{})


class ReloadingPolicy:
    """Revocation applies to the next API access; file identity is server configured."""
    def __init__(self,path):self.path=Path(path)
    def _policy(self):return AccessPolicy(json.loads(self.path.read_text(encoding='utf-8-sig')))
    @property
    def value(self):return self._policy().value
    def require(self,*args):return self._policy().require(*args)
    def applications(self,*args):return self._policy().applications(*args)
    def application(self,*args):return self._policy().application(*args)


_SECRET=re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:api[_-]?key|password|secret|access[_-]?token)\s*[=:]\s*[\"\x27]?[^\s\"\x27]{12,}',re.I)

# Request-local scope is propagated by asyncio.to_thread; never mutate a shared
# generator with one user's identity. Check the assembled prompt at the outlet.
from contextvars import ContextVar
model_scope=ContextVar('internal_model_scope',default=None)


def guard_model_messages(provider,messages):
    scope=model_scope.get()
    if scope is None:
        if os.environ.get('APP_ENV','').strip().casefold()=='internal':
            raise PermissionError('MODEL_REQUEST_SCOPE_REQUIRED')
        return
    policy,application=scope
    reason=denied_egress(policy.application(application),provider,
                         '\n'.join(str(row.get('content','')) for row in messages))
    if reason:raise PermissionError(reason)


def denied_egress(application,provider,text):
    if application.get('classification') not in ('public','internal'):return 'DATA_CLASSIFICATION_DENIED'
    if provider not in application.get('external_models',[]):return 'PROVIDER_NOT_APPROVED'
    if not isinstance(text,str) or len(text.encode())>200_000:return 'CONTEXT_BUDGET_EXCEEDED'
    if _SECRET.search(text):return 'POSSIBLE_SECRET'
    return None


@dataclass(frozen=True)
class InternalConfig:
    database_url: str
    issuer: str
    audience: str
    jwks_url: str
    policy_path: Path
    development: bool=False

    @classmethod
    def from_env(cls,env):
        required=['INTERNAL_DATABASE_URL','INTERNAL_OIDC_ISSUER','INTERNAL_OIDC_AUDIENCE',
                  'INTERNAL_OIDC_JWKS_URL','INTERNAL_ACCESS_POLICY']
        if any(not env.get(k,'').strip() for k in required):raise ValueError('Internal deployment configuration incomplete')
        development=env.get('INTERNAL_ENVIRONMENT','production')=='development'
        database=env['INTERNAL_DATABASE_URL']
        if not development and not database.startswith(('postgresql://','postgres://')):
            raise ValueError('Production requires PostgreSQL')
        for name in ('INTERNAL_OIDC_ISSUER','INTERNAL_OIDC_JWKS_URL'):
            parsed=urlsplit(env[name])
            if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                raise ValueError('Identity endpoints must be trusted HTTPS server configuration')
        if len(env['INTERNAL_OIDC_AUDIENCE'])>256:raise ValueError('Invalid audience')
        path=Path(env['INTERNAL_ACCESS_POLICY']).resolve()
        AccessPolicy(json.loads(path.read_text(encoding='utf-8-sig')))
        return cls(database,env['INTERNAL_OIDC_ISSUER'],env['INTERNAL_OIDC_AUDIENCE'],
                   env['INTERNAL_OIDC_JWKS_URL'],path,development)
