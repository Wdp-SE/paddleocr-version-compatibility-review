"""Short-lived authenticated requests; no shared user-token or connection cache."""
import requests
from services.public_knowledge_client import PublicKnowledgeClient
from services.rag_client import ServiceError


class InternalClient(PublicKnowledgeClient):
    def __init__(self,base_url,authorization,*,timeout=20):
        self.base_url,self.authorization,self.timeout=base_url,authorization,timeout
        self.application=None

    def _request(self,*args,**kwargs):
        with requests.Session() as session:
            session.headers['Authorization']=self.authorization
            if self.application:session.headers['X-Application-ID']=self.application
            try:
                return PublicKnowledgeClient(self.base_url,session=session,timeout=self.timeout,retry_limit=0)._request(*args,**kwargs)
            except ServiceError as exc:
                messages={'RAG_TIMEOUT':'内部知识服务响应超时，请通过任务编号查看状态。',
                          'RAG_UNAVAILABLE':'内部知识服务连接失败，请联系维护人员。',
                          'LLM_BUDGET_EXHAUSTED':'模型服务容量或已配置预算不足，仍可检索原文。'}
                if exc.code in messages:raise ServiceError(messages[exc.code],exc.detail,exc.code) from None
                raise
