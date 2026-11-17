"""Lease-fenced LangGraph execution, using JSON state in the task database.

Only the latest checkpoint and its pending writes are retained. No pickle, runtime
clients, authentication tokens or user-controlled Python types are deserialized.
An interrupted model-bearing node is deliberately not replayed automatically.
"""
import json
from collections import deque
from threading import RLock

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from src.internal_tasks import canonical, fingerprint


class UnsafeGraphReplay(ValueError):
    pass


def _json(value):
    return json.loads(canonical(value))


class _JSONSerializer:
    def dumps_typed(self,value):return 'json',canonical(value).encode()
    def loads_typed(self,value):
        if value[0]!='json':raise ValueError('Unsupported checkpoint encoding')
        return json.loads(value[1])


class _LeaseSaver(BaseCheckpointSaver):
    def __init__(self, runtime, graph, binding, state):
        super().__init__(serde=_JSONSerializer())
        self.runtime,self.graph,self.binding=runtime,graph,binding
        self.state=state or {'checkpoint':None,'active_nodes':[]}
        self.lock=RLock()
        self.retired_checkpoints=deque(maxlen=128)

    def persist(self):
        self.runtime.check()
        self.runtime.store.graph_state(self.runtime.claim,self.graph,self.binding,payload=self.state)

    def get_tuple(self, config):
        self.runtime.check()
        with self.lock:
            saved=_json(self.state.get('checkpoint'))
            if saved is None:return None
            requested=config['configurable'].get('checkpoint_id')
            if requested and requested!=saved['checkpoint']['id']:return None
            return CheckpointTuple(config=saved['config'],checkpoint=saved['checkpoint'],
                metadata=saved['metadata'],parent_config=None,
                pending_writes=[tuple(w) for w in saved.get('pending_writes',[])])

    def put(self, config, checkpoint, metadata, new_versions):
        with self.lock:
            previous=self.state.get('checkpoint')
            cfg={'configurable':{'thread_id':config['configurable']['thread_id'],
                'checkpoint_ns':config['configurable'].get('checkpoint_ns',''),
                'checkpoint_id':checkpoint['id']}}
            self.state['checkpoint']=_json({'config':cfg,'checkpoint':checkpoint,'metadata':metadata,'pending_writes':[]})
            # A durable superstep checkpoint commits all successfully completed nodes.
            completed=self.state['active_nodes'][:]
            self.state['active_nodes']=[]
            self.persist()
            if previous and previous['checkpoint']['id']!=checkpoint['id']:
                self.retired_checkpoints.append(previous['checkpoint']['id'])
            for node in completed:
                self.runtime.store.graph_event(self.runtime.claim,self.graph,node,'COMPLETED')
            return cfg

    def put_writes(self, config, writes, task_id, task_path=''):
        with self.lock:
            saved=self.state.get('checkpoint')
            checkpoint_id=config['configurable']['checkpoint_id']
            if checkpoint_id in self.retired_checkpoints:
                # LangGraph dispatches pending-write persistence concurrently.
                # A newer committed superstep already incorporates this write;
                # do not roll back its state or reject valid delayed persistence.
                self.runtime.check()
                self.runtime.store.graph_state(self.runtime.claim,self.graph,self.binding)
                return
            if saved is None or checkpoint_id!=saved['checkpoint']['id']:
                raise ValueError('Graph checkpoint sequence changed')
            # Failure objects can include untrusted exception strings and are never saved.
            clean=[(channel,value) for channel,value in writes if channel not in ('__error__','__interrupt__')]
            saved['pending_writes']=[w for w in saved['pending_writes'] if w[0]!=task_id]
            saved['pending_writes'].extend(_json([[task_id,c,v] for c,v in clean]))
            self.persist()


class _NodeEvents(BaseCallbackHandler):
    raise_error=True

    def __init__(self,saver):
        self.saver=saver

    def on_chain_start(self,serialized,inputs,*,run_id,metadata=None,**kwargs):
        metadata=metadata or {}
        node=metadata.get('langgraph_node')
        # Only top-level node runs; nested runnable callbacks carry the same metadata.
        if not node or not any(t.startswith('graph:step:') for t in kwargs.get('tags',[])):return
        with self.saver.lock:
            self.saver.runtime.check()
            if node not in self.saver.state['active_nodes']:
                self.saver.state['active_nodes'].append(node)
                self.saver.persist()
                self.saver.runtime.store.graph_event(self.saver.runtime.claim,self.saver.graph,node,'STARTED')


class TaskGraphRuntime:
    """Injected into author-owned review graphs by the leased worker only."""
    def __init__(self,store,claim,*,cancelled,authorize=lambda:None,scope=''):
        self.store,self.claim,self.cancelled,self.authorize=store,claim,cancelled,authorize
        self.scope=scope

    def scoped(self,scope):
        if scope not in ('baseline','candidate'):raise ValueError('Invalid review pass scope')
        return TaskGraphRuntime(self.store,self.claim,cancelled=self.cancelled,
                                authorize=self.authorize,scope=scope)

    def check(self):
        if self.cancelled():raise PermissionError('Graph lease lost or cancelled')
        self.authorize()

    def run(self,graph_name,graph_version,build_graph,initial_state):
        self.check()
        if self.scope:graph_name=graph_name+'_'+self.scope
        binding=fingerprint({'task':self.claim['id'],'request_hash':self.claim['request_hash'],
            'owner':self.claim['owner'],'application':self.claim['application'],
            'build':self.claim['request'].get('build_identity'),'graph':graph_name,
            'version':graph_version,'input':initial_state})
        state=self.store.graph_state(self.claim,graph_name,binding)
        active=set((state or {}).get('active_nodes',[]))
        if 'model_advice' in active or ('official_lookup' in active and self.claim['request'].get('generate_advice')):
            raise UnsafeGraphReplay('Ambiguous model invocation; submit a new task')
        saver=_LeaseSaver(self,graph_name,binding,state)
        graph=build_graph(checkpointer=saver)
        config={'configurable':{'thread_id':self.claim['id']+':'+graph_name},
            'callbacks':[_NodeEvents(saver)],'recursion_limit':64}
        # LangGraph resumes pending work instead of injecting the input again.
        output=graph.invoke(None if state and state.get('checkpoint') else initial_state,
                            config=config,durability='sync')
        self.check()
        return output
