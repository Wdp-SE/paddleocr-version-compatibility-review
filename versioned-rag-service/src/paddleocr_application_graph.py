"""Conservative, bounded graph of submitted text. No imports or execution."""
from __future__ import annotations
import ast
from collections import defaultdict,deque
from pathlib import PurePosixPath


def analyze_application(files:list[dict])->dict:
    from src.paddleocr_compatibility import _files, _tree
    rows=_files(files); modules={str(PurePosixPath(f['path']).with_suffix('')).replace('/','.'):f for f in rows if f['path'].endswith('.py')}
    graph={'schema_version':1,'nodes':[],'edges':[],'gaps':[],
           'coverage':{'analysis':'bounded_direct_calls_and_result_flow','runtime_verified':False}}
    functions={}; bindings={}; node_ids=set(); edge_ids=set()
    def location(file,node):
        start=max(1,getattr(node,'lineno',1)); end=min(len(file['lines']),getattr(node,'end_lineno',start) or start)
        return {'path':file['path'],'sha256':file['sha256'],'line_start':start,'line_end':end,
                'excerpt':'\n'.join(file['lines'][start-1:end])}
    def gap(code,file,node,detail):
        if len(graph['gaps'])<100:graph['gaps'].append({'code':code,'detail':detail,'application':location(file,node)})
    def add_node(module,name,file,node):
        nid=f"{file['path']}::{name}"
        if nid in node_ids or len(node_ids)>=2000:
            gap('graph_node_limit_or_ambiguous_symbol',file,node,'图节点上限或同名定义；不推导精确关系。');return None
        node_ids.add(nid); graph['nodes'].append({'id':nid,'module':module,'symbol':name,'application':location(file,node)})
        return nid
    for module,file in modules.items():
        try:tree=_tree(file['content'])
        except (SyntaxError,ValueError):
            gap('unparsed_application_file',file,ast.Constant(),'应用文件不能安全解析。');continue
        add_node(module,'<module>',file,tree)
        for statement in tree.body:
            if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef)):
                nid=add_node(module,statement.name,file,statement)
                if nid:
                    functions[(module,statement.name)]=(file,statement,nid)
                    if statement.decorator_list:
                        gap('decorated_function_flow_not_resolved',file,statement,'装饰器可能替换函数，不能证明调用和结果流。')
            elif isinstance(statement,ast.ClassDef):
                gap('class_method_flow_not_resolved',file,statement,'实例方法和动态绑定未做跨文件结果流证明。')
        bindings[module]={name:nid if not node.decorator_list else None for (m,name),(_,node,nid) in functions.items() if m==module}
        file['_graph_tree']=tree
    for module,file in modules.items():
        if '_graph_tree' not in file:continue
        for statement in file['_graph_tree'].body:
            if isinstance(statement,ast.ImportFrom):
                if statement.level:
                    parts=module.split('.')[:-1]
                    if statement.level>len(parts)+1:
                        gap('missing_local_module',file,statement,'相对导入越过已提交项目。');continue
                    prefix=parts[:len(parts)-(statement.level-1)]
                    target='.'.join(prefix+([statement.module] if statement.module else []))
                else:target=statement.module or ''
                for alias in statement.names:
                    name=alias.asname or alias.name
                    entry=functions.get((target,alias.name))
                    if entry:bindings[module][name]=entry[2] if not entry[1].decorator_list else None
                    elif statement.level or target in modules:
                        gap('missing_local_module',file,statement,'未提供本地模块/函数；不能证明包装器结果流。')
                    else:
                        bindings[module][name]=None
            elif isinstance(statement,ast.Import):
                for alias in statement.names:
                    if alias.name in modules:
                        for (m,name),(_,_,nid) in functions.items():
                            if m==alias.name:bindings[module][(alias.asname or alias.name)+'.'+name]=nid if not functions[(m,name)][1].decorator_list else None
    def edge(source,target,kind,file,node):
        key=(source,target,kind,file['path'],getattr(node,'lineno',1))
        if key in edge_ids:return
        if len(graph['edges'])>=4000:
            gap('graph_edge_limit',file,node,'图边上限；后续关系未覆盖。');return
        edge_ids.add(key);graph['edges'].append({'source':source,'target':target,'kind':kind,'application':location(file,node)})
    def call_name(expr):
        if isinstance(expr,ast.Name):return expr.id
        if isinstance(expr,ast.Attribute) and isinstance(expr.value,ast.Name):return expr.value.id+'.'+expr.attr
        return None
    def scan(module,file,statements,caller,local,origins):
        for statement in statements:
            if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):continue
            if isinstance(statement,(ast.If,ast.For,ast.While,ast.Try,ast.With,ast.Match)):
                gap('conditional_or_dynamic_flow',file,statement,'条件、循环或上下文内结果流尚未证明。')
                # Never infer result-flow through control flow, nor trust bindings after it.
                for n in ast.walk(statement):
                    if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store):local[n.id]=None;origins.pop(n.id,None)
                continue
            if isinstance(statement,(ast.Global,ast.Nonlocal)):
                gap('mutable_outer_binding',file,statement,'外部可变绑定未分析。');continue
            calls=[n for n in ast.walk(statement) if isinstance(n,ast.Call)]
            resolved={}
            for call in calls:
                name=call_name(call.func); target=local.get(name)
                if target:
                    edge(caller,target,'reference',file,call);resolved[id(call)]=target
                    for arg in call.args:
                        if isinstance(arg,ast.Name) and arg.id in origins:
                            edge(origins[arg.id],target,'result_flow',file,arg)
                elif name in ('getattr','__import__','eval','exec'):
                    gap('dynamic_call_unresolved',file,call,'动态调用未执行；影响关系需要人工确认。')
            value=getattr(statement,'value',None)
            producer=resolved.get(id(value)) if isinstance(value,ast.Call) else origins.get(value.id) if isinstance(value,ast.Name) else None
            if isinstance(statement,ast.Return) and producer:
                edge(producer,caller,'result_flow',file,statement)
            targets=statement.targets if isinstance(statement,ast.Assign) else [statement.target] if isinstance(statement,(ast.AnnAssign,ast.AugAssign)) else []
            for target in targets:
                if isinstance(target,ast.Attribute):
                    name=call_name(target)
                    if name in local:
                        local[name]=None
                        gap('rebound_symbol',file,statement,'模块属性重绑定，不能证明后续调用。')
                if isinstance(target,ast.Name):
                    if target.id in local and local[target.id]:gap('rebound_symbol',file,statement,'调用名称重绑定，之后不推导原导入。')
                    # Direct function aliases are supported; call results are data origins, not functions.
                    local[target.id]=local.get(value.id) if isinstance(value,ast.Name) else None
                    if producer:origins[target.id]=producer
                    else:origins.pop(target.id,None)
                    if isinstance(value,ast.Call) and producer:edge(producer,caller,'result_flow',file,statement)
    for module,file in modules.items():
        if '_graph_tree' not in file:continue
        module_bindings=dict(bindings[module])
        scan(module,file,file['_graph_tree'].body,file['path']+'::<module>',module_bindings,{})
        for (m,name),(f,node,nid) in functions.items():
            if m!=module:continue
            local=dict(module_bindings)
            # A binding assigned anywhere in a function is local for its whole body.
            assigned={n.id for statement in node.body for n in ast.walk(statement)
                      if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
            for name in assigned:
                if local.get(name):gap('shadowed_symbol',f,node,'函数局部赋值遮蔽名称；不使用模块导入绑定。')
                local[name]=None
            args=node.args.posonlyargs+node.args.args+node.args.kwonlyargs
            if node.args.vararg:args.append(node.args.vararg)
            if node.args.kwarg:args.append(node.args.kwarg)
            for arg in args:
                if local.get(arg.arg):gap('shadowed_symbol',f,arg,'参数遮蔽导入，不能推导原函数。')
                local[arg.arg]=None
            scan(module,f,node.body,nid,local,{})
    # Cycle detection is bounded and based on explicit reference edges.
    links=defaultdict(list)
    for e in graph['edges']:
        if e['kind']=='reference':links[e['source']].append(e['target'])
    for nid in node_ids:
        queue=deque([(nid,{nid})]); seen=set()
        while queue:
            current,path=queue.popleft()
            if current in seen:continue
            seen.add(current)
            for nxt in links[current]:
                if nxt in path:
                    if len(graph['gaps'])<100:graph['gaps'].append({'code':'cyclic_application_relation','detail':'循环关系已终止；不推导完整执行可达性。'})
                    queue.clear();break
                if len(path)<8:queue.append((nxt,path|{nxt}))
        if any(g['code']=='cyclic_application_relation' for g in graph['gaps']):break
    return graph


def trace_impacts(graph:dict,findings:list[dict])->list[dict]:
    paths=[];links=defaultdict(list)
    for e in graph['edges']:
        if e['kind']=='result_flow':links[e['source']].append((e['target'],e))
        else:links[e['target']].append((e['source'],e))
    nodes={n['id']:n for n in graph['nodes']}
    for finding in findings:
        app=finding['application']
        start=app.get('line_start',app.get('line',1))
        seeds=[n['id'] for n in graph['nodes'] if n['symbol']!='<module>' and n['application']['path']==app['path']
               and n['application']['line_start']<=start<=n['application']['line_end']]
        if not seeds:seeds=[n['id'] for n in graph['nodes'] if n['symbol']=='<module>' and n['application']['path']==app['path']]
        for seed in seeds:
            queue=deque([(seed,[],{seed})]);visited={seed}
            while queue and len(paths)<100:
                current,chain,seen=queue.popleft()
                for nxt,e in links[current]:
                    if nxt in seen or nxt in visited or len(chain)>=8:continue
                    visited.add(nxt);new=chain+[e]
                    paths.append({'finding_id':finding['finding_id'],'origin_node':seed,'target_node':nxt,
                                  'level':'candidate','flow_proven':all(x['kind']=='result_flow' for x in new),
                                  'edges':new,'application':nodes[nxt]['application'],
                                  'runtime_verified':False,'meaning':'依赖/结果流候选，尚未证明下游业务破坏'})
                    queue.append((nxt,new,seen|{nxt}))
    return paths
