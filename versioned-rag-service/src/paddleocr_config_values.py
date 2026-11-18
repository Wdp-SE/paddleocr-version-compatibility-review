"""Resolve a bounded declaration chain from submitted text, never Python execution or disk IO."""
from __future__ import annotations
import ast
import copy
import hashlib
import json


def _parse_data(path,text):
    if path.endswith('.json'):
        def pairs(items):
            value={}
            for key,item in items:
                if key in value:raise ValueError('duplicate configuration key')
                value[key]=item
            return value
        value=json.loads(text,object_pairs_hook=pairs)
    elif path.endswith(('.yaml','.yml')):
        import yaml
        tokens=list(yaml.scan(text))
        if len(tokens)>5000 or any(isinstance(t,(yaml.tokens.AliasToken,yaml.tokens.AnchorToken,yaml.tokens.TagToken)) for t in tokens):
            raise ValueError('unsupported YAML aliases or tags')
        # YAML duplicate keys must not silently change effective precedence.
        class UniqueLoader(yaml.SafeLoader):pass
        def mapping(loader,node):
            pairs=loader.construct_pairs(node,deep=True);result={}
            for key,item in pairs:
                if not isinstance(key,str) or key in result:raise ValueError('ambiguous YAML mapping')
                result[key]=item
            return result
        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
        value=yaml.load(text,Loader=UniqueLoader)
    else:raise ValueError('configuration is not JSON/YAML')
    pending=[(value,0)];count=0
    while pending:
        item,depth=pending.pop();count+=1
        if count>2000 or depth>32:raise ValueError('configuration tree budget')
        if isinstance(item,dict):
            if any(not isinstance(k,str) for k in item):raise ValueError('configuration keys')
            pending.extend((v,depth+1) for v in item.values())
        elif isinstance(item,list):pending.extend((v,depth+1) for v in item)
        elif type(item) not in (str,int,float,bool,type(None)):raise ValueError('configuration scalar')
    return value


def parse_data(path,text):
    import yaml
    try:return _parse_data(path,text)
    except (yaml.YAMLError,RecursionError,TypeError,ValueError) as exc:
        raise ValueError('configuration data cannot be parsed safely') from exc


class DeclarationResolver:
    def __init__(self,files,path,tree):
        self.files={f['path']:f['content'] for f in files};self.path=path;self.tree=tree
        self.nodes=list(ast.walk(tree))
        if len(self.nodes)>15000:raise ValueError('AST budget')
        self.parents={id(c):n for n in self.nodes for c in ast.iter_child_nodes(n)}

    def scope(self,node):
        parent=self.parents.get(id(node))
        while parent is not None:
            if isinstance(parent,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):return parent
            parent=self.parents.get(id(parent))
        return self.tree

    def unconditional(self,node):
        parent=self.parents.get(id(node))
        while parent is not None:
            if isinstance(parent,(ast.If,ast.For,ast.While,ast.Try,ast.With,ast.Match,ast.IfExp,ast.Lambda)):
                return False
            parent=self.parents.get(id(parent))
        return True

    def source(self,node):
        lines=self.files[self.path].splitlines()
        return {'path':self.path,'sha256':hashlib.sha256(self.files[self.path].encode()).hexdigest(),
                'line':node.lineno,'end_line':node.end_lineno,'col':node.col_offset,'end_col':node.end_col_offset,
                'snippet':'\n'.join(lines[node.lineno-1:node.end_lineno]),'kind':'python_scalar'}

    def check_container_uses(self,name,value):
        if not isinstance(value,(dict,list)):return
        for ref in self.nodes:
            if not isinstance(ref,ast.Name) or ref.id!=name or not isinstance(ref.ctx,ast.Load):continue
            current=ref;leaf=value
            while isinstance(self.parents.get(id(current)),ast.Subscript) and self.parents[id(current)].value is current:
                current=self.parents[id(current)]
                try:leaf=leaf[ast.literal_eval(current.slice)]
                except (ValueError,TypeError,KeyError,IndexError):raise ValueError('configuration selector unknown')
            if isinstance(leaf,(dict,list)):raise ValueError('configuration container alias or escape')

    def resolve(self,node,*,depth=0,seen=()):
        if depth>8 or id(node) in seen:raise ValueError('configuration resolution budget')
        nested={'depth':depth+1,'seen':(*seen,id(node))}
        if isinstance(node,(ast.Constant,ast.UnaryOp)):
            return ast.literal_eval(node),[self.source(node)]
        if isinstance(node,ast.Subscript):
            value,sources=self.resolve(node.value,**nested);key=ast.literal_eval(node.slice)
            if not isinstance(value,dict) or not isinstance(key,str) or key not in value:raise ValueError('unknown configuration field')
            return value[key],[{**s,'key_path':s.get('key_path',[])+[key]} if s['kind']=='configuration_file' else s for s in sources]
        if isinstance(node,ast.Name):
            scope=self.scope(node)
            outer=self.parents.get(id(scope))
            while outer is not None:
                if isinstance(outer,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    raise ValueError('configuration closure binding unsupported')
                outer=self.parents.get(id(outer))
            if isinstance(scope,ast.FunctionDef) and any(a.arg==node.id for a in (*scope.args.posonlyargs,*scope.args.kwonlyargs)):
                raise ValueError('unsupported factory parameter kind')
            def base(value):
                while isinstance(value,(ast.Subscript,ast.Attribute)):value=value.value
                return value.id if isinstance(value,ast.Name) else None
            if any((isinstance(n,(ast.Subscript,ast.Attribute)) and isinstance(n.ctx,(ast.Store,ast.Del)) and base(n)==node.id)
                   or (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and base(n.func.value)==node.id)
                   or (isinstance(n,ast.Call) and any(isinstance(a,ast.Name) and a.id==node.id for a in n.args))
                   for n in self.nodes):raise ValueError('configuration object mutation or escape')
            if any(isinstance(n,ast.Global) and node.id in n.names for n in self.nodes):raise ValueError('mutable global')
            if isinstance(scope,ast.FunctionDef) and node.id in [a.arg for a in scope.args.args]:
                if any(isinstance(n,ast.Name) and n.id==node.id and isinstance(n.ctx,(ast.Store,ast.Del))
                       and self.scope(n) is scope for n in self.nodes):raise ValueError('factory parameter rebound')
                calls=[n for n in self.nodes if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==scope.name]
                if len(calls)!=1 or scope.decorator_list or scope.args.vararg or scope.args.kwarg or scope.args.posonlyargs:
                    raise ValueError('unknown factory invocation')
                call=calls[0];names=[a.arg for a in scope.args.args]
                if len(call.args)>len(names) or any(k.arg is None for k in call.keywords):raise ValueError('factory argument binding')
                values=dict(zip(names,call.args))
                for kw in call.keywords:
                    if kw.arg not in names or kw.arg in values:raise ValueError('factory argument binding')
                    values[kw.arg]=kw.value
                required=names[:len(names)-len(scope.args.defaults)]
                if any(name not in values for name in required):raise ValueError('missing factory argument')
                if node.id not in values or not self.unconditional(call):raise ValueError('factory argument not explicit')
                if any(isinstance(n,ast.Name) and n.id==scope.name and isinstance(n.ctx,ast.Store) for n in self.nodes):raise ValueError('factory rebound')
                if any(isinstance(n,ast.arg) and n.arg==scope.name or isinstance(n,ast.alias) and (n.asname or n.name)==scope.name
                       for n in self.nodes):raise ValueError('factory callable binding shadowed')
                if sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==scope.name for n in self.nodes)!=1:
                    raise ValueError('factory definition ambiguous')
                if any(isinstance(n,ast.Name) and n.id==scope.name and isinstance(n.ctx,ast.Load)
                       and not (isinstance(self.parents.get(id(n)),ast.Call) and self.parents[id(n)].func is n)
                       for n in self.nodes):raise ValueError('factory alias escapes')
                return self.resolve(values[node.id],**nested)
            writes=[n for n in self.nodes if isinstance(n,ast.Name) and n.id==node.id and isinstance(n.ctx,(ast.Store,ast.Del))
                    and self.scope(n) is scope]
            if not writes and scope is not self.tree:
                writes=[n for n in self.nodes if isinstance(n,ast.Name) and n.id==node.id and isinstance(n.ctx,(ast.Store,ast.Del))
                        and self.scope(n) is self.tree]
            if len(writes)!=1:raise ValueError('configuration binding not unique')
            statement=self.parents[id(writes[0])]
            if not isinstance(statement,(ast.Assign,ast.AnnAssign)) or statement.lineno>=node.lineno or not self.unconditional(statement):
                raise ValueError('configuration binding not dominant')
            if isinstance(statement,ast.Assign) and len(statement.targets)!=1:raise ValueError('multiple assignment')
            value,sources=self.resolve(statement.value,**nested)
            self.check_container_uses(node.id,value)
            uses=[n for n in self.nodes if isinstance(n,ast.Name) and n.id==node.id and isinstance(n.ctx,ast.Load)]
            if type(value) in (int,float) and len(uses)>1:
                sources=[{**s,'shared_consumers':[self.source(n) for n in uses if n is not node]} for s in sources]
            return value,sources
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            root=node.func.value
            if not isinstance(root,ast.Name) or len(node.args)!=1 or node.keywords:raise ValueError('configuration loader')
            imports=[n for n in self.nodes if isinstance(n,ast.Import) for a in n.names
                     if (a.asname or a.name)==root.id and (a.name,node.func.attr) in (('json','load'),('yaml','safe_load'))
                     and self.scope(n) in (self.tree,self.scope(node))]
            if len(imports)!=1 or imports[0].lineno>node.lineno or not self.unconditional(imports[0]):raise ValueError('unverified loader import')
            if sum(isinstance(n,ast.alias) and (n.asname or n.name)==root.id for n in self.nodes)!=1:
                raise ValueError('loader import shadowed')
            if any((isinstance(n,ast.Name) and n.id in (root.id,'open') and isinstance(n.ctx,(ast.Store,ast.Del)))
                   or (isinstance(n,ast.arg) and n.arg in (root.id,'open'))
                   or (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name in (root.id,'open'))
                   or (isinstance(n,ast.alias) and (n.asname or n.name)=='open')
                   or (isinstance(n,ast.Attribute) and isinstance(n.ctx,(ast.Store,ast.Del))
                       and isinstance(n.value,ast.Name) and n.value.id==root.id) for n in self.nodes):raise ValueError('loader rebound')
            opened=node.args[0]
            if not isinstance(opened,ast.Call) or not isinstance(opened.func,ast.Name) or opened.func.id!='open' or not 1<=len(opened.args)<=2:
                raise ValueError('only submitted literal file inputs')
            name=ast.literal_eval(opened.args[0])
            if (not isinstance(name,str) or ':' in name or '\\' in name or name.startswith('/') or '..' in name.split('/')
                    or len(opened.args)==2 and ast.literal_eval(opened.args[1])!='r'
                    or any(k.arg!='encoding' or ast.literal_eval(k.value) not in ('utf-8','utf8') for k in opened.keywords)):
                raise ValueError('unsafe input label')
            # open() resolves against process cwd, not the source file folder.
            # Only bind a literal matching the submitted project-relative label.
            path=name
            if path not in self.files:raise ValueError('configuration file not submitted')
            text=self.files[path];value=parse_data(path,text)
            return value,[{'path':path,'sha256':hashlib.sha256(text.encode()).hexdigest(),'line':1,
                           'end_line':len(text.splitlines()),'snippet':text,'kind':'configuration_file','key_path':[]}]
        raise ValueError('dynamic configuration expression')


def masked_declaration(path,text,location):
    sources=[s for s in location.get('value_sources',[]) if s['path']==path]
    if path.endswith(('.json','.yaml','.yml')):
        value=copy.deepcopy(parse_data(path,text))
        for source in sources:
            keys=source.get('key_path',[])
            if not keys:raise ValueError('missing field identity')
            target=value
            for key in keys[:-1]:target=target[key]
            target[keys[-1]]='__reviewed_threshold__'
        return json.dumps(value,sort_keys=True,ensure_ascii=False)
    tree=ast.parse(text)
    positions={(s['line'],s.get('col'),s['end_line'],s.get('end_col')) for s in sources if s['kind']=='python_scalar'}
    class Mask(ast.NodeTransformer):
        def visit(self,node):
            if (getattr(node,'lineno',None),getattr(node,'col_offset',None),getattr(node,'end_lineno',None),getattr(node,'end_col_offset',None)) in positions:
                return ast.Constant(value='__reviewed_threshold__')
            return super().visit(node)
    return ast.dump(Mask().visit(tree),include_attributes=False)
