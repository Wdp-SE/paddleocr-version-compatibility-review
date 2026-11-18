"""Offline administrative operations. No destructive restore or credentials in logs."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

from src.internal_tasks import TaskStore


def sqlite_backup(source,target):
    source,target=Path(source).resolve(),Path(target).resolve()
    if not source.is_file() or target.exists():raise ValueError('Source missing or backup exists')
    # Reserve exclusively; never overwrite a file created after the existence check.
    with target.open('xb'):pass
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)
            dst.commit()
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix+'.sha256').write_text(digest,encoding='ascii')
    return digest


def sqlite_restore(source,target):
    source,target=Path(source).resolve(),Path(target).resolve()
    expected=source.with_suffix(source.suffix+'.sha256').read_text(encoding='ascii').strip()
    if target.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
        raise ValueError('Restore requires verified backup and new scratch database')
    # Exclusive creation prevents an existing database from being overwritten by a race.
    with target.open('xb') as output,source.open('rb') as input_stream:shutil.copyfileobj(input_stream,output)
    TaskStore('sqlite:///'+str(target),initialize=False).health()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('migrate')
    for action in ('sqlite-backup','sqlite-restore'):
        item=sub.add_parser(action);item.add_argument('source',type=Path);item.add_argument('target',type=Path)
    release=sub.add_parser('release')
    release.add_argument('source',type=Path);release.add_argument('releases',type=Path);release.add_argument('gate',type=Path)
    release.add_argument('--activate',action='store_true');release.add_argument('--expected',default=None)
    args=parser.parse_args()
    if args.command=='migrate':
        store=TaskStore(os.environ['INTERNAL_DATABASE_URL']);print(store.health());return
    if args.command.startswith('sqlite-'):
        operation=sqlite_backup if args.command=='sqlite-backup' else sqlite_restore
        operation(args.source,args.target);print('Verified scratch operation complete');return
    from src.corpus_release import stage,activate
    from src.build_identity import content_fingerprint
    code=content_fingerprint({p.name:p for p in Path(__file__).parent.glob('*.py')})['fingerprint_sha256']
    name=stage(args.source,args.releases,json.loads(args.gate.read_text(encoding='utf8')),code_identity=code)
    if args.activate:activate(args.releases,args.releases/'active.json',name,expected=args.expected,code_identity=code)
    print(name)


if __name__=='__main__':main()
