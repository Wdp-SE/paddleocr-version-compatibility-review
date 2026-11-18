"""Produce a hash-locked requirements file from pip-resolved, downloaded wheels.
Target: CPython 3.11, Linux amd64. Never derives package names from shell text.
"""
import argparse
from email.parser import Parser
import hashlib
from pathlib import Path
import zipfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('wheel_directory',type=Path);parser.add_argument('output',type=Path)
args=parser.parse_args();rows={}
for path in args.wheel_directory.glob('*.whl'):
    with zipfile.ZipFile(path) as archive:
        metadata=Parser().parsestr(archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))).decode())
    name,version=metadata['Name'],metadata['Version']
    if name in rows:raise ValueError('Multiple versions in wheel directory')
    rows[name]=f'{name}=={version} --hash=sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}'
if len(rows)<10:raise ValueError('Incomplete dependency download')
args.output.write_text('# CPython 3.11 Linux amd64; generated from resolved wheels; install with --require-hashes.\n'+
    '\n'.join(rows[n] for n in sorted(rows,key=str.casefold))+'\n',encoding='utf8')
print(f'Locked {len(rows)} resolved wheels')
