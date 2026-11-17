"""Read mounted secrets without shell interpolation, then replace the process."""
import os
from pathlib import Path
import sys

for name in ('INTERNAL_DATABASE_URL','DEEPSEEK_API_KEY','DASHSCOPE_API_KEY'):
    path=os.environ.get(name+'_FILE')
    if path:os.environ[name]=Path(path).read_text(encoding='utf8').strip()
if len(sys.argv)<2:raise SystemExit('A service command is required')
os.execvp(sys.argv[1],sys.argv[1:])
