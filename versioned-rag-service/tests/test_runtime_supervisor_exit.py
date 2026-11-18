"""A failed/expired runtime probe must also fail the supervisor command."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('expired,expected', [(False,7),(True,124)])
def test_runtime_probe_failure_is_not_reported_as_cli_success(tmp_path,monkeypatch,expired,expected):
    source=Path(__file__).resolve().parents[2]/'evaluation/business_quality_v1/v3_supervise.py'
    spec=importlib.util.spec_from_file_location('runtime_supervisor_test',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    probe=tmp_path/'evaluation/business_quality_v1/v3_runtime_probe.py'
    probe.parent.mkdir(parents=True)
    probe.write_text('raise SystemExit(7)',encoding='utf8')
    def run(command,**kwargs):
        if expired:raise subprocess.TimeoutExpired(command,1)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(module.subprocess,'run',run)
    assert module.main(['--name','failure','--timeout','1'])==expected
    record=json.loads((tmp_path/'.local-demo-temp/resolve-20261114/failure.json').read_text(encoding='utf8'))
    assert record['timed_out'] is expired
    assert record['exit_code']==(None if expired else 7)
