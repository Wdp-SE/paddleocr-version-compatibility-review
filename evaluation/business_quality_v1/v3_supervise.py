"""Bounded subprocess supervisor for the fixed developer runtime probe only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--application-acceptance', action='store_true')
    parser.add_argument('probe_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.name.replace('-', '').replace('_', '').isalnum() or not 1 <= args.timeout <= 1200:
        raise ValueError('invalid bounded probe name/timeout')
    output = ROOT/'.local-demo-temp/resolve-20261114'
    output.mkdir(parents=True, exist_ok=True)
    log_path = output/(args.name+'.log')
    report_path = output/(args.name+'.json')
    if log_path.exists() or report_path.exists():
        raise FileExistsError('probe runs are immutable; choose a new name')
    probe = ROOT/'evaluation/business_quality_v1'/('v3_application_acceptance.py' if args.application_acceptance else 'v3_runtime_probe.py')
    probe_args = args.probe_args[1:] if args.probe_args[:1] == ['--'] else args.probe_args
    if args.application_acceptance:
        if probe_args:
            raise ValueError('fixed application acceptance does not accept arbitrary extra arguments')
        probe_args = ['--output', str(output/(args.name+'.child.json'))]
    command = [str(ROOT/'.local-demo-temp/ocr-v3/Scripts/python.exe'), '-u', str(probe), *probe_args]
    # Preserve the exact diagnostic source alongside its hash, even if later edited.
    output.joinpath(args.name+'.source.py').write_bytes(probe.read_bytes())
    child_env = {**os.environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONUNBUFFERED':'1'}
    started = time.perf_counter()
    report = {'command': command, 'probe_sha256': hashlib.sha256(probe.read_bytes()).hexdigest(),
              'stdout_encoding':'utf-8',
              'environment': {k:v for k,v in os.environ.items() if k.startswith(('FLAGS_', 'OMP_', 'MKL_', 'KMP_'))}}
    with log_path.open('w', encoding='utf-8') as log:
        try:
            child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.timeout, check=False, env=child_env)
            report.update(exit_code=child.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            report.update(exit_code=None, timed_out=True)
    report.update(seconds=round(time.perf_counter()-started, 3), raw_log=str(log_path))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 124 if report['timed_out'] else report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
