"""Bounded subprocess capture for sandbox launchers (no direct candidate execution)."""
import os
import subprocess
import tempfile
import time
from coderlab.contract import RewardInfrastructureError


def run_bounded(command, stdin=None, timeout=30, env=None, cwd=None, max_output=1024*1024):
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, tempfile.TemporaryFile() as inp:
        inp.write((stdin or '').encode('utf-8')); inp.seek(0)
        try:
            proc = subprocess.Popen(command, stdin=inp, stdout=out, stderr=err, env=env, cwd=cwd)
        except OSError as exc:
            raise RewardInfrastructureError('Sandbox launcher unavailable') from exc
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    return -1, '', 'wall_timeout'
                if sum(os.fstat(f.fileno()).st_size for f in (out, err)) > max_output:
                    return -1, '', 'output_limit'
                time.sleep(.02)
            if sum(os.fstat(f.fileno()).st_size for f in (out, err)) > max_output:
                return -1, '', 'output_limit'
            out.seek(0); err.seek(0)
            return proc.returncode, out.read(max_output).decode('utf-8', 'replace'), err.read(max_output).decode('utf-8', 'replace')
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
