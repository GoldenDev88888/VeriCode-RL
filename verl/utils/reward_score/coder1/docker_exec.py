"""Modified upstream Docker adapter: no shell interpolation or host execution."""
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from coderlab.contract import RewardInfrastructureError
from coderlab.process import run_bounded
from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

CLI_ARG_SIZE_LIMIT = 1024 * 3


def code_exec_docker(code, stdin=None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest=None):
    if not shutil.which('docker'):
        raise RewardInfrastructureError('Docker CLI not installed')
    if pytest is not None and stdin is not None:
        raise ValueError('STDIN is not supported with pytest')
    image = os.getenv('CODER1_DOCKER_IMAGE', 'python:3.11-slim')
    name = 'coder1-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); root.chmod(0o755)
        (root/'solution.py').write_text(code, encoding='utf-8'); (root/'solution.py').chmod(0o644)
        command = ['docker', 'run', '--rm', '--pull=never', '--name', name, '-i',
                   '--network=none', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                   '--pids-limit=32', '--memory=512m', '--memory-swap=512m', '--cpus=1',
                   '--user=65534:65534', '--log-driver=none', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
                   '--ulimit', 'fsize=1048576:1048576', '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
                   '--mount', f'type=bind,source={root.resolve()},target=/task,readonly', '-w', '/task', image]
        if pytest is None:
            command += ['python', '-B', '/task/solution.py']
        else:
            (root/'test_solution.py').write_text(pytest, encoding='utf-8'); (root/'test_solution.py').chmod(0o644)
            command += ['python', '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '/task/test_solution.py']
        try:
            rc, out, err = run_bounded(command, stdin, timeout)
        finally:
            try:
                subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
    if rc in (125, 126, 127) or (pytest is not None and 'No module named pytest' in err):
        raise RewardInfrastructureError('Docker launch/image/pytest unavailable')
    return (True, out) if rc == 0 else (False, _ERROR_MSG_PREFIX + err[:2048])
