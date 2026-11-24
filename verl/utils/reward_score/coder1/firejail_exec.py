"""Modified upstream Firejail adapter: bounded output, time and environment."""
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from coderlab.contract import RewardInfrastructureError
from coderlab.process import run_bounded
from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

CLI_ARG_SIZE_LIMIT = 1024 * 3  # retained for callers


def code_exec_firejail(code, stdin=None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest=None):
    if not shutil.which('firejail'):
        raise RewardInfrastructureError('Firejail is not installed')
    if pytest is not None and stdin is not None:
        raise ValueError('STDIN is not supported with pytest')
    # Exclude API tokens and other inherited application environment values.
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'USER', 'LOGNAME') if k in os.environ}
    env.update(OPENBLAS_NUM_THREADS='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root/'solution.py').write_text(code, encoding='utf-8')
        command = ['firejail', '--private', '--quiet', '--net=none', '--seccomp=socket',
                   '--profile=pip', f'--whitelist={directory}', '--rlimit-nproc=32',
                   '--rlimit-nofile=32', '--rlimit-fsize=2m', '--rlimit-as=4096m',
                   f'--timeout={int(timeout)//3600:02d}:{int(timeout)%3600//60:02d}:{int(timeout)%60:02d}']
        if pytest is not None:
            (root/'test_solution.py').write_text(pytest, encoding='utf-8')
            command += ['python3', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(root/'test_solution.py')]
        else:
            command += ['python3', str(root/'solution.py')]
        rc, stdout, stderr = run_bounded(command, stdin, timeout+3, env=env, cwd=directory)
    if rc == 0:
        return True, stdout
    if rc in (126, 127) or 'Error: ' in stderr and ('firejail' in stderr.lower() or 'namespace' in stderr.lower()):
        raise RewardInfrastructureError('Firejail failed to initialize')
    return False, _ERROR_MSG_PREFIX + stderr[:2048]
