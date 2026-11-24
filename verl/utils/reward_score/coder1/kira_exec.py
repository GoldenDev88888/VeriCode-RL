"""Modified upstream Kira adapter with bounded retries."""
import os
import time
import requests
from coderlab.contract import RewardInfrastructureError
from .utils import _DEFAULT_TIMEOUT_SECONDS


def remote_code_exec_kira(code, stdin=None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest=None):
    if pytest is not None:
        raise RewardInfrastructureError('Kira backend does not support pytest')
    url = os.getenv('CODER1_KIRA_URL', 'http://localhost:9090').rstrip('/')
    for attempt in range(3):
        try:
            r = requests.post(url+'/execute', json={'language':'python','content':code,'tests':[{'stdin':stdin}]}, timeout=(3,timeout+5))
            r.raise_for_status()
            result = r.json()['testOutput']['results'][0]
            stdout, stderr = result['received'], result['runError']
            if not isinstance(stdout, str) or not isinstance(stderr, str): raise ValueError('Malformed result')
            return (False, stderr) if stderr else (True, stdout)
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
            if attempt < 2: time.sleep(.5 * (attempt+1))
    raise RewardInfrastructureError('Kira unavailable or malformed response after 3 attempts')
