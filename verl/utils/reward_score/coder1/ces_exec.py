"""Modified upstream CES adapter with bounded retries."""
import os
import time
import requests
from coderlab.contract import RewardInfrastructureError
from .utils import _DEFAULT_TIMEOUT_SECONDS


def remote_code_exec_ces(code, stdin=None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest=None):
    if pytest is not None:
        raise RewardInfrastructureError('CES backend does not support pytest')
    url = os.getenv('CODER1_CES_URL', 'http://localhost:8000').rstrip('/')
    for attempt in range(3):
        try:
            r = requests.post(url+'/py_exec', json={'code':code,'timeout':timeout,'stdin':stdin}, timeout=(3,timeout+5))
            r.raise_for_status()
            status, output = r.text.split('\n', 1)
            int(status)
            if status != '0' and not output:
                raise ValueError('Ambiguous empty failure')
            return status == '0', output
        except (requests.RequestException, ValueError):
            if attempt < 2: time.sleep(.5 * (attempt+1))
    raise RewardInfrastructureError('CES unavailable or returned ambiguous results after 3 attempts')
