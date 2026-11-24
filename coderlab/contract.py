"""Shared Code-R1 oracle contract. Modified fork: 2026-11-24, Apache-2.0."""
import json
import re
import secrets
import os
import time
from concurrent.futures import ThreadPoolExecutor


class RewardInfrastructureError(RuntimeError):
    """An unreliable evaluation must not become a training label."""


class InvalidOracle(RewardInfrastructureError):
    pass


def validate_oracle(raw):
    try:
        oracle = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as exc:
        raise InvalidOracle('Invalid oracle JSON') from exc
    if not isinstance(oracle, dict):
        raise InvalidOracle('Oracle must be an object')
    forms = [key for key in ('functional', 'pytest', 'inputs') if key in oracle]
    if len(forms) != 1 or ('outputs' in oracle) != ('inputs' in oracle):
        raise InvalidOracle('Expected exactly one supported oracle form')
    if forms[0] == 'inputs':
        ins, outs = oracle['inputs'], oracle['outputs']
        if not isinstance(ins, list) or not isinstance(outs, list) or not ins or len(ins) != len(outs):
            raise InvalidOracle('Input/output tests must be nonempty and equally sized')
        if not all(isinstance(v, str) for v in ins + outs):
            raise InvalidOracle('Input/output tests must contain strings')
    elif not isinstance(oracle[forms[0]], str) or not oracle[forms[0]].strip():
        raise InvalidOracle('Empty or invalid test code')
    return oracle


_RESPONSE = re.compile(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>', re.S)
_CODE = re.compile(r'```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```', re.S | re.I)


def extract_completion(text):
    match = _RESPONSE.fullmatch(text.strip())
    if not match:
        return None
    # Accept one Python solution, never reasoning snippets or concatenated alternatives.
    blocks = _CODE.findall(match.group(2))
    if len(blocks) != 1 or not blocks[0].strip():
        return None
    return blocks[0].strip()


def evaluate_completion(text, raw_oracle, execute, format_reward=0.1, answer_reward=1.0, workers=8):
    """Returns (score, status, case count). Backend exceptions propagate to trainer.

    Functional tests retain upstream's shared-process semantics; the marker catches
    accidental/premature exits, not malicious introspection of the test program.
    """
    oracle = validate_oracle(raw_oracle)
    code = extract_completion(text)
    if code is None or len(code.encode('utf-8')) > 128 * 1024:
        return -answer_reward-format_reward, 'invalid_format', 0
    if 'functional' in oracle:
        marker = 'CODER1_DONE_' + secrets.token_hex(16)
        ok, output = execute(code + '\n' + oracle['functional'] + '\nprint(' + repr(marker) + ')\n')
        ok = ok and bool(output.splitlines()) and output.splitlines()[-1] == marker
        count = 1
    elif 'pytest' in oracle:
        ok, _ = execute(code, pytest=oracle['pytest'])
        count = 1
    else:
        pairs = list(zip(oracle['inputs'], oracle['outputs']))
        def check(pair):
            stdin, expected = pair
            success, actual = execute(code, stdin=stdin)
            return success and actual.strip() == expected.strip()
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pairs)))) as pool:
            results = list(pool.map(check, pairs))
        ok, count = all(results), len(results)
    return (format_reward + answer_reward if ok else format_reward), ('pass' if ok else 'fail'), count


def record_reward(status, score, cases, elapsed, backend):
    path = os.environ.get('CODER1_REWARD_LOG')
    if not path:
        return
    record = json.dumps({'timestamp': time.time(), 'status': status, 'score': score,
                         'cases': cases, 'elapsed_ms': round(elapsed*1000, 2), 'backend': backend}) + '\n'
    # Single append call: no candidate source, oracle, prompt, or credentials in telemetry.
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, record.encode('utf-8'))
    finally:
        os.close(fd)
