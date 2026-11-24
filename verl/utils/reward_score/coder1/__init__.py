"""Code-R1 reward adapter. Modified fork: validated, fail-closed reward pipeline."""
import os
import time
from coderlab.contract import (RewardInfrastructureError, evaluate_completion,
                               extract_completion, record_reward)
from .utils import _ERROR_MSG_PREFIX

CODER1_EXEC = os.environ.get('CODER1_EXEC', 'firejail')
if CODER1_EXEC == 'docker':
    from .docker_exec import code_exec_docker as code_exec
elif CODER1_EXEC == 'firejail':
    from .firejail_exec import code_exec_firejail as code_exec
elif CODER1_EXEC == 'ces':
    from .ces_exec import remote_code_exec_ces as code_exec
elif CODER1_EXEC == 'kira':
    from .kira_exec import remote_code_exec_kira as code_exec
else:
    raise ValueError(f'Unknown CODER1_EXEC: {CODER1_EXEC}')


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def validate_response_structure(processed_str):
    return extract_completion(processed_str) is not None


def try_extract_solution(solution_str):
    import re
    matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.S)
    return matches[-1].strip() if matches else solution_str


def extract_code_from_string(solution_str):
    return extract_completion(solution_str) or ''


def _compute_score(solution_str, ground_truth, extra_info=None, format_reward=0.1, answer_reward=1.):
    start = time.monotonic()
    try:
        score, status, cases = evaluate_completion(solution_str, ground_truth, code_exec,
                                                   format_reward, answer_reward,
                                                   workers=int(os.getenv('CODER1_CASE_WORKERS', '4')))
    except Exception:
        record_reward('infrastructure_error', None, 0, time.monotonic()-start, CODER1_EXEC)
        raise
    record_reward(status, score, cases, time.monotonic()-start, CODER1_EXEC)
    return score, f'Code-R1 reward: {status}, cases={cases}, score={score}'


def compute_score(solution_str, ground_truth, extra_info=None, format_reward=0.1, answer_reward=1.):
    score, message = _compute_score(solution_str, ground_truth, extra_info, format_reward, answer_reward)
    if os.getenv('CODER1_VERBOSE') == '1':
        print(message)
    return score
