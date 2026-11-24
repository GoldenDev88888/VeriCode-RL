"""Strict execution-based reward; incomplete or unavailable infrastructure is not a negative label."""
from .providers import extract_code

def execution_reward(grader,task,completion):
    code=extract_code(completion[0]['content'] if isinstance(completion,list) else completion)
    result=grader.grade(task,code)
    if result.status=='infra_error':
        raise RuntimeError('Reward infrastructure failure: abort rather than poison training')
    return float(result.status=='pass')
