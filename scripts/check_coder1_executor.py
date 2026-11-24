"""Trusted sandbox preflight for the original training adapter. Run from repo root."""
import os
from verl.utils.reward_score.coder1 import code_exec, CODER1_EXEC


def main():
    ok, output = code_exec('print(input())', stdin="preflight ' quotes")
    if not ok or output.strip() != "preflight ' quotes":
        raise RuntimeError('Executor failed stdin round-trip')
    ok, _ = code_exec('raise ValueError("expected preflight failure")')
    if ok:
        raise RuntimeError('Executor failed to distinguish runtime errors')
    if CODER1_EXEC in ('firejail', 'docker'):
        ok, _ = code_exec('while True: pass', timeout=2)
        if ok:
            raise RuntimeError('Executor failed timeout check')
    print('Executor preflight passed:', CODER1_EXEC)


if __name__ == '__main__':
    main()
