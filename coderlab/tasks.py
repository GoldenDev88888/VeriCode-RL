from dataclasses import dataclass
from pathlib import Path
import json, hashlib, re

@dataclass(frozen=True)
class Case:
    args: list
    expected: object

@dataclass(frozen=True)
class Task:
    id: str
    category: str
    split: str
    prompt: str
    entrypoint: str
    reference: str
    cases: tuple[Case, ...]
    mutants: tuple[str, ...]


def load_tasks(path: str | Path) -> list[Task]:
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty JSON array')
    tasks, ids, fingerprints = [], set(), set()
    for row in rows:
        task = Task(**{**row, 'cases': tuple(Case(**c) for c in row['cases']), 'mutants': tuple(row['mutants'])})
        fingerprint = hashlib.sha256(' '.join(task.prompt.lower().split()).encode()).hexdigest()
        if task.id in ids or fingerprint in fingerprints:
            raise ValueError(f'Duplicate id or prompt: {task.id}')
        if task.split not in {'train','validation','test'} or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',task.entrypoint):
            raise ValueError(f'Invalid task: {task.id}')
        if not task.cases or not task.mutants or not all(isinstance(c.args,list) for c in task.cases):
            raise ValueError(f'Missing cases/mutants or invalid args: {task.id}')
        if not task.prompt.strip() or not task.reference.strip():
            raise ValueError(f'Missing prompt/reference: {task.id}')
        ids.add(task.id); fingerprints.add(fingerprint); tasks.append(task)
    return tasks


def dataset_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
