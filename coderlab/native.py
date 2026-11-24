"""Bridge custom tasks to the original verl/Code-R1 RL dataset schema."""
import hashlib
import json
from pathlib import Path
from .contract import validate_oracle

SYSTEM = ('Solve the programming problem in Python. Return reasoning inside <think>...</think> '
          'and one Python markdown code block inside <answer>...</answer>.')


def task_row(task, index):
    # Same functional oracle form consumed by the upstream reward adapter.
    # This shares a process with candidate code; use the separate evaluation harness
    # for stronger oracle separation, not hostile-model training without further hardening.
    lines = []
    for case in task.cases:
        lines.append('assert '+task.entrypoint+'(*'+repr(case.args)+') == '+repr(case.expected))
    oracle = json.dumps({'functional': '\n'.join(lines)})
    validate_oracle(oracle)
    return {'data_source':'code', 'ability':'coding',
            'prompt':[{'role':'system','content':SYSTEM},{'role':'user','content':task.prompt}],
            'reward_model':{'style':'rule','ground_truth':oracle},
            'extra_info':{'split':task.split,'index':index,'prompt':task.prompt,
                          'reference':task.reference,'dataset':'coderlab-custom','task_id':task.id}}


def export_native(tasks, directory, parquet=False):
    if parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError('Install pyarrow to export native Parquet files') from exc
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'code-r1/verl','format':'parquet' if parquet else 'jsonl','splits':{}}
    for split in ('train','validation','test'):
        rows=[task_row(t,i) for i,t in enumerate(tasks) if t.split==split]
        if not rows: raise ValueError('Empty split: '+split)
        path=root/(split+('.parquet' if parquet else '.jsonl'))
        if parquet: pq.write_table(pa.Table.from_pylist(rows),path)
        else: path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        manifest['splits'][split]={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'file':path.name}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def audit_native(paths):
    seen={}; report=[]; invalid=0
    for path in paths:
        path=Path(path); count=0
        if path.suffix=='.parquet':
            import pyarrow.parquet as pq
            batches=(batch.to_pylist() for batch in pq.ParquetFile(path).iter_batches(batch_size=512))
        else:
            batches=([json.loads(line)] for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
        for batch in batches:
            for row in batch:
                count+=1
                try:
                    validate_oracle(row['reward_model']['ground_truth'])
                    if row['data_source'] != 'code': raise ValueError('Unexpected data source')
                    prompt=' '.join(m['content'] for m in row['prompt'] if m['role']=='user')
                    if not prompt.strip(): raise ValueError('Empty prompt')
                    key=hashlib.sha256(' '.join(prompt.lower().split()).encode()).hexdigest()
                    if key in seen:
                        raise ValueError('Duplicate normalized prompt; first in '+seen[key])
                    seen[key]=str(path)
                except (ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
                    invalid+=1; report.append({'file':str(path),'row':count,'issue':str(exc)})
        if not count: invalid+=1; report.append({'file':str(path),'issue':'Empty dataset'})
    return {'valid':invalid==0,'invalid':invalid,'unique_prompts':len(seen),'issues':report}
