"""Execute one case per isolated container; expected answers never enter it.
Docker reduces risk but is not a security boundary for hostile multi-tenant use.
Use an isolated disposable VM for untrusted models. No host execution fallback.
"""
import json, subprocess, uuid, shutil
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class Result:
    status: str
    passed: int
    total: int
    details: tuple[str, ...] = ()

class Grader(Protocol):
    def grade(self, task, code: str) -> Result: ...

WORKER = '''import json,sys,contextlib,os
p=json.load(sys.stdin)
ns={}
with open(os.devnull,'w') as sink, contextlib.redirect_stdout(sink):
    exec(compile(p['code'],'candidate.py','exec'),ns)
    result=ns[p['entrypoint']](*p['args'])
print(json.dumps(result,allow_nan=False,separators=(',',':')))
'''

class DockerGrader:
    def __init__(self, image='python:3.11-slim', timeout=8):
        if not shutil.which('docker'):
            raise RuntimeError('Docker is required; no unsafe host fallback is available')
        self.image, self.timeout = image, timeout
        subprocess.run(['docker','image','inspect',image],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def grade(self, task, code):
        if len(code.encode()) > 32000:
            return Result('invalid',0,len(task.cases),('candidate too large',))
        passed, details = 0, []
        for case in task.cases:
            name = 'coderlab-'+uuid.uuid4().hex
            cmd=['docker','run','--rm','--name',name,'--pull=never','-i','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=128m','--memory-swap=128m','--cpus=0.5','--user=65534:65534','--ulimit','fsize=1048576:1048576','--tmpfs','/tmp:rw,noexec,nosuid,size=8m','--log-driver=none',self.image,'python','-I','-c',WORKER]
            payload=json.dumps({'code':code,'entrypoint':task.entrypoint,'args':case.args})
            # Bound output in a temporary host file rather than buffering arbitrary stdout in RAM.
            import tempfile, time, os
            try:
                with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as inp:
                    inp.write(payload.encode()); inp.seek(0)
                    p=subprocess.Popen(cmd,stdin=inp,stdout=out,stderr=subprocess.DEVNULL)
                    try:
                        deadline=time.monotonic()+self.timeout
                        while p.poll() is None:
                            if os.fstat(out.fileno()).st_size > 65536:
                                details.append('output_limit'); break
                            if time.monotonic() > deadline:
                                details.append('timeout'); break
                            time.sleep(.05)
                        else:
                            if p.returncode in (125,126,127):
                                return Result('infra_error',passed,len(task.cases),tuple(details+['docker execution failed']))
                            if p.returncode != 0: details.append('runtime_error')
                            elif os.fstat(out.fileno()).st_size > 65536: details.append('output_limit')
                            else:
                                out.seek(0)
                                try:
                                    value=json.loads(out.read(65537))
                                    ok=json.dumps(value,sort_keys=True,allow_nan=False)==json.dumps(case.expected,sort_keys=True,allow_nan=False)
                                except (ValueError,TypeError): ok=False
                                passed+=int(ok); details.append('pass' if ok else 'wrong_answer')
                    finally:
                        if p.poll() is None:
                            p.kill(); p.wait()
            except (OSError,subprocess.SubprocessError):
                return Result('infra_error',passed,len(task.cases),tuple(details+['runner unavailable']))
            finally:
                try: subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
                except (OSError,subprocess.SubprocessError): pass
        status='pass' if passed==len(task.cases) else 'fail'
        return Result(status,passed,len(task.cases),tuple(details))

class FixtureGrader:
    """Deterministic offline fixture, never executes code or measures model skill."""
    def grade(self,task,code):
        count=len(task.cases)
        if code==task.reference: return Result('pass',count,count,('fixture',)*count)
        return Result('fail',0,count,('fixture',)*count)
