import os,pytest
from coderlab.sandbox import DockerGrader
from coderlab.tasks import load_tasks
pytestmark=pytest.mark.skipif(os.getenv('CODERLAB_DOCKER_TEST')!='1',reason='requires Docker and pre-pulled image')
def test_docker_reference_and_mutation():
    t=load_tasks('benchmarks/tasks.json')[0];g=DockerGrader();assert g.grade(t,t.reference).status=='pass';assert g.grade(t,t.mutants[0]).status=='fail'
def test_docker_timeout():
    t=load_tasks('benchmarks/tasks.json')[0];g=DockerGrader(timeout=2);r=g.grade(t,'while True: pass');assert r.status=='fail' and 'timeout' in r.details
