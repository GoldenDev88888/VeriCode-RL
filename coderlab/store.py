import sqlite3,json,datetime,uuid
from pathlib import Path
class Store:
    def __init__(self,path):
        self.path=str(path); Path(path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, created TEXT, metadata TEXT);
            CREATE TABLE IF NOT EXISTS samples(run_id TEXT,model TEXT,task_id TEXT,sample_index INTEGER,data TEXT,PRIMARY KEY(run_id,model,task_id,sample_index));''')
    def connect(self):
        return sqlite3.connect(self.path,timeout=30)
    def start(self,metadata):
        rid=uuid.uuid4().hex
        with self.connect() as c: c.execute('INSERT INTO runs VALUES (?,?,?)',(rid,datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(metadata)))
        return rid
    def add(self,rid,row):
        with self.connect() as c: c.execute('INSERT INTO samples VALUES (?,?,?,?,?)',(rid,row['model'],row['task_id'],row['sample_index'],json.dumps(row)))
    def runs(self):
        with self.connect() as c: return [{'id':r[0],'created':r[1],**json.loads(r[2])} for r in c.execute('SELECT * FROM runs ORDER BY created DESC')]
    def rows(self,rid):
        with self.connect() as c: return [json.loads(r[0]) for r in c.execute('SELECT data FROM samples WHERE run_id=? ORDER BY model,task_id,sample_index',(rid,))]
