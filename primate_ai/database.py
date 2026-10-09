from __future__ import annotations
import sqlite3, json, time
from pathlib import Path
from typing import Optional, Iterable

SCHEMA = '''
CREATE TABLE IF NOT EXISTS projects(
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 name TEXT NOT NULL,
 created_at REAL NOT NULL,
 mode TEXT NOT NULL,
 before_path TEXT,
 after_path TEXT,
 single_path TEXT,
 roi_path TEXT,
 output_dir TEXT,
 status TEXT NOT NULL DEFAULT 'draft'
);
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 project_id INTEGER NOT NULL,
 event_index INTEGER,
 start_s REAL,
 end_s REAL,
 peak_s REAL,
 score REAL,
 region TEXT,
 label TEXT,
 review TEXT DEFAULT 'unreviewed',
 note TEXT DEFAULT '',
 FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE IF NOT EXISTS app_settings(
 key TEXT PRIMARY KEY,
 value TEXT NOT NULL
);
'''

class Database:
    def __init__(self, path):
        self.path=Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c: c.executescript(SCHEMA)
    def connect(self):
        c=sqlite3.connect(self.path); c.row_factory=sqlite3.Row; return c
    def create_project(self, name:str, mode:str, before_path='', after_path='', single_path='', roi_path='', output_dir='') -> int:
        with self.connect() as c:
            cur=c.execute('INSERT INTO projects(name,created_at,mode,before_path,after_path,single_path,roi_path,output_dir,status) VALUES(?,?,?,?,?,?,?,?,?)',
                          (name,time.time(),mode,before_path,after_path,single_path,roi_path,output_dir,'queued'))
            return int(cur.lastrowid)
    def set_project_status(self, pid:int, status:str):
        with self.connect() as c: c.execute('UPDATE projects SET status=? WHERE id=?',(status,pid))
    def update_project(self,pid:int,**kw):
        allowed={'name','mode','before_path','after_path','single_path','roi_path','output_dir','status'}
        items=[(k,v) for k,v in kw.items() if k in allowed]
        if not items:return
        q='UPDATE projects SET '+','.join(f'{k}=?' for k,_ in items)+' WHERE id=?'
        with self.connect() as c:c.execute(q,[v for _,v in items]+[pid])
    def list_projects(self, limit=200):
        with self.connect() as c:return c.execute('SELECT * FROM projects ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
    def replace_events(self,pid:int, rows:Iterable[dict]):
        with self.connect() as c:
            c.execute('DELETE FROM events WHERE project_id=?',(pid,))
            for i,r in enumerate(rows,1):
                c.execute('''INSERT INTO events(project_id,event_index,start_s,end_s,peak_s,score,region,label)
                           VALUES(?,?,?,?,?,?,?,?)''',(pid,int(r.get('event_id',r.get('event',i))),float(r.get('start_s',0)),float(r.get('end_s',0)),float(r.get('peak_s',r.get('peak_time_s',0))),float(r.get('peak_score',r.get('score',0))),str(r.get('primary_region',r.get('region',''))),str(r.get('candidate_label',r.get('label',r.get('explanation',''))))))
    def list_events(self,pid:int):
        with self.connect() as c:return c.execute('SELECT * FROM events WHERE project_id=? ORDER BY start_s',(pid,)).fetchall()
    def review_event(self,event_id:int,review:str,note:str=''):
        with self.connect() as c:c.execute('UPDATE events SET review=?,note=? WHERE id=?',(review,note,event_id))
    def get_setting(self,key:str,default=None):
        with self.connect() as c:
            r=c.execute('SELECT value FROM app_settings WHERE key=?',(key,)).fetchone()
        if not r:return default
        try:return json.loads(r['value'])
        except Exception:return r['value']
    def set_setting(self,key:str,value):
        v=json.dumps(value,ensure_ascii=False)
        with self.connect() as c:c.execute('INSERT INTO app_settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,v))
