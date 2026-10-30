/** Real SQLite lock ordering and original transaction/CAS semantics; no production ledger. */
import {test,expect} from 'bun:test';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {Memory as OldMemory} from '../.native-migration/research/native-product-runtime/backend/memory';
// Exercise the current source; frozen release bundles are not rebuilt by these regressions.
import {Memory} from '../backend/memory';

async function fixture(parent=tmpdir()){const root=await mkdtemp(join(parent,'luban-memory-lock-')),memory=new Memory(root);memory.connect(db=>{db.exec('CREATE TABLE cp(id TEXT PRIMARY KEY,body TEXT NOT NULL,updated INTEGER NOT NULL)');db.query('INSERT INTO cp VALUES(?,?,?)').run('run','240',1);});return{root,memory,path:join(root,'app/memory.sqlite3'),async close(){memory.close();await rm(root,{recursive:true,force:true});}};}
test('actual old Memory schema readers cause COMMIT SQLITE_BUSY after successful update; rollback preserves durable row',async()=>{
 const h=await fixture(),oldA=new OldMemory(h.root),oldB=new OldMemory(h.root),a=(oldA as any).open(),b=(oldB as any).open();
 try{a.exec('PRAGMA busy_timeout=0');b.exec('PRAGMA busy_timeout=0');a.exec('BEGIN IMMEDIATE');expect(a.query('UPDATE cp SET body=?,updated=? WHERE id=? AND body=? AND updated=?').run('250',2,'run','240',1).changes).toBe(1);expect(()=>a.exec('COMMIT')).toThrow('database is locked');expect(a.inTransaction).toBe(true);a.exec('ROLLBACK');b.close();expect(a.query('SELECT body FROM cp').get()).toEqual({body:'240'});}
 finally{a.close();b.close();oldA.close();oldB.close();await h.close();}
});
async function holdReader(path:string,change:boolean){
 const code=`import{Database}from'bun:sqlite';const db=new Database(process.argv[1]);db.exec('PRAGMA locking_mode=EXCLUSIVE;PRAGMA busy_timeout=2000');${change?"db.exec(\"BEGIN IMMEDIATE;UPDATE cp SET body='rival',updated=2;COMMIT\");":''}db.query('SELECT body FROM cp').all();console.log('reader-held');setTimeout(()=>{db.close();},200);`;
 const child=Bun.spawn([process.execPath,'-e',code,path],{stdout:'pipe',stderr:'pipe'}),reader=child.stdout.getReader();const chunk=await reader.read();expect(new TextDecoder().decode(chunk.value)).toContain('reader-held');return{child,reader};
}
for(const [filesystem,parent] of [['APFS',tmpdir()],['exFAT','/Volumes/移动数据库/lubanllm']] as const)test('new actual Memory '+filesystem+' waits before callback/schema and commits once after competing process releases its reader',async()=>{
 const h=await fixture(parent);let other:any;
 try{other=await holdReader(h.path,false);let calls=0;const start=performance.now();h.memory.connect(db=>{calls++;expect(db.query('UPDATE cp SET body=?,updated=? WHERE id=? AND body=? AND updated=?').run('250',2,'run','240',1).changes).toBe(1);});expect(calls).toBe(1);expect(performance.now()-start).toBeGreaterThan(80);expect(await other.child.exited).toBe(0);expect(h.memory.connect(db=>db.query('SELECT body FROM cp').get())).toEqual({body:'250'});}
 finally{await other?.reader.cancel();await h.close();}
});
test('competing committed row still rejects original full-row CAS; operation errors rollback without replay',async()=>{
 const h=await fixture();let other:any;
 try{other=await holdReader(h.path,true);let calls=0;expect(()=>h.memory.connect(db=>{calls++;const changed=db.query('UPDATE cp SET body=?,updated=? WHERE id=? AND body=? AND updated=?').run('250',2,'run','240',1);if(changed.changes!==1)throw Error('full-row CAS rejected');})).toThrow('full-row CAS rejected');expect(calls).toBe(1);expect(await other.child.exited).toBe(0);expect(h.memory.connect(db=>db.query('SELECT body FROM cp').get())).toEqual({body:'rival'});
  expect(()=>h.memory.connect(db=>{db.exec("UPDATE cp SET body='bad'");throw Error('operation failed');})).toThrow('operation failed');expect(h.memory.connect(db=>db.query('SELECT body FROM cp').get())).toEqual({body:'rival'});
 }finally{await other?.reader.cancel();await h.close();}
});
