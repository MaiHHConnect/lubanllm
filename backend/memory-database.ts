import type {Database} from 'bun:sqlite';

/** Acquire the writer lock before schema reads can retain a SHARED lock. */
export function initializeMemoryDatabase(db:Database,readonly:boolean,schema:string){
 db.exec('PRAGMA busy_timeout=20000; PRAGMA locking_mode=EXCLUSIVE;');
 if(!readonly){
  db.exec('BEGIN EXCLUSIVE');
  try{db.exec(schema);db.exec('COMMIT');}
  catch(error){if(db.inTransaction)db.exec('ROLLBACK');throw error;}
 }
}
