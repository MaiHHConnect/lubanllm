/** Six previously frozen natural prompts through the actual app, with no teacher or training. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {homedir} from 'node:os';
import {parseArgs} from 'node:util';
const {values}=parseArgs({args:Bun.argv.slice(2),options:{core:{type:'string'},output:{type:'string'},prompts:{type:'string'}},strict:true});
assert(values.core&&/^core-[a-f0-9]{12}$/.test(values.core)&&values.output&&values.prompts);
const prompts=JSON.parse(await readFile(resolve(values.prompts),'utf8'));
assert(prompts.training_allowed===false&&prompts.teacher_calls_allowed===0&&prompts.cases.length===6);
const launch=JSON.parse(await readFile(join(homedir(),'Library/Application Support/Luban/launch.json'),'utf8'));
const request=async(path:string,body?:unknown,method=body===undefined?'GET':'POST')=>{
 const response=await fetch('http://127.0.0.1:'+launch.port+path,{method,headers:{cookie:'luban_session='+launch.token,'content-type':'application/json'},
  body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(180000),proxy:null});
 assert(response.ok,'Application HTTP '+response.status+' at '+path);return response;
};
const initial:any=await(await request('/api/status')).json();
assert(initial.runtime.state==='ready'&&initial.runtime.core_version===values.core,'Actual application must load the imported core before evaluation');
const config:any=await(await request('/api/settings')).json();
const previous={answer_mode:config.teacher.answer_mode,teaching_modalities:config.teacher.teaching_modalities};
const report:any={schema:'native-identity-final-application-v1',core:values.core,teacher_calls:0,optimizer_steps:0,training_allowed:false,
 started:new Date().toISOString(),engine:initial.runtime.inference,cases:[],settings_restored:false};
try{
 await request('/api/settings',{teacher:{answer_mode:'local_first',teaching_modalities:[]}},'PATCH');
 for(const sample of prompts.cases){
  const started=performance.now();
  const response=await request('/api/chat',{messages:[{role:'user',content:sample.prompt}],route:'local',thinking:'off',temperature:0,max_tokens:128});
  const frames=(await response.text()).split('\n').filter(line=>line.startsWith('data: ')).map(line=>JSON.parse(line.slice(6)));
  assert(!frames.some(row=>row.type==='error'),'Application generation returned an error');
  const meta=frames.find(row=>row.type==='meta'),done=frames.find(row=>row.type==='done');assert(meta&&done);
  const events:any=await(await request('/api/events')).json();
  const decision=events.events.find((row:any)=>row.kind==='decision'&&row.body.request_id===meta.request_id)?.body;
  assert(decision&&decision.decision.route==='local'&&decision.decision.action==='local_core'
   &&decision.model_context.core_version===values.core&&decision.model_context.expert_id===null
   &&decision.usage.reference_used===false,'Final app evidence must be the actual imported core without a reference, expert or teacher');
  const answer=frames.filter(row=>row.type==='delta').map(row=>row.content??'').join('');assert.equal(answer,decision.answer);
  const row={...sample,answer,complete:decision.complete,finish_reason:done.usage.finish_reason,
   model_context:decision.model_context,decision:decision.decision,usage:done.usage,seconds:(performance.now()-started)/1000};
  report.cases.push(row);console.log(JSON.stringify({id:sample.id,answer,complete:row.complete,seconds:row.seconds}));
 }
}catch(error){report.error=String(error);throw error;}
finally{
 try{await request('/api/settings',{teacher:previous},'PATCH');report.settings_restored=true;}
 finally{report.finished=new Date().toISOString();await writeFile(resolve(values.output),JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});}
}
