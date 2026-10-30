/** Installed local model: both protocols and streams, with harmless client tools. */
import {readFileSync,writeFileSync,mkdtempSync,statSync} from 'node:fs';
import {homedir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {Database} from 'bun:sqlite';
const [sdkDirectory,parent] = process.argv.slice(2);
const volume='/Volumes/移动数据库';
if(!sdkDirectory || !parent || !resolve(parent).startsWith(volume+'/') || statSync(volume).dev===statSync('/Volumes').dev) throw Error('Expected SDK and mounted experiment parent');
const home=join(homedir(),'Library/Application Support/Luban');
const settings=JSON.parse(readFileSync(join(home,'settings.json'),'utf8'));
const keys=JSON.parse(readFileSync(join(home,'credentials.json'),'utf8'));
const launch=JSON.parse(readFileSync(join(home,'launch.json'),'utf8'));
const base=`http://127.0.0.1:${launch.port}`;
const db=new Database(join(settings.storage_root,'app/memory.sqlite3'),{readonly:true});
const snapshot=()=>({samples:db.query('SELECT count(*) AS n FROM samples').get(),materials:db.query('SELECT count(*) AS n FROM teacher_materials').get(),teacher:db.query("SELECT * FROM usage WHERE route='teacher'").all(),decisions:db.query("SELECT count(*) AS n FROM events WHERE kind='decision'").get(),practice:db.query('SELECT count(*) AS n FROM teacher_practice').get()});
const before=snapshot();
const dir=mkdtempSync(join(resolve(parent),'tools-protocol-20261026-'));
const result:any={schema:'luban-tools-api-smoke-v1',date:'2026-10-26',status:'running',raw_directory:dir,
 precision:settings.native_inference_precision,seed:'API has no seed parameter; temperature=0',split:'Synthetic protocol smoke, no training or evaluation split',
 memory_measurement:'Not measured; no performance claim',client_tool:'Synthetic lookup returning 26; no shell, file or network tool execution',cases:[]};
const save=()=>writeFileSync(join(dir,'result.json'),[keys.access_key,keys.ui_token,keys.teacher_key,launch.token].filter(Boolean).reduce((s,k)=>s.replaceAll(k,'[redacted]'),JSON.stringify(result,null,2))+'\n');
const must=(ok:any,message:string)=>{if(!ok)throw Error(message);};
const system='你是工具助手。查询温度必须使用lookup工具。收到工具结果后，只回答温度数字，不要再次调用工具。';
const prompt='使用lookup工具查询上海的温度。';
const schema={type:'object',properties:{city:{type:'string',enum:['上海']}},required:['city'],additionalProperties:false};
const tools=[{type:'function',function:{name:'lookup',description:'查询指定城市温度，必须通过此工具取得真实结果。',parameters:schema}}];
async function openai(messages:any[],choice:any,stream:boolean){
 const res=await fetch(base+'/v1/chat/completions',{method:'POST',headers:{authorization:'Bearer '+keys.access_key,'content-type':'application/json'},body:JSON.stringify({model:'luban0',messages,tools,tool_choice:choice,parallel_tool_calls:false,temperature:0,thinking:'off',max_tokens:256,stream}),signal:AbortSignal.timeout(45000)});
 must(res.ok,`OpenAI status ${res.status}: ${await (!res.ok?res.text():Promise.resolve(''))}`);
 if(!stream)return await res.json() as any;
 const frames=(await res.text()).split(/\r?\n\r?\n/).flatMap(f=>f.split(/\r?\n/).filter(x=>x.startsWith('data: ')).map(x=>x.slice(6))).filter(x=>x!=='[DONE]').map(x=>JSON.parse(x));
 const calls:any[]=[];let text='',finish='',usage:any;
 for(const f of frames){must(!f.error,JSON.stringify(f.error));const c=f.choices?.[0];text+=c?.delta?.content??'';
 for(const call of c?.delta?.tool_calls??[])calls[call.index]={id:call.id,type:call.type,function:call.function};if(c?.finish_reason)finish=c.finish_reason;if(f.usage)usage=f.usage;}
 return {choices:[{message:{role:'assistant',content:text||null,...(calls.length?{tool_calls:calls}:{})},finish_reason:finish}],usage,frame_count:frames.length};
}
const start=performance.now();
try{
 result.health=await (await fetch(base+'/api/health')).json();must(result.health.state==='ready','Model must be ready');
 const {default:Anthropic}=await import(pathToFileURL(join(resolve(sdkDirectory),'index.mjs')).href);
 result.sdk_version=JSON.parse(readFileSync(join(sdkDirectory,'package.json'),'utf8')).version;
 const client=new Anthropic({baseURL:base,apiKey:keys.access_key,maxRetries:0,timeout:45000});
 for(const stream of [false,true]){
  const messages:any[]=[{role:'system',content:system},{role:'user',content:prompt}];
  const first=await openai(messages,stream?{type:'function',function:{name:'lookup'}}:'required',stream);
  result.cases.push({protocol:'openai',stream,first});save();
  const calls=first.choices[0].message.tool_calls;must(first.choices[0].finish_reason==='tool_calls'&&calls?.length===1,'Expected one OpenAI tool request');
  const call=calls[0];must(call.function.name==='lookup'&&JSON.parse(call.function.arguments).city==='上海','Wrong OpenAI tool arguments');
  const final=await openai([...messages,first.choices[0].message,{role:'tool',tool_call_id:call.id,content:'26'}],'none',stream);
  result.cases.at(-1).final=final;must(final.choices[0].finish_reason==='stop'&&final.choices[0].message.content?.includes('26'),'OpenAI final answer did not use tool result');save();
 }
 for(const stream of [false,true]){
  const params={model:'luban0',max_tokens:256,temperature:0,system:[{type:'text',text:system,cache_control:{type:'ephemeral'}}],tools:[{name:'lookup',description:tools[0].function.description,input_schema:schema,cache_control:{type:'ephemeral'},eager_input_streaming:true}],messages:[{role:'user',content:[{type:'text',text:prompt,cache_control:{type:'ephemeral'}}]}],tool_choice:{type:stream?'tool':'any',...(stream?{name:'lookup'}:{}),disable_parallel_tool_use:true}};
  const first=stream?await client.messages.stream(params).finalMessage():await client.messages.create(params);
  result.cases.push({protocol:'claude-sdk',stream,first});save();
  const calls=first.content.filter((x:any)=>x.type==='tool_use');must(first.stop_reason==='tool_use'&&calls.length===1,'Expected one Claude tool request');
  const call=calls[0];must(call.name==='lookup'&&call.input.city==='上海','Wrong Claude tool arguments');
  const next={...params,tool_choice:{type:'none'},messages:[...params.messages,{role:'assistant',content:first.content},{role:'user',content:[{type:'tool_result',tool_use_id:call.id,content:'26'}]}]};
  const final=stream?await client.messages.stream(next).finalMessage():await client.messages.create(next);
  result.cases.at(-1).final=final;must(final.stop_reason==='end_turn'&&final.content.some((x:any)=>x.type==='text'&&x.text.includes('26')),'Claude final answer did not use tool result');save();
 }
 result.after=snapshot();result.before=before;must(JSON.stringify(before)===JSON.stringify(result.after),'Tool smoke changed samples/materials/teacher usage');
 result.status='passed';result.model_generations=8;result.new_training_samples=0;result.teacher_requests=0;result.adam_updates=0;
}catch(error){result.status='failed';result.error=String(error);throw error;}
finally{result.elapsed_seconds=(performance.now()-start)/1000;save();db.close();console.log(JSON.stringify({status:result.status,raw_directory:dir,cases:result.cases.length,error:result.error}));}
