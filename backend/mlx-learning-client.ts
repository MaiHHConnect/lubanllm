/** One resident native MLX process; requests and cancellation share the existing JSONL protocol. */
import assert from 'node:assert/strict';
import {spawn,type ChildProcess} from 'node:child_process';
import {openSync,closeSync,constants,fstatSync,lstatSync,mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {dlopen} from 'bun:ffi';
import {cleanRuntimeEnvironment} from './native-process.ts';
import type {MLXSource,MLXRow} from './mlx-learning-source.ts';
import {materialMessages} from './native-evolution-material.ts';
import {evaluateGeneration,validateGenerationCheck} from './generation-checks.ts';
import {materialProvenance} from './native-evolution-material.ts';

export type PreparedMLX={tokens:number[];labels:number[];prompt_tokens:number;eos:number};
export const MLX_CHAT_END_TOKENS=[248044,248046];
export function mlxGenerationComplete(result:MLXRow){return result.finish_reason==='stop'&&Array.isArray(result.tokens)&&result.tokens.length>0&&MLX_CHAT_END_TOKENS.includes(result.tokens.at(-1));}
const flock=dlopen('/usr/lib/libSystem.B.dylib',{flock:{args:['i32','i32'],returns:'i32'}});
export function lockMLXModel(identity:string){
 const root='/tmp/luban-model-locks-'+process.getuid!();mkdirSync(root,{recursive:true,mode:0o700});const ds=lstatSync(root);assert(ds.isDirectory()&&!ds.isSymbolicLink()&&ds.uid===process.getuid!()&&(ds.mode&0o777)===0o700);
 const fd=openSync(join(root,identity+'-mps.lock'),constants.O_CREAT|constants.O_RDWR|constants.O_NOFOLLOW,0o600);try{const st=fstatSync(fd);assert(st.isFile()&&st.nlink===1&&st.uid===process.getuid!()&&(st.mode&0o777)===0o600&&flock.symbols.flock(fd,2|4)===0,'本地模型正被其他任务使用');return()=>closeSync(fd);}catch(e){closeSync(fd);throw e;}
}
export class MLXLearningClient{
 private child?:ChildProcess;private ended?:Promise<void>;private queue:MLXRow[]=[];private waiting?:{resolve:(row:MLXRow)=>void;reject:(e:Error)=>void};private failure?:Error;private serial=0;private inflight=false;
 constructor(readonly source:MLXSource,readonly model:string,readonly manifest:string){}
 private fail(e:Error){this.failure??=e;this.waiting?.reject(this.failure);this.waiting=undefined;}
 private async next():Promise<MLXRow>{if(this.failure)throw this.failure;if(this.queue.length)return this.queue.shift()!;assert(this.child?.exitCode===null&&!this.waiting,'MLX process has ended');let timer:ReturnType<typeof setTimeout>;try{return await new Promise<MLXRow>((resolve,reject)=>{this.waiting={resolve,reject};timer=setTimeout(()=>this.fail(Error('MLX原生学习请求超时')),300000);});}finally{clearTimeout(timer!);}}
 async start(){assert(!this.child);const s=this.source,child=this.child=spawn(s.binary,[this.model,this.manifest,'3',s.coreFile,'model.language_model.layers.23.mlp.base.down_proj.weight',s.core.sha256,'--mlx-q4','--mlx-model='+s.modelPath,'--mlx-model-sha256='+s.bound.mlx_model_sha256,'--expert-file='+s.expertFile,'--expert-sha='+s.expert.sha256],{stdio:['pipe','pipe','pipe','pipe'],env:cleanRuntimeEnvironment()});let pending='',stderr='';
  child.on('error',e=>this.fail(e));child.stdin!.on('error',e=>this.fail(e));child.stderr!.on('data',b=>{stderr=(stderr+b).slice(-8192);});child.stdout!.setEncoding('utf8');child.stdout!.on('data',(chunk:string)=>{try{pending+=chunk;assert(Buffer.byteLength(pending)<4<<20);let at:number;while((at=pending.indexOf('\n'))>=0){const row=JSON.parse(pending.slice(0,at));pending=pending.slice(at+1);assert(row&&typeof row.event==='string');if(row.event==='error'){this.fail(Error(String(row.error)));continue;}if(this.waiting){const w=this.waiting;this.waiting=undefined;w.resolve(row);}else{assert(this.queue.length<4096);this.queue.push(row);}}}catch(e){this.fail(e as Error);}});
  this.ended=new Promise(done=>child.once('close',(code,signal)=>{if(code!==0||signal||this.waiting||pending)this.fail(Error('MLX进程退出：'+(signal??code)+' '+stderr));done();}));const ack=await this.next();
  assert(ack.event==='loaded'&&ack.inference_precision==='mlx-q4'&&ack.mlx_training===false&&ack.mlx_model_sha256===s.bound.mlx_model_sha256&&ack.core_loaded===true&&ack.core_file_sha256===s.core.sha256&&ack.expert_loaded===true&&ack.expert_file_sha256===s.expert.sha256&&ack.expert_width===s.expert.width&&ack.layers===24&&ack.execution_skip_mask===0&&ack.execution_exit_after===24&&ack.layer_paging_enabled===false,'MLX actual loaded source differs');return ack;
 }
 async command(op:string,body:MLXRow={},signal?:AbortSignal){assert(this.child&&!this.inflight);signal?.throwIfAborted();this.inflight=true;const id='learning-'+ ++this.serial;const cancel=()=>{const pipe=this.child?.stdio[3]as import('node:stream').Writable;pipe?.write(Buffer.from([1]));};signal?.addEventListener('abort',cancel,{once:true});
  try{this.child.stdin!.write(JSON.stringify({id,op,...body})+'\n');const tokens:number[]=[];for(;;){const row=await this.next();assert(row.id===id,'MLX response request differs');if(row.event==='token'){tokens.push(row.token);continue;}if(op==='generate')assert(JSON.stringify(tokens)===JSON.stringify(row.tokens),'MLX generated and streamed tokens differ');signal?.throwIfAborted();return row;}}
  finally{signal?.removeEventListener('abort',cancel);this.inflight=false;}
 }
 async prepare(row:MLXRow):Promise<PreparedMLX>{const r=await this.command('prepare',{messages:materialMessages(row),answer:row.answer});assert(r.event==='prepared'&&Array.isArray(r.tokens)&&r.tokens.length>=2&&r.tokens.length<=512&&Array.isArray(r.labels)&&r.labels.length===r.tokens.length&&r.eos===248046,'MLX complete training tokens exceed supported bound');return r as PreparedMLX;}
 async generate(row:MLXRow,source:'core'|'parent'|'candidate',signal?:AbortSignal){const selected=await this.command('train_select',{source},signal);assert(selected.event==='train_selected'&&selected.source===source&&selected.compilation_enabled===false,'MLX学习观察需要当前来源的非编译推理路径');const check=validateGenerationCheck(materialProvenance(row).generation_check,row.answer),p=await this.command('prepare',{messages:materialMessages(row)},signal);const input=p.tokens;assert(Array.isArray(input)&&input.length>0);const result=await this.command('generate',{tokens:input,max_tokens:Number(check.max_new_tokens),eos_ids:MLX_CHAT_END_TOKENS,temperature:0,seed:42,cache_reuse:false},signal);assert(result.event==='result'&&['stop','length'].includes(result.finish_reason)&&Array.isArray(result.tokens));const decoded=await this.command('decode',{tokens:result.tokens},signal);assert(decoded.event==='decoded'&&typeof decoded.text==='string');const answer=decoded.text.trim(),complete=mlxGenerationComplete(result);return{id:row.id,answer,complete,passed:evaluateGeneration(answer,check,complete).passed,tokens:result.tokens,input_tokens:input,finish_reason:result.finish_reason,branch_enabled:source!=='core',reference_used:false,source,thinking:'off',temperature:0,seed:42};}
 async loss(row:MLXRow,source:'core'|'parent'|'candidate',prepared:PreparedMLX,signal?:AbortSignal){const value=await this.command('train_observe',{sample:{tokens:prepared.tokens,labels:prepared.labels},source},signal);assert(value.event==='train_observed'&&Number.isFinite(value.loss)&&value.loss>=0,'MLX loss observation invalid');return{id:row.id,loss:value.loss};}
 async close(){if(!this.child)return;assert(!this.inflight,'MLX request still owns model');this.child.stdin!.end();let timer:ReturnType<typeof setTimeout>;try{await Promise.race([this.ended!,new Promise<never>((_,reject)=>{timer=setTimeout(()=>reject(Error('MLX进程尚未关闭，保留租约')),15000);})]);this.child=undefined;}finally{clearTimeout(timer!);}}
}
