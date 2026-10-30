import {test,expect,afterAll} from 'bun:test';
import {readFileSync,writeFileSync,openSync,closeSync,rmSync,readdirSync,mkdtempSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {spawn,spawnSync} from 'node:child_process';
import {fixture,sha,STATE} from '../.native-migration/research/native-mps-training/fixture.ts';
import {cleanRuntimeEnvironment} from '../.native-migration/backend/native-process.ts';
import {validateNativeEvolutionTerminalConversion} from '../backend/native-evolution-conversion.ts';
import {patchNativeEvolutionTrainerConversion} from '../scripts/native-evolution-trainer-conversion.ts';
const root=resolve(import.meta.dir,'..'),out=join(root,'build/native-evolution-conversion'),build=JSON.parse(readFileSync(join(out,'build.json'),'utf8'));
const scratch=mkdtempSync(join(tmpdir(),'luban-conversion-cpu-')),fixtures:ReturnType<typeof fixture>[]=[],executions:any[]=[];
const runSelf=spawnSync(build.cpu,['--self-test',join(scratch,'kernel')],{encoding:'utf8',timeout:30000});
if(runSelf.status!==0)throw Error(runSelf.stderr);const self=JSON.parse(runSelf.stdout),kernel=join(scratch,'kernel');
let unchanged:any,resumed=false,boundary=false;
function make(steps:number,offset=0,previous?:string){const f=fixture('cpu',steps,offset,previous);fixtures.push(f);return f;}
function widened(f:ReturnType<typeof fixture>,previous?:string){
 function put(name:string,dtype:string,shape:number[],raw:Buffer){writeFileSync(join(f.input,name+'.raw'),raw,{mode:0o600});f.specs[name]={name,dtype,shape,file:name+'.raw',bytes:raw.length,sha256:sha(raw)};}
 for(const name of STATE)put(name,name==='candidate'?'bf16':'f32',name==='adam_step'?[]:name==='basis'?[160,1]:[3,160],readFileSync(join(previous??kernel,(previous?'final_':'')+name+'.raw')));
 for(const [name,shape]of [['norm',[3]],['head',[8,3]]]as const)put(name,'bf16',[...shape],readFileSync(join(kernel,name+'.raw')));
 for(const row of ['r000','r001'])for(const [suffix,shape]of [['x',[1,3,160]],['residual',[1,3,3]]]as const)put(row+'_'+suffix,'bf16',[...shape],readFileSync(join(kernel,suffix+'.raw')));
 put('constraint_inputs','f32',[8,160],readFileSync(join(kernel,'constraint_inputs.raw')));
}
async function run(f:ReturnType<typeof fixture>,binary:string,conversion=false,mode='fresh'){
 f.manifest(3,mode);let text=readFileSync(join(f.input,'manifest.tsv'),'utf8');if(conversion)text=text.replace('LUBAN_NATIVE_TRAINING_V3\n','LUBAN_NATIVE_TRAINING_V3\nconfig\tfinal_conversion\tgptq-bf16-refine-v1\nconfig\tconstraint_old_positions\t3\n');
 writeFileSync(join(f.input,'manifest.tsv'),text);const cpu=openSync(join(f.input,'cpu_rng.raw'),'r');
 const child=spawn(binary,['--input',f.input,'--output',f.output,'--manifest-sha',sha(text)],{detached:true,env:{...cleanRuntimeEnvironment(),LUBAN_MODEL_LOCK_FD:'3',LUBAN_CPU_RNG_SHA256:f.specs.cpu_rng!.sha256},stdio:['ignore','pipe','pipe',f.fd,cpu]});
 let stdout='',stderr='';child.stdout!.on('data',v=>stdout+=v);child.stderr!.on('data',v=>stderr+=v);
 const timer=setTimeout(()=>{try{process.kill(-child.pid!,'SIGKILL');}catch{}},30000);
 try{const code=await new Promise<number|null>((res,rej)=>{child.once('error',rej);child.once('close',res);});executions.push({binary,code,pid:child.pid,offset:f.offset,steps:f.steps,conversion,stderr});expect(code,stderr).toBe(0);const status=JSON.parse(readFileSync(join(f.output,'rng_status.json'),'utf8'));expect(status.parent_restored).toBe(true);return JSON.parse(stdout);}finally{clearTimeout(timer);closeSync(cpu);}
}
test('overlay inserts only terminal conversion before checkpoint save and exactly restores parent',()=>{
 const source=join(root,'.native-migration/research/native-mps-training/Training.v3.cpp'),original=readFileSync(source,'utf8');
 const patch=patchNativeEvolutionTrainerConversion(original,source,join(root,'native/EvolutionConversion.h'));
 expect(patch.inverse_exact).toBe(true);expect(readFileSync(join(out,'EvolutionTraining.cpp'),'utf8')).toBe(patch.body);
 const call=patch.body.indexOf('luban_conversion::applyAt450');expect(call).toBeGreaterThan(patch.body.indexOf('nativeAdam(master,avg,sq,stateStep,learningRate);'));expect(call).toBeLessThan(patch.body.indexOf('bool checkpoint=checkpoints.count(step.index);'));
 expect(patch.body).toContain('if(finalConversion&&stateStep==450)');
 expect(self.terminal_boundary&&self.resume_bit_exact&&self.cancel_before_commit&&self.rng_unchanged).toBe(true);expect(self.cancellation_cases).toBe(3);
});
test('actual unchanged trainer two-step all 37 math/RNG raw outputs match prior binary',async()=>{
 const old=make(2),current=make(2);await run(old,build.original_trainer);const report=await run(current,build.binary);expect(report.terminal_conversion).toBe(null);
 const names=readdirSync(old.output).filter(n=>n.endsWith('.raw')&&!n.startsWith('parent_'));
 for(const name of names)expect(readFileSync(join(current.output,name)).equals(readFileSync(join(old.output,name))),name).toBe(true);
 expect(names.length).toBe(37);unchanged={actual_steps_each_arm:2,all_seven_state_bit_exact:true,deterministic_raw_equal:names.length};
},30000);
test('actual448-450 agrees with saved449 then resumed450, conversion only at final checkpoint',async()=>{
 const full=make(2,448),first=make(1,448);widened(full);widened(first);
 const a=await run(full,build.binary,true),b=await run(first,build.binary,true);expect(b.terminal_conversion).toBe(null);
 const second=make(1,449,first.output);widened(second,first.output);const c=await run(second,build.binary,true,'resume');
 expect(a.terminal_conversion.absolute_step).toBe(450);expect(a.terminal_conversion.passes).toBe(2);expect(a.terminal_conversion.eligible).toBe(true);expect(c.terminal_conversion).toEqual(a.terminal_conversion);
 for(const name of [...STATE,'cpu_rng'])expect(readFileSync(join(full.output,'final_'+name+'.raw')).equals(readFileSync(join(second.output,'final_'+name+'.raw'))),name).toBe(true);
 const input={config:{final_conversion:'gptq-bf16-refine-v1',constraint_old_positions:'3',training_offset:'448'},tensors:full.specs,steps:[0,1]};
 expect(validateNativeEvolutionTerminalConversion(a,input)).toEqual(a.terminal_conversion);
 expect(()=>validateNativeEvolutionTerminalConversion({...a,terminal_conversion:null},input)).toThrow();
 const bad=structuredClone(a);bad.terminal_conversion.rounds[2].known_old_frobenius=bad.terminal_conversion.rounds[1].known_old_frobenius*2;
 expect(()=>validateNativeEvolutionTerminalConversion(bad,input)).toThrow();
 expect(()=>validateNativeEvolutionTerminalConversion(a,{...input,steps:[0]})).toThrow();
 expect(readFileSync(join(full.output,'checkpoint000001_candidate.raw')).equals(readFileSync(join(first.output,'final_candidate.raw')))).toBe(true);
 expect(readFileSync(join(full.output,'checkpoint000002_candidate.raw')).equals(readFileSync(join(full.output,'final_candidate.raw')))).toBe(true);
 expect(readFileSync(join(full.output,'final_basis.raw')).equals(readFileSync(join(full.input,'basis.raw')))).toBe(true);
 expect(readFileSync(join(full.output,'final_reference.raw')).equals(readFileSync(join(full.input,'reference.raw')))).toBe(true);
 boundary=true;resumed=true;
},30000);
afterAll(()=>{
 if(unchanged&&resumed&&boundary)writeFileSync(join(out,'cpu-proof.json'),JSON.stringify({binary_sha256:build.binary_sha256,unchanged_math:unchanged,terminal_boundary:boundary,resume_bit_exact:resumed,cancel_before_commit:self.cancel_before_commit,parent_rng_restored:true,kernel:self,executions,model_loaded:false,GPU:false,independent_actual_shape_qualified:false},null,2)+'\n');
 for(const f of fixtures){f.close();rmSync(f.dir,{recursive:true,force:true});}rmSync(scratch,{recursive:true,force:true});
});
