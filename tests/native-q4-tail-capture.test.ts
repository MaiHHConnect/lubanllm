import {test,expect} from 'bun:test';
import {readFile} from 'node:fs/promises';
import {replayTraceCases,validateTailFrame} from '../scripts/capture-native-q4-tail.ts';
const root='/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/identity-native450-v19';
const evidence=JSON.parse(await readFile('research/native_evolution_final_policy_20261017.json','utf8'));
const baseline=JSON.parse(await readFile(root+'/q4-candidate-v1/baseline.json','utf8'));
test('actual two-pass Q4 baseline selects only authorized correct TRAIN and aligns EOS prediction',()=>{
 const rows=replayTraceCases(evidence,baseline,true);expect(rows).toHaveLength(16);expect(rows.every((r:any)=>r.id.includes('replay-training'))).toBe(true);
 const [json]=replayTraceCases(evidence,baseline,false);expect(json.id).toBe('q4-id-replay-training-8');expect(json.targets).toHaveLength(8);expect(json.forced_tokens).toHaveLength(7);expect(json.prediction_positions).toEqual([25,26,27,28,29,30,31,32]);expect(json.targets.at(-1)).toBe(248044);
 for(const change of [(b:any)=>b[0].core={},(b:any)=>b[0].rows.find((r:any)=>r.id===json.id).passed=false,(b:any)=>b[0].rows.find((r:any)=>r.id===json.id).tokens.pop()]){const bad=structuredClone(baseline);change(bad);bad[1]=structuredClone(bad[0]);expect(()=>replayTraceCases(evidence,bad,false)).toThrow();}
 const bad=structuredClone(evidence);bad.records.groups.replay_training[0].purpose='validation';expect(()=>replayTraceCases(bad,baseline,true)).toThrow();
});
test('actual BF16 tail frame contract rejects position, truncated/nonfinite bytes and changed predictions',()=>{
 const [row]=replayTraceCases(evidence,baseline,false),tail=Buffer.alloc(20480),logits=Buffer.alloc(496640);logits.writeUInt16LE(0x3f80,row.targets[0]*2);
 const event={event:'trace',id:row.id,step:0,committed_tokens:26,tail_only:true,layer:23,tail_values:10240,vocab:248320,top1:row.targets[0],tail_path:'/trace/'+row.id+'-step0000.tail.bf16',logits_path:'/trace/'+row.id+'-step0000.logits.bf16'};
 expect(validateTailFrame(event,row,0,'/trace',tail,logits).prediction_position).toBe(25);
 expect(()=>validateTailFrame({...event,committed_tokens:25},row,0,'/trace',tail,logits)).toThrow();
 expect(()=>validateTailFrame(event,row,0,'/trace',tail.subarray(2),logits)).toThrow();
 const invalid=Buffer.from(tail);invalid.writeUInt16LE(0x7f80,0);expect(()=>validateTailFrame(event,row,0,'/trace',invalid,logits)).toThrow();
 expect(()=>validateTailFrame({...event,top1:0},row,0,'/trace',tail,logits)).toThrow();
 expect(()=>validateTailFrame({...event,tail_path:'/other'},row,0,'/trace',tail,logits)).toThrow();
});
