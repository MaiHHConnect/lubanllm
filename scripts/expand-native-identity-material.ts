/** Authored paraphrases of the same three user-authorized facts; no teacher/model calls. */
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {readIdentityMaterial} from './run-native-identity-round.ts';

const prompts:Record<string,string[]>={
 model_name:[
  '请区分模型名称和微调者：这里只需要模型名称。',
  '当前这一个模型应该用什么名称称呼？',
  'What should I call the model I am talking to? Give only its model name.',
  '你的模型标识是什么？只给标识。'],
 model_fine_tuner:[
  '请区分模型名称和微调者：这里只需要微调者。',
  '这份模型由谁完成微调？只给微调者名称。',
  'Name the entity responsible for your fine-tuning. Reply only with that name.',
  'Who is your fine-tuner? Give only the fine-tuner name.'],
 model_positioning:[
  '你的持续进化定位用哪四个字表达？只回答这四个字。',
  '作为luban0，你的进化方向叫什么？仅给定位名称。',
  'What is the name of your evolution goal? Reply with its four Chinese characters only.',
  '请只给出麦好火设定的四字进化定位，不要解释。'],
};
export function expandIdentityMaterial(original:Awaited<ReturnType<typeof readIdentityMaterial>>){
 assert.equal(original.samples.length,50);
 const result=structuredClone(original),seen=new Set(original.samples.map(r=>r.prompt.trim().toLowerCase()));
 for(const[fact,questions]of Object.entries(prompts)){
  const template=original.samples.find(r=>r.purpose==='training'&&r.provenance?.teaching_fact===fact)!;assert(template);
  for(const[i,prompt]of questions.entries()){
   assert(!seen.has(prompt.trim().toLowerCase()));seen.add(prompt.trim().toLowerCase());
   const id=`q4-id-${fact}-training-expanded-${i}`,row=structuredClone(template);
   Object.assign(row,{id,prompt});Object.assign(row.provenance,{split_family:id,messages:[{role:'user',content:prompt}]});
   result.samples.push(row);result.directedTeaching.facts[fact].training_ids.push(id);
  }
 }
 assert.equal(result.samples.length,62);return result;
}
if(import.meta.main){
 const root='/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/';
 const material=expandIdentityMaterial(await readIdentityMaterial(root+'authorized-identity-material.json'));
 const path=root+'authorized-identity-material-expanded-v1.json';
 await writeFile(path,JSON.stringify(material,null,2)+'\n',{flag:'wx',mode:0o600});
 await readIdentityMaterial(path);console.log(JSON.stringify({path,rows:62,added_training:12,validation_changed:false,teacher_calls:0,model_loaded:false}));
}
