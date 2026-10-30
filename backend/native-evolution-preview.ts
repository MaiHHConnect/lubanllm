/** Borrow the main ledger through its existing per-call identity context; never change its persistent identity. */
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import type {Memory} from './memory';
export type NativeEvolutionPreviewMemory=Pick<Memory,'root'|'connect'|'dispatch'>;
const previews=new WeakSet<object>();
export const isNativeEvolutionPreviewMemory=(value:object)=>previews.has(value);
export async function nativeEvolutionPreviewMemory(source:NativeEvolutionPreviewMemory,root:string,identity:string){
 assert(source.root===resolve(root)&&/^[a-f0-9]{64}$/.test(identity),'进化预览资料库与已验证作用域不一致');
 // dispatch restores the shared identity synchronously in finally, before its Promise settles.
 // It also preserves Memory's model filter, growth exclusions and active SQL lease handling.
 const samples=await source.dispatch('samples',[],{},identity);
 assert(Array.isArray(samples),'进化预览资料库未返回材料');
 const preview=Object.freeze({model_identity:identity,samples:(purpose?:string)=>purpose?samples.filter(row=>row.purpose===purpose):samples,connect:source.connect.bind(source)});
 previews.add(preview);return preview;
}
