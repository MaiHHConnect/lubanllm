/** The retained source protocol accepts a complete raw BF16 matrix, not a Safetensors container. */
import assert from 'node:assert/strict';

export function assertNativeRawCoreDescriptor(text:string){
 const lines=text.split('\n'),config=new Map<string,string>();
 assert.equal(lines.shift(),'LUBAN_QWEN35_CACHE_CORE_INPUT_V1','Core source descriptor header differs');
 for(const line of lines){
  const fields=line.split('\t');if(fields[0]!=='config')continue;
  assert(fields.length===3&&!config.has(fields[1]!), 'Core source descriptor config is ambiguous');
  config.set(fields[1]!,fields[2]!);
 }
 const version=config.get('core_version');
 if(version==='null'){
  for(const key of ['core_path','core_file_sha256','core_tensor_sha256','core_tensor_key','core_offset','core_bytes'])
   assert.equal(config.get(key),'null','Original Core descriptor must not include an override');
 }else{
  assert(typeof version==='string'&&/^core-[a-f0-9]{12}$/.test(version),'Core source version differs');
  assert.equal(config.get('core_offset'),'0','Core source requires a raw tensor at offset zero');
  assert.equal(config.get('core_bytes'),'25165824','Core source requires full BF16 [2048,6144]');
  const pin=config.get('core_tensor_sha256');assert(pin&&/^[a-f0-9]{64}$/.test(pin));
  assert.equal(config.get('core_file_sha256'),pin,'Raw Core file and tensor bindings differ');
  assert(/^model\.(?:language_model\.)?layers\.23\.mlp\.(?:base\.)?down_proj\.weight$/.test(config.get('core_tensor_key')??''));
  assert(config.get('core_path')?.startsWith('/')&&!config.get('core_path')!.includes('\0'));
 }
}
