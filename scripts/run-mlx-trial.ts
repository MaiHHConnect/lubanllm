/** One application GPU handoff for the finite MLX expert trial. */
import {applicationConnection} from './run-native-identity-round.ts';
import {homedir} from 'node:os';
import {join, resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const {values} = parseArgs({options: {directory: {type: 'string'}, output: {type: 'string'},
  binary: {type: 'string'}, training: {type: 'boolean', default: false}, 'skip-routes': {type: 'boolean', default: false}}});
assert(values.directory && values.output);
assert(!values.training || !values['skip-routes']);
const directory = resolve(values.directory), output = resolve(values.output);
assert(directory.startsWith('/Volumes/移动数据库/lubanllm/experiments/') && output.startsWith(directory + '/'));
const connection = await applicationConnection(join(homedir(), 'Library/Application Support/Luban'));
const lease_id = crypto.randomUUID();
const receipt = {lease_id, acquire_attempted: false, acquired: false, released: false,
  reloaded: false, exit_code: null as number | null};
const receiptPath = output + '.lease.json';
await writeFile(receiptPath, JSON.stringify(receipt) + '\n', {flag: 'wx', mode: 0o600});
const save = () => writeFile(receiptPath, JSON.stringify(receipt) + '\n', {mode: 0o600});
const command = async (path: string, body: unknown) => {
  const response = await fetch(connection.url + path, {method: 'POST',
    headers: {authorization: 'Bearer ' + connection.token, 'content-type': 'application/json'},
    body: JSON.stringify(body), proxy: null});
  const value: any = await response.json();
  assert(response.ok, 'Application handoff failed: ' + JSON.stringify(value));
  return value;
};
let wasReady = false, child: ReturnType<typeof Bun.spawn> | undefined;
const cancel = () => child?.kill('SIGINT');
process.on('SIGINT', cancel); process.on('SIGTERM', cancel);
try {
  receipt.acquire_attempted = true; await save();
  const lease = await command('/internal/native-evolution/acquire', {lease_id});
  receipt.acquired = true; wasReady = lease.was_ready === true; await save();
  const args = [resolve('.venv/bin/python'), resolve(values.training ? 'scripts/verify_mlx_native_training.py' : 'scripts/verify_mlx_trial_router.py'),
    '--directory', directory, '--output', output];
  if (values.binary) args.push('--binary', resolve(values.binary));
  if (values['skip-routes']) args.push('--skip-routes');
  child = Bun.spawn(args, {env: {...process.env, PYTHONPATH: resolve('src')}, stdout: 'inherit', stderr: 'inherit'});
  receipt.exit_code = await child.exited; child = undefined; await save();
  assert(receipt.exit_code === 0, 'MLX trial failed; partial evidence retained');
} finally {
  if (child) { child.kill('SIGINT'); await child.exited; }
  if (receipt.acquire_attempted) {
    const released = await command('/internal/native-evolution/release', {lease_id});
    receipt.released = released.released === true; wasReady ||= released.was_ready === true; await save();
  }
  if (receipt.released && wasReady) {
    await command('/api/model/reload', {}); receipt.reloaded = true; await save();
  }
  process.removeListener('SIGINT', cancel); process.removeListener('SIGTERM', cancel);
}
console.log(JSON.stringify(receipt));
