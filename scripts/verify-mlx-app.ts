/** Actual installed application smoke test; no teacher, training or claimed mastery. */
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {homedir} from 'node:os';
import {resolve} from 'node:path';
import {parseArgs} from 'node:util';

const {values} = parseArgs({options: {cases: {type: 'string'}, output: {type: 'string'}}});
assert(values.cases && values.output);
const samples = JSON.parse(await readFile(resolve(values.cases), 'utf8'));
assert(Array.isArray(samples) && samples.length > 0 && samples.length <= 16);
const launch = JSON.parse(await readFile(homedir() + '/Library/Application Support/Luban/launch.json', 'utf8'));
const request = async (path: string, body?: unknown, method = body === undefined ? 'GET' : 'POST') => {
  const response = await fetch('http://127.0.0.1:' + launch.port + path, {method,
    headers: {cookie: 'luban_session=' + launch.token, 'content-type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(180000), proxy: null});
  assert(response.ok, `Application HTTP ${response.status} at ${path}`);
  return response;
};
const initial: any = await (await request('/api/status')).json();
assert(initial.runtime.state === 'ready' && initial.runtime.inference.engine === 'luban-native-mlx-q4-metal');
const config: any = await (await request('/api/settings')).json();
const previous = {answer_mode: config.teacher.answer_mode, teaching_modalities: config.teacher.teaching_modalities};
const report: any = {schema: 'mlx-expert-application-trial-v1', started: new Date().toISOString(),
  scope: 'local-trial', quality_admitted: false, optimizer_updates: 0, teacher_requests: 0,
  engine: initial.runtime.inference, core: initial.runtime.core_version, rows: [], settings_restored: false};
try {
  await request('/api/settings', {teacher: {answer_mode: 'local_first', teaching_modalities: []}}, 'PATCH');
  for (const sample of samples) {
    assert(typeof sample.id === 'string' && Array.isArray(sample.messages));
    const response = await request('/api/chat', {messages: sample.messages, route: 'auto',
      thinking: 'off', temperature: 0, max_tokens: 128});
    const frames = (await response.text()).split('\n').filter(line => line.startsWith('data: '))
      .map(line => JSON.parse(line.slice(6)));
    const error = frames.find(row => row.type === 'error');
    assert(!error, JSON.stringify(error));
    const meta = frames.find(row => row.type === 'meta'), done = frames.find(row => row.type === 'done');
    assert(meta && done);
    const events: any = await (await request('/api/events')).json();
    const decision = events.events.find((row: any) => row.kind === 'decision' && row.body.request_id === meta.request_id)?.body;
    assert(decision && decision.decision.route === 'local' && decision.usage.reference_used === false);
    const answer = frames.filter(row => row.type === 'delta').map(row => row.content ?? '').join('');
    assert.equal(answer, decision.answer);
    const status: any = await (await request('/api/status')).json();
    const row = {id: sample.id, answer, decision: decision.decision, model_context: decision.model_context,
      complete: decision.complete, usage: done.usage, native_process: status.runtime.native_process,
      context: status.runtime.context};
    report.rows.push(row);
    if (sample.expected_action) assert.equal(decision.decision.action, sample.expected_action);
    if (sample.expected_action === 'neural_expert') assert(decision.model_context.expert_id);
    console.log(JSON.stringify({id: row.id, answer, decision: row.decision, usage: row.usage}));
  }
  assert(report.rows.some((row: any) => row.decision.action === 'neural_expert'), 'No actual expert route observed');
  report.complete = true;
} catch (error) {
  report.error = String(error);
  throw error;
} finally {
  try { await request('/api/settings', {teacher: previous}, 'PATCH'); report.settings_restored = true; }
  finally {
    report.finished = new Date().toISOString();
    await writeFile(resolve(values.output), JSON.stringify(report, null, 2) + '\n', {flag: 'wx', mode: 0o600});
  }
}
