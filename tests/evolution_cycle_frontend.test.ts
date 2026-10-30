import { test, expect } from 'bun:test';
import { initializeEvolutionSummary } from '../frontend/evolution-summary';
function controller() {
  const elements = new Map<string, any>();
  const get = (id: string) => {
    if (!elements.has(id)) elements.set(id, { hidden: false, textContent: '', value: undefined, max: 1,
      style: { setProperty() {} }, addEventListener() {}, removeAttribute(key: string) { delete this[key]; } });
    return elements.get(id);
  };
  (globalThis as any).document = { getElementById: get };
  let summary: any;
  const actions = initializeEvolutionSummary(async () => { if (summary instanceof Error) throw summary; return summary; });
  return { get, ...actions, async update(value: any) { summary = value; await actions.refresh(); } };
}
const summary = (step: number, phase = 'training') => ({ model_name: '鲁班', round_number: 3,
  current: { state: 'running', phase, step, steps: 450, materials: 38, gaps: 12 },
  previous: null, snapshot: { available: false }, error: null });
test('cycle foreground uses actual round and optimizer steps, validation stays separate', async () => {
  const app = controller(); await app.update(summary(320));
  expect(app.get('evolution-cycle-round').textContent).toBe('第 3 轮进化');
  expect(app.get('evolution-cycle-progress').textContent).toBe('正在成长：320 / 450 步');
  expect(app.get('evolution-progress').value).toBe(320);
  expect(app.get('evolution-progress').max).toBe(450);
  await app.update(summary(450, 'validation'));
  expect(app.get('evolution-cycle-next').textContent).toBe('正在验证新知识与原有能力');
});
test('absolute continuation progress never becomes a round step, and tokens cannot create progress or a round', async () => {
  const app = controller(); await app.update(new Error('Not Found'));
  app.renderRuntime({ state: 'evolving', job: { kind: 'evolution', state: 'running', phase: 'training',
    step: 770, steps: 900, absolute_step: 770, round_step: 320, round_steps: 450 } });
  expect(app.get('evolution-cycle-progress').textContent).toBe('正在成长：320 / 450 步');
  expect(app.get('evolution-cycle-round').textContent).toBe('轮次待确认');
  app.renderRuntime({ state: 'evolving', job: { kind: 'evolution', state: 'running',
    step: 770, steps: 900, absolute_step: 770, round_step: 451, round_steps: 450 } });
  expect(app.get('evolution-cycle-progress').textContent).toBe('正在成长 · 步数待确认');
  expect(app.get('evolution-progress').value).toBeUndefined();
  app.renderRuntime({ state: 'ready', usage: { total_tokens: 900000 } } as any);
  expect(app.get('evolution-cycle-progress').textContent).toBe('等待开始成长');
});

test('accepted round waiting for material displays the actual pending reason over the saved completed summary', async () => {
  const app = controller(), saved = summary(450, 'committed'); saved.current.state = 'complete';
  await app.update(saved);
  const reason = '训练材料已掌握，独立验证仍有待学项；等待新增或改进训练材料，不把验证题改作训练。';
  app.renderRuntime({ state: 'ready', job: { kind: 'evolution', state: 'waiting_material',
    phase: 'committed_waiting_material', step: 450, steps: 450, result: { continuation_reason: reason } } });
  expect(app.get('evolution-cycle-next').textContent).toBe(reason);
  expect(app.get('evolution-cycle-progress').textContent).toBe('等待新训练材料：450 / 450 步');
});

test('current expert actual updates override an older fixed core round summary', async () => {
  const app = controller(); await app.update(summary(450, 'committed'));
  app.renderRuntime({ state: 'evolving', job: { kind: 'learning', state: 'running', phase: 'validation', step: 8, steps: 192 } });
  expect(app.get('evolution-progress').value).toBe(8);
  expect(app.get('evolution-progress').max).toBe(192);
  expect(app.get('evolution-cycle-progress').textContent).toBe('正在成长：8 / 192 步');
});
