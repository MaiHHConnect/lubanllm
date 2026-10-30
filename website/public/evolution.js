(() => {
  const parameters = document.querySelector('#evolution-scale');
  const completed = document.querySelector('#evolution-completed');
  const clients = document.querySelector('#evolution-clients');
  const status = document.querySelector('#evolution-status');
  const updated = document.querySelector('#evolution-updated');
  if (!parameters || !completed || !clients || !status || !updated) return;
  const english = document.documentElement.lang.toLowerCase().startsWith('en');
  const format = (value) => {
    try { return BigInt(String(value)).toLocaleString(english ? 'en-US' : 'zh-CN'); }
    catch { return '—'; }
  };
  const refresh = async () => {
    try {
      const response = await fetch('/api/evolution/summary', { credentials: 'omit', cache: 'no-store' });
      if (!response.ok) throw new Error('summary');
      const value = await response.json();
      parameters.textContent = format(value.parameter_evolution_scale);
      completed.textContent = format(value.ranking_evolutions);
      clients.textContent = Number.isSafeInteger(value.ranking_entries) && value.ranking_entries >= 0 ? value.ranking_entries.toLocaleString(english ? 'en-US' : 'zh-CN') : '—';
      status.textContent = '';
      status.hidden = true;
      if (typeof value.updated_at === 'string' && value.updated_at) {
        const date = new Date(value.updated_at);
        if (!Number.isNaN(date.getTime())) {
          updated.dateTime = date.toISOString();
          updated.textContent = `${english ? 'Last updated ' : '最近更新 '}${date.toLocaleString(english ? 'en-US' : 'zh-CN')}`;
        }
      }
    } catch {
      status.hidden = false;
      status.textContent = english ? 'The latest summary is temporarily unavailable' : '暂时无法读取最新统计';
    }
  };
  refresh();
  window.setInterval(refresh, 30000);
})();
