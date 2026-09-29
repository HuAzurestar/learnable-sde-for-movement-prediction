'use strict';
const token = new URLSearchParams(location.hash.slice(1)).get('session') || '';
history.replaceState(null, '', location.pathname);
const el = id => document.getElementById(id);
let view = 'runs', cursor = null, rows = [];
async function api(path, raw = false) {
  const response = await fetch(path, {headers: {'X-Session-Token': token}});
  if (!response.ok) { const body = await response.json(); throw new Error(body.error?.safe_details || body.error?.code || 'Result unavailable'); }
  return raw ? response.blob() : response.json();
}
function text(tag, value, className) {const node = document.createElement(tag); node.textContent = String(value ?? '—'); if (className) node.className = className; return node;}
function showError(error) {el('error').hidden = false; el('error').textContent = error.message;}
function table(headers, values) {
  const result = document.createElement('table'), head = document.createElement('tr');
  headers.forEach(h => head.append(text('th', h))); const thead = document.createElement('thead'); thead.append(head); result.append(thead);
  const body = document.createElement('tbody'); values.forEach(cells => {const row = document.createElement('tr'); cells.forEach(value => {const td = document.createElement('td'); td.append(value instanceof Node ? value : text('span', value)); row.append(td);}); body.append(row);}); result.append(body); return result;
}
function action(label, fn) {const button = text('button', label); button.onclick = () => Promise.resolve().then(fn).catch(showError); return button;}
function forecastPlot(result) {
  const samples = result.forecast?.samples;
  if (!samples?.length) return text('p', 'Per-case metrics and provenance are shown below.');
  const points = samples.flat().filter(p => p.length >= 2 && p.slice(0,2).every(Number.isFinite));
  if (!points.length) return text('p', 'No finite forecast coordinates available.');
  const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 600 300'); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Forecast state sample paths; display only');
  const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
  const minX = Math.min(...xs), minY = Math.min(...ys), spanX = Math.max(...xs)-minX || 1, spanY = Math.max(...ys)-minY || 1;
  samples.forEach(sample => {const line = document.createElementNS(ns, 'polyline'); line.setAttribute('points', sample.map(p => `${40+(p[0]-minX)/spanX*520},${250-(p[1]-minY)/spanY*220}`).join(' ')); line.setAttribute('fill','none'); line.setAttribute('stroke','#287c9c'); line.setAttribute('stroke-opacity','.4'); svg.append(line);});
  const caption = document.createElementNS(ns,'text'); caption.setAttribute('x','40'); caption.setAttribute('y','285'); caption.textContent = `${result.state_order[0]} (${result.units[0]}) / ${result.state_order[1]} (${result.units[1]}) · ${samples.length} sample paths · display only`; svg.append(caption); return svg;
}
async function download(id) {
  const blob = await api('/api/artifacts/' + encodeURIComponent(id) + '?download=1', true);
  const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = 'research-metrics.csv'; link.click(); URL.revokeObjectURL(url);
}
async function detail(row) {
  const target = el('detail-content'); target.replaceChildren(); el('detail').hidden = false;
  if (view === 'comparisons') {
    const data = await api('/api/comparisons/' + encodeURIComponent(row.manifest.aggregate_hash));
    target.append(text('p', 'Aggregate version: ' + data.aggregate.aggregate_hash));
    target.append(table(['Arm', 'Metric', 'Value', 'Unit', 'Independent blocks', 'Cells (successful / expected)', 'Status'],
      data.aggregate.arms.flatMap(arm => Object.keys(arm.metrics).length ? Object.entries(arm.metrics).map(([metric, value]) => [arm.arm_id, metric, value, arm.metric_units[metric], arm.independent_n, `${arm.successful_cells} / ${arm.expected_cells}`, arm.status]) : [[arm.arm_id, 'No complete metric', '—', '—', arm.independent_n, `${arm.successful_cells} / ${arm.expected_cells}`, arm.status]])));
    target.append(action('Download frozen CSV', () => download(data.package.table)));
    target.append(text('h3', 'Comparison and provenance')); target.append(text('pre', JSON.stringify(data.aggregate, null, 2)));
  } else {
    const data = await api('/api/runs/' + encodeURIComponent(row.manifest.run_id));
    target.append(text('pre', JSON.stringify(data, null, 2)));
    data.attempts.filter(a => a.artifact_id).forEach(attempt => target.append(action('Inspect result ' + attempt.attempt_id.slice(0, 8), async () => {
      const result = await api('/api/artifacts/' + encodeURIComponent(attempt.artifact_id)); target.append(forecastPlot(result)); target.append(text('pre', JSON.stringify(result, null, 2)));
    })));
  }
  el('detail').scrollIntoView({behavior: 'smooth', block: 'start'});
}
function render() {
  const target = el('results'); target.replaceChildren();
  if (!rows.length) {target.append(text('p', view === 'runs' ? 'No runs yet. Registered cells remain available for execution through the CLI.' : 'No frozen comparisons yet. Import a validated evidence package to view it here.', 'empty')); return;}
  if (view === 'runs') target.append(table(['Arm', 'Run', 'Seed / block', 'State', 'Attempts', ''], rows.map(row => [row.manifest.arm_id, row.manifest.run_id?.slice(0, 14) || 'Not started', `${row.manifest.seed} / ${row.manifest.cell.block_id}`, text('span', row.state, 'status ' + row.state.toLowerCase()), row.attempts.length, row.manifest.run_id ? action('Inspect', () => detail(row)) : 'Registered cell'])));
  else target.append(table(['Study', 'Aggregate version', ''], rows.map(row => [row.manifest.study_id, row.manifest.aggregate_hash, action('Compare & export', () => detail(row))])));
}
async function load(more = false) {
  el('error').hidden = true;
  if (!token) throw new Error('Open the local session link printed by the serve command.');
  const studies = await api('/api/studies');
  const study = studies.items[0]?.manifest.spec; el('study').textContent = study?.study_id || 'No study';
  el('summary').textContent = study ? `${study.cells.length} registered cells · ${study.arms.length} arms · read-only session` : 'No authorized study found.';
  const page = await api('/api/' + view + (more && cursor ? '?cursor=' + encodeURIComponent(cursor) : ''));
  rows = more ? rows.concat(page.items) : page.items; cursor = page.next_cursor; el('more').hidden = !cursor; render();
}
document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => {view = button.dataset.view; document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active', b === button)); el('detail').hidden = true; load().catch(showError);});
el('refresh').onclick = () => load().catch(showError); el('more').onclick = () => load(true).catch(showError); el('close').onclick = () => {el('detail').hidden = true;}; load().catch(showError);
