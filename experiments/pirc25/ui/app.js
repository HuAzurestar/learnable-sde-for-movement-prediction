'use strict';
const token = new URLSearchParams(location.hash.slice(1)).get('session') || '';
history.replaceState(null, '', location.pathname);
const el = id => document.getElementById(id);
let view = 'runs', cursor = null, rows = [];
let appliedFilters = {}, loadSequence = 0;
const filterKeys = ['model', 'version', 'trainer', 'predictor', 'horizon', 'seed', 'state'];
async function api(path, raw = false) {
  const response = await fetch(path, {headers: {'X-Session-Token': token}});
  if (!response.ok) { const body = await response.json(); throw new Error([body.error?.code, body.error?.safe_details].filter(Boolean).join(': ') || 'Result unavailable'); }
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
function forecastPlot(result, selected = null) {
  const samples = result.forecast?.samples;
  if (!samples?.length) return text('p', 'Per-case metrics and provenance are shown below.');
  const points = samples.flat().filter(p => p.length >= 2 && p.slice(0,2).every(Number.isFinite));
  if (!points.length) return text('p', 'No finite forecast coordinates available.');
  const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 600 300'); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Forecast state sample paths; display only');
  const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
  const minX = Math.min(...xs), minY = Math.min(...ys), spanX = Math.max(...xs)-minX || 1, spanY = Math.max(...ys)-minY || 1;
  samples.forEach(sample => {const visible = selected === null ? sample : sample.slice(0, selected + 1); const line = document.createElementNS(ns, 'polyline'); line.setAttribute('points', visible.map(p => `${40+(p[0]-minX)/spanX*520},${250-(p[1]-minY)/spanY*220}`).join(' ')); line.setAttribute('fill','none'); line.setAttribute('stroke','#287c9c'); line.setAttribute('stroke-opacity','.4'); svg.append(line);});
  const caption = document.createElementNS(ns,'text'); caption.setAttribute('x','40'); caption.setAttribute('y','285'); caption.textContent = `${result.state_order[0]} (${result.units[0]}) / ${result.state_order[1]} (${result.units[1]}) · ${samples.length} sample paths · display only`; svg.append(caption); return svg;
}
function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function sourceFigure(svg, source, title) {
  if (svg.namespaceURI !== 'http://www.w3.org/2000/svg') throw new Error('No figure is available for this result.');
  const copy = svg.cloneNode(true), ns = copy.namespaceURI;
  copy.setAttribute('xmlns', ns);
  const metadata = document.createElementNS(ns, 'metadata'); metadata.textContent = JSON.stringify(source); copy.prepend(metadata);
  const description = document.createElementNS(ns, 'desc'); description.textContent = title + ' · ' + JSON.stringify(source); copy.prepend(description);
  const bounds = copy.getAttribute('viewBox').split(' ').map(Number), height = bounds[3];
  copy.setAttribute('viewBox', `0 0 ${bounds[2]} ${height + 65}`);
  ['artifact_id','spec_hash','protocol_hash'].forEach((key,i) => {const caption = document.createElementNS(ns,'text'); caption.setAttribute('x','12'); caption.setAttribute('y',String(height + 15 + i*16)); caption.setAttribute('font-size','9'); caption.textContent = `${key}: ${source[key] || 'unspecified'}`; copy.append(caption);});
  saveBlob(new Blob([new XMLSerializer().serializeToString(copy)], {type:'image/svg+xml'}), 'research-figure.svg');
}
function dimensionsLabel(value) {return JSON.stringify(value || {});}
function selector(id, label, values, changed) {
  const wrapper = text('label', label + ' '), select = document.createElement('select'); select.id = id;
  values.forEach(([value, title]) => {const option = text('option', title); option.value = value; select.append(option);});
  select.onchange = () => Promise.resolve().then(changed).catch(showError); wrapper.append(select); return wrapper;
}
function budgetView(budget) {
  const section = document.createElement('section'); section.append(text('h3', 'Budget & cost provenance'));
  section.append(text('p', `${budget.committed_ms} committed / ${budget.limit_ms} slot-ms; ${budget.remaining_ms} remaining · ${budget.closed ? 'arm closed' : 'arm open'} · cumulative arm, not filtered horizon`));
  section.append(table(['Reservation', 'Reserved slot-ms', 'Charged slot-ms', 'Measured slot-ms', 'Basis', 'Event source'],
    budget.sources.map(source => [source.reservation_id, source.reserved_ms, source.settled ? source.charged_ms : 'Not settled', source.monotonic_elapsed_ms ?? 'Unknown', source.cost_basis, `${source.event_sequence}: ${source.event_hash}`])));
  if (!budget.sources.length) section.append(text('p', 'No reservation source for this run; this is not a measured zero cost.'));
  return section;
}
async function inspectCase(target, artifactId) {
  const result = await api('/api/artifacts/' + encodeURIComponent(artifactId));
  const section = document.createElement('section'), chart = document.createElement('div');
  const horizons = result.forecast?.horizons;
  let selected = null;
  const draw = () => {chart.replaceChildren(forecastPlot(result, selected)); chart.id = 'case-chart'; chart.dataset.horizon = selected === null ? 'all' : String(horizons[selected]);};
  if (Array.isArray(horizons) && horizons.length) section.append(selector('case-horizon', 'Case horizon', [['all','All horizons'], ...horizons.map((h,i) => [String(i), `${h} ${result.time_unit || '(time unit unspecified)'}`])], () => {const value = el('case-horizon').value; selected = value === 'all' ? null : Number(value); draw();}));
  else section.append(text('p', 'No explicit horizon grid in this artifact; horizon switching is unavailable.'));
  draw(); section.append(chart);
  const preview = document.createElement('section'); preview.id = 'preview-provenance';
  preview.append(text('h3','Preview provenance'));
  const policy = result.forecast?.preview || {};
  preview.append(table(['Declared field','Value'], ['case_selection_rule','sample_selection_rule','sample_ids','generation_version','n_samples'].map(key => [key,policy[key] === undefined ? 'Unavailable — not declared in this artifact' : JSON.stringify(policy[key])])));
  preview.append(text('p',`${result.forecast?.samples?.length ?? 0} saved paths displayed; no best-case selection or metric recomputation is performed by this viewer.`));
  section.append(preview);
  const optional = document.createElement('section'); optional.id = 'optional-payloads';
  optional.append(text('h3','Saved optional payloads'));
  const fields = [['History',result.observations?.history],['Evaluation truth',result.observations?.truth],
    ['Moments (estimation kind as stored)',result.forecast?.moments],['Density',result.forecast?.density],
    ['Region estimates',result.forecast?.region_estimates],['Uncertainty',result.forecast?.uncertainty],
    ['Mode probabilities',result.forecast?.mode_probabilities],['Mode paths',result.forecast?.mode_paths],
    ['Per-segment results',result.forecast?.per_segment]];
  fields.forEach(([name,value]) => {
    if (value === undefined || value === null) {optional.append(text('p',`${name}: unavailable — not saved in this artifact.`)); return;}
    optional.append(text('h4',name + ' (saved values; original definitions retained)'));
    if (Array.isArray(value) && value.every(item => item && typeof item === 'object' && !Array.isArray(item))) {
      const keys = [...new Set(value.flatMap(Object.keys))];
      optional.append(table(keys,value.slice(0,200).map(item => keys.map(key => JSON.stringify(item[key]) ?? 'Unavailable'))));
      if (value.length > 200) optional.append(text('p',`Showing first 200 of ${value.length} saved rows in stored order; no changes to metrics.`));
    } else optional.append(text('pre',JSON.stringify(value,null,2)));
  });
  section.append(optional);
  section.append(action('Download result manifest', async () => saveBlob(await api('/api/artifacts/' + encodeURIComponent(artifactId) + '?manifest=1',true),'research-result-manifest.json')));
  section.append(action('Export case figure', async () => {
    const fresh = await (await api('/api/artifacts/' + encodeURIComponent(artifactId) + '?download=1', true)).text();
    const exported = JSON.parse(fresh);
    sourceFigure(forecastPlot(exported, selected), {artifact_id:artifactId, spec_hash:exported.spec_hash, cell_hash:exported.cell_hash,
      protocol_hash:exported.protocol_hash, horizon:selected === null ? null : horizons[selected], units:exported.units}, 'Forecast display; not a recomputed metric');
  }));
  section.append(text('pre', JSON.stringify(result, null, 2))); target.querySelector('[data-case]')?.remove(); section.dataset.case = 'true'; target.append(section);
}
function comparisonRows(aggregate, horizon) {
  return aggregate.arms.filter(arm => horizon === 'all' || String(arm.comparison_dimensions?.horizon) === horizon);
}
function adjudicationLines(aggregate, receipt) {
  const decision = aggregate.adjudication;
  if (!decision) return ['Formal adjudication unavailable in this frozen version; descriptive intervals are not a verdict.'];
  const policy = decision.adjudication_spec;
  const lines = [`Frozen adjudication ${decision.compare_hash} · ${decision.status} · ${decision.qualification}`,
    'Full fixed comparison family; horizon display filtering does not recompute or subset the decision.'];
  if (policy) lines.push(`Primary ${policy.primary_metric.name} (${policy.primary_metric.unit}) · ${policy.primary_metric.direction} · threshold ${policy.practical_threshold} · multiplicity ${policy.multiplicity} · seed pairing ${policy.seed_pairing}`);
  (decision.diagnostics || []).forEach(message => lines.push('Diagnostic: ' + message));
  decision.records.forEach(record => {
    lines.push(`${record.comparison_id}: ${record.verdict} · effect ${record.effect ?? 'Unavailable'} ${record.unit} · interval ${JSON.stringify(record.interval)} · confidence ${record.interval_confidence ?? 'Unavailable'}`);
    lines.push(`  Independent paired blocks ${record.independent_n} · registered-cell denominator ${record.status_rates?.denominator ?? 'Unavailable'} · ${JSON.stringify(record.status_rates?.values || {})}`);
  });
  lines.push(`Computation receipt ${aggregate.computation_ref?.manifest_id ?? 'Unavailable'} · ${receipt?.cost?.charged_ms ?? 'Unknown'} ${receipt?.cost?.unit || 'slot-ms'} · ${receipt?.cost?.scope || 'Unknown cost scope'}`);
  return lines;
}
function adjudicationView(aggregate, receipt) {
  const section = document.createElement('section'); section.id = 'comparison-adjudication';
  section.append(text('h3','Frozen preregistered adjudication'));
  const decision = aggregate.adjudication;
  if (!decision) {section.append(text('p',adjudicationLines(aggregate,receipt)[0])); return section;}
  section.append(text('p',`${decision.status} · ${decision.qualification} · ${decision.compare_hash}`));
  section.append(text('p','Full fixed comparison family. Changing the horizon display does not recompute intervals, weights, thresholds or multiplicity. NO_GAIN is not equivalence. Fixture verdicts are engineering tests, not scientific qualification.'));
  const policy = decision.adjudication_spec;
  if (policy) section.append(table(['Frozen policy field','Value'],[
    ['Primary metric',JSON.stringify(policy.primary_metric)],['Practical threshold',policy.practical_threshold],
    ['Interval',JSON.stringify(policy.interval)],['Multiplicity / fixed family',`${policy.multiplicity} / ${decision.family_size}`],
    ['Independent unit / seed policy',`${policy.independent_unit} / ${policy.seed_aggregation} / ${policy.seed_pairing}`],
    ['Minimum seeds / paired blocks',`${policy.minimum_seeds} / ${policy.minimum_paired_blocks}`],
    ['Weighted strata',JSON.stringify(policy.contrasts)],['Failure / stopping rules',`${policy.missing_policy} / ${policy.attempt_policy} / ${policy.stopping_rule}`]
  ]));
  section.append(table(['Comparison','Verdict','Oriented benefit','Unit','Conditional interval','Confidence','Independent paired blocks','All-cell dispositions / denominator'], decision.records.map(record => [
    record.comparison_id,record.verdict,record.effect ?? 'Unavailable',record.unit,record.interval ? record.interval.join(' to ') : 'Unavailable',record.interval_confidence ?? 'Unavailable',record.independent_n,
    `${JSON.stringify(record.status_rates?.values || {})} / ${record.status_rates?.denominator ?? 'Unavailable'}`])));
  section.append(text('pre',JSON.stringify({diagnostics:decision.diagnostics,records:decision.records},null,2)));
  const cost = receipt?.cost;
  section.append(text('p',`Shared computation: ${cost?.charged_ms ?? 'Unknown'} ${cost?.unit || 'slot-ms'} · ${cost?.scope || 'Unknown scope'} · charged arm ${cost?.arm_id || 'Unknown'}. This job cost is separate from frozen experimental costs; it is not repeated per contrast.`));
  section.append(text('pre',JSON.stringify({computation_ref:aggregate.computation_ref,computation_receipt:receipt ?? null},null,2)));
  return section;
}
function comparisonFigure(aggregate, horizon, receipt = null) {
  const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
  const lines = [`Frozen comparison · ${aggregate.aggregate_hash}`, `Horizon: ${horizon} · independent unit: block · ${aggregate.qualification}`];
  comparisonRows(aggregate, horizon).forEach(arm => {
    lines.push(`${arm.arm_id} · ${dimensionsLabel(arm.comparison_dimensions)} · n=${arm.independent_n} · ${arm.status}`);
    Object.entries(arm.metrics).forEach(([metric,value]) => lines.push(`  ${metric}: ${value} ${arm.metric_units[metric]} · cells ${arm.successful_cells}/${arm.expected_cells}`));
    if (!Object.keys(arm.metrics).length) lines.push('  No complete metric; missing/failed cells retained');
    lines.push(`  Frozen cost (all attempts): charged ${arm.cost?.charged_ms ?? 'unavailable'}, reserved ${arm.cost?.reserved_ms ?? 'unavailable'}, measured ${arm.cost?.measured_ms ?? 'unknown'} slot-ms`);
  });
  lines.push(...adjudicationLines(aggregate,receipt));
  svg.setAttribute('viewBox', `0 0 1100 ${50 + lines.length * 25}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Frozen comparison values and provenance');
  lines.forEach((line,i) => {const node = document.createElementNS(ns,'text'); node.setAttribute('x','12'); node.setAttribute('y',String(25 + i*25)); node.setAttribute('font-size','13'); node.textContent=line; svg.append(node);});
  let y = 55 + lines.length * 25;
  const selected = comparisonRows(aggregate,horizon), metrics = [...new Set(selected.flatMap(arm => Object.keys(arm.metrics)))];
  const label = (value,x,at) => {const node=document.createElementNS(ns,'text'); node.setAttribute('x',String(x)); node.setAttribute('y',String(at)); node.setAttribute('font-size','12'); node.textContent=value; svg.append(node);};
  metrics.forEach(metric => {
    const values = selected.filter(arm => Number.isFinite(arm.metrics[metric]));
    const min = Math.min(0,...values.map(arm=>arm.metrics[metric])), max = Math.max(0,...values.map(arm=>arm.metrics[metric]));
    const x = value => 440 + (value-min)/(max-min || 1)*470;
    label(`${metric} (${values[0].metric_units[metric]}) · separate metric scale; no pooled horizons`,12,y); y+=25;
    values.forEach(arm => {
      label(`${arm.arm_id} ${dimensionsLabel(arm.comparison_dimensions)}`,12,y+12);
      const bar=document.createElementNS(ns,'rect'); bar.setAttribute('x',String(Math.min(x(0),x(arm.metrics[metric])))); bar.setAttribute('y',String(y)); bar.setAttribute('width',String(Math.max(1,Math.abs(x(arm.metrics[metric])-x(0))))); bar.setAttribute('height','14'); bar.setAttribute('fill','#287c9c'); svg.append(bar);
      label(String(arm.metrics[metric]),930,y+12); y+=25;
    }); y+=20;
  });
  svg.setAttribute('viewBox',`0 0 1100 ${y+15}`); return svg;
}
function showComparison(target, data) {
  const aggregate = data.aggregate, pane = document.createElement('div'); let horizon = 'all';
  const horizons = [...new Set(aggregate.arms.map(arm => arm.comparison_dimensions?.horizon).filter(h => h !== undefined).map(String))].sort((a,b) => Number(a)-Number(b));
  const draw = () => {
    pane.replaceChildren();
    const summary = table(['Arm / dimensions', 'Metric', 'Value', 'Unit', 'Independent blocks', 'Cells (successful / expected)', 'Status'],
      comparisonRows(aggregate,horizon).flatMap(arm => Object.keys(arm.metrics).length ? Object.entries(arm.metrics).map(([metric,value]) => [arm.arm_id + ' ' + dimensionsLabel(arm.comparison_dimensions), metric,value,arm.metric_units[metric],arm.independent_n,`${arm.successful_cells} / ${arm.expected_cells}`,arm.status]) : [[arm.arm_id + ' ' + dimensionsLabel(arm.comparison_dimensions),'No complete metric','—','—',arm.independent_n,`${arm.successful_cells} / ${arm.expected_cells}`,arm.status]]));
    summary.id = 'comparison-table'; pane.append(summary);
    pane.append(text('h3','Paired comparisons and failure dispositions'));
    pane.append(table(['Reference / candidate', 'Dimensions', 'Metric', 'Candidate − reference', '95% interval', 'Paired blocks', 'Interval kind'], aggregate.comparisons.filter(c => horizon === 'all' || String(c.comparison_dimensions?.horizon) === horizon).flatMap(c => {
      const metrics = Object.entries(c.metrics);
      return metrics.length ? metrics.map(([name,result]) => [`${c.reference} / ${c.candidate}`,dimensionsLabel(c.comparison_dimensions),name,result.candidate_minus_reference,result.interval95 ? result.interval95.join(' to ') : 'Unavailable',c.independent_n,c.interval_kind]) : [[`${c.reference} / ${c.candidate}`,dimensionsLabel(c.comparison_dimensions),'No paired metric','—','Unavailable',c.independent_n,c.interval_kind]];
    })));
    pane.append(table(['Arm / dimensions', 'Dispositions'], comparisonRows(aggregate,horizon).map(arm => [arm.arm_id + ' ' + dimensionsLabel(arm.comparison_dimensions),JSON.stringify(arm.dispositions)])));
    pane.append(text('h3','Frozen status rates'));
    pane.append(text('p','Fractions use all registered cells in each arm and comparison stratum, not independent blocks or attempt counts. Missing, active, failed and timed-out cells remain separate.'));
    const rates = table(['Arm / dimensions','Status','Rate','Unit','Denominator','Denominator kind'], comparisonRows(aggregate,horizon).flatMap(arm => arm.status_rates ? Object.entries(arm.status_rates.values).map(([state,value]) => [arm.arm_id + ' ' + dimensionsLabel(arm.comparison_dimensions),state,value,arm.status_rates.unit,arm.status_rates.denominator,arm.status_rates.denominator_kind]) : [[arm.arm_id,'Unavailable in this frozen version','—','—','—','—']]));
    rates.id = 'comparison-status-rates'; pane.append(rates);
    pane.append(text('h3','Frozen quality & cost'));
    pane.append(text('p','Costs cover all attempts in this stratum, including failed retries and unscored blocks. Unknown is not zero; these are frozen export costs, not the current arm balance.'));
    const costs = table(['Arm / dimensions','Charged slot-ms','Reserved slot-ms','Measured slot-ms','Unknown / pending attempts','Cost sources'], comparisonRows(aggregate,horizon).map(arm => [arm.arm_id + ' ' + dimensionsLabel(arm.comparison_dimensions),arm.cost?.charged_ms ?? 'Unavailable',arm.cost?.reserved_ms ?? 'Unavailable',arm.cost?.measured_ms ?? 'Unknown',`${arm.cost?.unknown_attempt_ids?.length ?? '?'} / ${arm.cost?.pending_attempt_ids?.length ?? '?'}`,arm.cost?.source_event_hashes?.join('\n') || 'No frozen source']));
    costs.id = 'comparison-costs'; pane.append(costs);
    pane.append(adjudicationView(aggregate,data.computation_receipt));
    pane.append(comparisonFigure(aggregate,horizon,data.computation_receipt));
  };
  target.append(text('p','Aggregate version: ' + aggregate.aggregate_hash));
  target.append(selector('comparison-horizon','Comparison horizon',[['all','All horizons'],...horizons.map(h => [h,h])], () => {horizon = el('comparison-horizon').value; draw();}));
  target.append(pane); draw();
  target.append(action('Export comparison figure', async () => {
    const fresh = JSON.parse(await (await api('/api/artifacts/' + encodeURIComponent(data.package.aggregate_id) + '?download=1',true)).text());
    if (fresh.aggregate_hash !== aggregate.aggregate_hash) throw new Error('Aggregate version changed.');
    const receipt = data.package['computation-receipt'] ? JSON.parse(await (await api('/api/artifacts/' + encodeURIComponent(data.package['computation-receipt']) + '?download=1',true)).text()) : null;
    sourceFigure(comparisonFigure(fresh,horizon,receipt), {artifact_id:data.package.aggregate_id, aggregate_hash:fresh.aggregate_hash,
      spec_hash:fresh.spec_hash, protocol_hash:fresh.protocol_hash, horizon:horizon === 'all' ? null : Number(horizon),
      adjudication:fresh.adjudication ?? null,computation_ref:fresh.computation_ref ?? null,computation_receipt:receipt,
      strata:comparisonRows(fresh,horizon).map(arm => ({stratum_id:arm.stratum_id,dimensions:arm.comparison_dimensions,units:arm.metric_units,cost:arm.cost}))}, 'Frozen comparison');
  }));
  target.append(action('Download frozen CSV', () => download(data.package.table)));
  if (data.package['computation-receipt']) target.append(action('Download computation receipt', async () => saveBlob(await api('/api/artifacts/' + encodeURIComponent(data.package['computation-receipt']) + '?download=1',true),'research-computation-receipt.json')));
  target.append(action('Download aggregate manifest', async () => saveBlob(await api('/api/artifacts/' + encodeURIComponent(data.package.aggregate_id) + '?manifest=1',true),'research-aggregate-manifest.json')));
  target.append(text('h3','Comparison and provenance'),text('pre',JSON.stringify(aggregate,null,2)));
}
async function download(id) {
  const blob = await api('/api/artifacts/' + encodeURIComponent(id) + '?download=1', true);
  const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = 'research-metrics.csv'; link.click(); URL.revokeObjectURL(url);
}
async function detail(row) {
  const target = el('detail-content'); target.replaceChildren(); el('detail').hidden = false;
  if (view === 'comparisons') {
    const data = await api('/api/comparisons/' + encodeURIComponent(row.manifest.aggregate_hash));
    showComparison(target, data);
  } else {
    const data = await api('/api/runs/' + encodeURIComponent(row.manifest.run_id));
    target.append(budgetView(data.budget));
    target.append(text('h3','Trace bindings'),table(['Field','Frozen value'],Object.entries(data.provenance)));
    target.append(text('h3','Checkpoint references'),data.checkpoints.length ? table(['Attempt','Artifact','Recovery level'],data.checkpoints.map(c => [c.attempt_id,c.artifact_id,c.resume_level])) : text('p','No checkpoint recorded for this run.'));
    target.append(text('h3','Study paper evidence references'));
    target.append(text('p','These references share this study; contributing runs and attempts are identified by each frozen evidence package, not by this navigation list.'));
    if (!data.paper_evidence.length) target.append(text('p','No frozen paper evidence imported for this study.'));
    const evidenceSection = document.createElement('section'); let evidenceOpenCount = 0;
    data.paper_evidence.forEach(packageInfo => target.append(action('Open evidence ' + packageInfo.aggregate_hash.slice(0,12),async () => {
      const evidence = await api('/api/comparisons/' + encodeURIComponent(packageInfo.aggregate_hash));
      evidenceSection.replaceChildren(); showComparison(evidenceSection,evidence);
      evidenceSection.dataset.evidenceOpenCount = String(++evidenceOpenCount);
    })));
    target.append(evidenceSection);
    target.append(text('pre', JSON.stringify(data, null, 2)));
    data.attempts.filter(a => a.artifact_id).forEach(attempt => target.append(action('Inspect result ' + attempt.attempt_id.slice(0, 8), async () => {
      await inspectCase(target, attempt.artifact_id);
    })));
  }
  el('detail').scrollIntoView({behavior: 'smooth', block: 'start'});
}
function render() {
  const target = el('results'); target.replaceChildren();
  if (!rows.length) {target.append(text('p', view === 'runs' ? (Object.keys(appliedFilters).length ? 'No runs match the current filters. Reset filters to see all registered cells.' : 'No registered runs or cells are available.') : 'No frozen comparisons yet. Import a validated evidence package to view it here.', 'empty')); return;}
  if (view === 'runs') target.append(table(['Model / arm', 'Version', 'Horizon / seed / block', 'State', 'Budget (whole arm)', ''], rows.map(row => [row.selectors.model + ' / ' + row.manifest.arm_id, row.selectors.version, `${row.selectors.horizon ?? 'Unspecified'} / ${row.manifest.seed} / ${row.manifest.cell.block_id}`, text('span', row.state, 'status ' + row.state.toLowerCase()), `${row.budget.committed_ms} committed / ${row.budget.limit_ms} slot-ms`, row.manifest.run_id ? action('Inspect', () => detail(row)) : 'Registered cell'])));
  else target.append(table(['Study', 'Aggregate version', ''], rows.map(row => [row.manifest.study_id, row.manifest.aggregate_hash, action('Compare & export', () => detail(row))])));
}
async function load(more = false) {
  const sequence = ++loadSequence, selectedView = view;
  el('error').hidden = true;
  if (!token) throw new Error('Open the local session link printed by the serve command.');
  const studies = await api('/api/studies');
  const study = studies.items[0]?.manifest.spec; el('study').textContent = study?.study_id || 'No study';
  el('summary').textContent = study ? `${study.cells.length} registered cells · ${study.arms.length} arms · read-only session` : 'No authorized study found.';
  const params = new URLSearchParams(selectedView === 'runs' ? appliedFilters : {});
  if (more && cursor) params.set('cursor',cursor);
  const page = await api('/api/' + selectedView + '?' + params);
  if (sequence !== loadSequence || selectedView !== view) return;
  rows = more ? rows.concat(page.items) : page.items; cursor = page.next_cursor; el('more').hidden = !cursor; render();
}
document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => {view = button.dataset.view; document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active', b === button)); el('filters').hidden = view !== 'runs'; el('detail').hidden = true; cursor = null; load().catch(showError);});
el('filters').onsubmit = event => {event.preventDefault(); appliedFilters = Object.fromEntries(filterKeys.map(key => [key,el('filter-' + key).value.trim()]).filter(([,value]) => value)); cursor = null; el('detail').hidden = true; load().catch(showError);};
el('reset-filters').onclick = () => {el('filters').reset(); appliedFilters = {}; cursor = null; load().catch(showError);};
el('refresh').onclick = () => load().catch(showError); el('more').onclick = () => load(true).catch(showError); el('close').onclick = () => {el('detail').hidden = true;}; load().catch(showError);
