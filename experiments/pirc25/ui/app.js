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
function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
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
  inspectCase.views ??= new WeakMap();
  inspectCase.views.get(target)?.dispose();
  let previewURL = null, drawSequence = 0;
  const state = {active:true, dispose:() => {state.active = false; ++drawSequence; if (previewURL) {URL.revokeObjectURL(previewURL); previewURL = null;}}};
  inspectCase.views.set(target, state);
  const data = await api('/api/cases/' + encodeURIComponent(artifactId));
  if (!state.active) return;
  const result = data.result;
  const section = document.createElement('section'), chart = document.createElement('div');
  const horizons = result.forecast?.horizons;
  let selected = null;
  const draw = async () => {
    const sequence = ++drawSequence, selection = selected;
    if (previewURL) {URL.revokeObjectURL(previewURL); previewURL = null;}
    chart.replaceChildren(); chart.id = 'case-chart';
    chart.dataset.horizon = selection === null ? 'all' : String(horizons[selection]);
    try {
      const entry = frozenCaseEntry(data, artifactId, selection);
      if (!entry) {chart.append(text('p','Frozen case figure unavailable in this version. Saved values remain below; use the authorized offline render-case command to prepare figures.')); return;}
      const blob = await frozenFigureBlob(entry);
      if (!state.active || sequence !== drawSequence) return;
      previewURL = URL.createObjectURL(blob);
      const image = document.createElement('img'); image.src = previewURL;
      image.alt = 'Frozen saved case paths; display only'; image.dataset.artifactId = entry.sha256;
      image.dataset.horizon = selection === null ? 'all' : String(entry.horizon);
      image.dataset.horizonIndex = selection === null ? 'all' : String(selection);
      chart.append(image);
    } catch (error) {if (state.active && sequence === drawSequence) chart.append(text('p','Frozen case figure unavailable: ' + error.message));}
  };
  if (Array.isArray(horizons) && horizons.length) section.append(selector('case-horizon', 'Case horizon', [['all','All horizons'], ...horizons.map((h,i) => [String(i), `${h} ${result.time_unit || '(time unit unspecified)'}`])], async () => {const value = el('case-horizon').value; selected = value === 'all' ? null : Number(value); await draw();}));
  else section.append(text('p', 'No explicit horizon grid in this artifact; horizon switching is unavailable.'));
  section.append(chart);
  const preview = document.createElement('section'); preview.id = 'preview-provenance';
  preview.append(text('h3','Preview provenance'));
  const policy = result.forecast?.preview || {};
  preview.append(table(['Declared field','Value'], ['case_selection_rule','sample_selection_rule','sample_ids','generation_version','n_samples'].map(key => [key,policy[key] === undefined ? 'Unavailable — not declared in this artifact' : JSON.stringify(policy[key])])));
  preview.append(text('p',`${result.forecast?.samples?.length ?? 0} saved paths; displayed only when a verified frozen graph exists. No best-case selection or metric recomputation is performed by this viewer.`));
  section.append(preview);
  const cost = data.computation_receipt?.cost, trace = document.createElement('section'); trace.id = 'case-computation';
  trace.append(text('h3','Frozen case graph source & cost'));
  trace.append(text('p',`Source ${artifactId} · ${cost?.charged_ms ?? 'Unknown'} ${cost?.unit || 'slot-ms'} · charged arm ${cost?.arm_id || 'Unknown'}. Graph cost is measured once for the whole saved-path job, not repeated on horizon switching.`));
  trace.append(text('pre',JSON.stringify({computation_ref:data.package?.computation_ref ?? null,computation_receipt:data.computation_receipt ?? null,figure_index:data.figure_index ?? null},null,2)));
  section.append(trace);
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
    const entry = frozenCaseEntry(data, artifactId, selected);
    if (!entry) throw new Error('Frozen case figure unavailable in this version.');
    saveBlob(await frozenFigureBlob(entry, true), entry.filename);
  }));
  section.append(text('pre', JSON.stringify(result, null, 2))); target.querySelector('[data-case]')?.remove(); section.dataset.case = 'true'; target.append(section);
  await draw();
}
function frozenCaseEntry(data, sourceId, selected) {
  if (data.figure_status === 'UNAVAILABLE') return null;
  const index = data.figure_index, result = data.result, published = data.package, receipt = data.computation_receipt;
  const horizons = result.forecast?.horizons;
  if (data.schema_version !== 'pirc25-case-view-v1' || data.case_id !== sourceId || data.figure_status !== 'AVAILABLE' ||
      index?.schema_version !== 'pirc25-case-figure-index-v1' || index.source_artifact_id !== sourceId ||
      ['spec_hash','cell_hash','protocol_hash'].some(key => index[key] !== result[key]) ||
      published?.schema_version !== 'pirc25-case-package-v1' || published.case_id !== sourceId ||
      JSON.stringify(published.figure_index) !== JSON.stringify(index) ||
      !index.computation_ref || JSON.stringify(index.computation_ref) !== JSON.stringify(published.computation_ref) ||
      receipt?.schema_version !== 'pirc25-case-graph-receipt-v1' ||
      JSON.stringify(receipt.computation_ref) !== JSON.stringify(index.computation_ref) ||
      !Array.isArray(horizons) || horizons.length < 1 || horizons.length > 512 ||
      !Array.isArray(index.figures) || index.figures.length !== horizons.length + 1 ||
      !Array.isArray(published.figures) || published.figures.length !== index.figures.length) throw new Error('Frozen case index mismatch.');
  const seen = new Set();
  for (const entry of index.figures) {
    const ordinal = entry.horizon_index, hash = entry.sha256;
    const matches = published.figures.filter(value => value.sha256 === hash);
    if ((ordinal !== null && (!Number.isSafeInteger(ordinal) || ordinal < 0 || ordinal >= horizons.length)) ||
        entry.horizon !== (ordinal === null ? null : horizons[ordinal]) || seen.has(ordinal) ||
        typeof hash !== 'string' || !/^[0-9a-f]{64}$/.test(hash) || entry.filename !== `case-figure-${hash}.svg` ||
        entry.kind !== 'case' || entry.media_type !== 'image/svg+xml' ||
        !Number.isSafeInteger(entry.size_bytes) || entry.size_bytes < 1 || entry.size_bytes > 2*1024*1024 ||
        matches.length !== 1 || matches[0].artifact_id !== hash ||
        ['filename','kind','horizon','horizon_index','size_bytes','media_type'].some(key => matches[0][key] !== entry[key])) throw new Error('Frozen case package mismatch.');
    seen.add(ordinal);
  }
  const selectedEntry = index.figures.find(entry => entry.horizon_index === selected);
  if (!selectedEntry) throw new Error('Frozen case selection mismatch.');
  return selectedEntry;
}
function comparisonRows(aggregate, horizon) {
  return aggregate.arms.filter(arm => horizon === 'all' || String(arm.comparison_dimensions?.horizon) === horizon);
}
function adjudicationLines(aggregate, receipt) {
  const decision = aggregate.adjudication;
  if (!decision) return ['Formal adjudication unavailable in this frozen version; descriptive intervals are not a verdict.'];
  const policy = decision.adjudication_spec;
  const declared = value => value ?? 'Unavailable';
  const lines = [`Frozen adjudication ${decision.compare_hash} · ${decision.status} · ${decision.qualification}`,
    'Full fixed comparison family; horizon display filtering does not recompute or subset the decision.'];
  if (policy) lines.push(`Primary ${declared(policy.primary_metric?.name)} (${declared(policy.primary_metric?.unit)}) · ${declared(policy.primary_metric?.direction)} · threshold ${declared(policy.practical_threshold)} · multiplicity ${declared(policy.multiplicity)} · seed pairing ${declared(policy.seed_pairing)}`);
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
  const declared = value => value ?? 'Unavailable';
  if (policy) section.append(table(['Frozen policy field','Value'],[
    ['Primary metric',JSON.stringify(policy.primary_metric)],['Practical threshold',policy.practical_threshold],
    ['Interval',JSON.stringify(policy.interval)],['Multiplicity / fixed family',`${declared(policy.multiplicity)} / ${declared(decision.family_size)}`],
    ['Independent unit / seed policy',`${declared(policy.independent_unit)} / ${declared(policy.seed_aggregation)} / ${declared(policy.seed_pairing)}`],
    ['Minimum seeds / paired blocks',`${declared(policy.minimum_seeds)} / ${declared(policy.minimum_paired_blocks)}`],
    ['Weighted strata',JSON.stringify(policy.contrasts)],['Failure / stopping rules',`${declared(policy.missing_policy)} / ${declared(policy.attempt_policy)} / ${declared(policy.stopping_rule)}`]
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
function frozenComparisonEntry(data, horizon) {
  const index = data.figure_index, aggregate = data.aggregate;
  if (!index) return null;
  if (index.schema_version !== 'pirc25-figure-index-v1' || index.aggregate_hash !== aggregate.aggregate_hash ||
      index.compare_hash !== (aggregate.adjudication?.compare_hash ?? null) ||
      JSON.stringify(index.computation_ref ?? null) !== JSON.stringify(aggregate.computation_ref ?? null) ||
      !Array.isArray(index.figures) || index.figures.length < 1 || index.figures.length > 257 ||
      !Array.isArray(data.package.figures)) throw new Error('Frozen figure index mismatch.');
  const matches = index.figures.filter(entry => (entry.horizon === null ? 'all' : String(entry.horizon)) === horizon);
  if (!matches.length) return null;
  if (matches.length !== 1) throw new Error('Frozen figure selection mismatch.');
  const entry = matches[0], hash = entry.sha256;
  const published = data.package.figures.filter(item => item.sha256 === hash);
  if (typeof hash !== 'string' || !/^[0-9a-f]{64}$/.test(hash) || entry.filename !== `figure-${hash}.svg` ||
      entry.kind !== 'comparison' || entry.media_type !== 'image/svg+xml' ||
      !Number.isSafeInteger(entry.size_bytes) || entry.size_bytes < 1 || entry.size_bytes > 2*1024*1024 ||
      published.length !== 1 || published[0].artifact_id !== hash ||
      ['filename','kind','horizon','size_bytes','media_type'].some(key => published[0][key] !== entry[key])) {
    throw new Error('Frozen figure package mismatch.');
  }
  return entry;
}
async function frozenFigureBlob(entry, exporting = false) {
  if (!entry) throw new Error('Frozen figure unavailable in this version; generate an authorized budgeted comparison.');
  const blob = await api('/api/artifacts/' + encodeURIComponent(entry.sha256) + (exporting ? '?download=1' : ''), true);
  if (blob.type.split(';', 1)[0].trim().toLowerCase() !== 'image/svg+xml' || blob.size !== entry.size_bytes || blob.size > 2*1024*1024) {
    throw new Error('Frozen figure response mismatch.');
  }
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())),
    byte => byte.toString(16).padStart(2, '0')).join('');
  if (hash !== entry.sha256) throw new Error('Frozen figure hash mismatch.');
  return blob;
}
function showComparison(target, data) {
  const aggregate = data.aggregate, pane = document.createElement('div'); let horizon = 'all', drawSequence = 0, previewURL = null;
  const horizons = [...new Set(aggregate.arms.map(arm => arm.comparison_dimensions?.horizon).filter(h => h !== undefined).map(String))].sort((a,b) => Number(a)-Number(b));
  const draw = async () => {
    const sequence = ++drawSequence, selectedHorizon = horizon;
    if (previewURL) {URL.revokeObjectURL(previewURL); previewURL = null;}
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
    const figure = text('div','Loading frozen worker figure…'); figure.id = 'comparison-figure'; pane.append(figure);
    try {
      const entry = frozenComparisonEntry(data, selectedHorizon), blob = await frozenFigureBlob(entry);
      if (sequence !== drawSequence) return;
      const image = document.createElement('img'), url = URL.createObjectURL(blob); previewURL = url;
      image.alt = `Frozen comparison ${aggregate.aggregate_hash} · horizon ${selectedHorizon}`;
      image.dataset.artifactId = entry.sha256; image.dataset.horizon = selectedHorizon;
      image.onload = () => {URL.revokeObjectURL(url); if (previewURL === url) previewURL = null;};
      image.onerror = () => {image.onload(); if (sequence === drawSequence) figure.replaceChildren(text('p','Frozen figure unavailable: image decoding failed.'));};
      image.src = url; figure.replaceChildren(image);
    } catch (error) {
      if (sequence === drawSequence) figure.replaceChildren(text('p','Frozen figure unavailable: ' + error.message));
    }
  };
  target.append(text('p','Aggregate version: ' + aggregate.aggregate_hash));
  target.append(selector('comparison-horizon','Comparison horizon',[['all','All horizons'],...horizons.map(h => [h,h])], () => {horizon = el('comparison-horizon').value; return draw();}));
  target.append(pane); const ready = draw();
  target.append(action('Export comparison figure', async () => {
    const entry = frozenComparisonEntry(data, horizon);
    // Never reuse preview bytes: export must pass a fresh server-side grant check.
    saveBlob(await frozenFigureBlob(entry, true), entry.filename);
  }));
  target.append(action('Download frozen CSV', () => download(data.package.table)));
  if (data.package['computation-receipt']) target.append(action('Download computation receipt', async () => saveBlob(await api('/api/artifacts/' + encodeURIComponent(data.package['computation-receipt']) + '?download=1',true),'research-computation-receipt.json')));
  target.append(action('Download aggregate manifest', async () => saveBlob(await api('/api/artifacts/' + encodeURIComponent(data.package.aggregate_id) + '?manifest=1',true),'research-aggregate-manifest.json')));
  target.append(text('h3','Comparison and provenance'),text('pre',JSON.stringify(aggregate,null,2)));
  return ready;
}
async function download(id) {
  const blob = await api('/api/artifacts/' + encodeURIComponent(id) + '?download=1', true);
  const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = 'research-metrics.csv'; link.click(); URL.revokeObjectURL(url);
}
async function detail(row) {
  const target = el('detail-content'); inspectCase.views?.get(target)?.dispose(); target.replaceChildren(); el('detail').hidden = false;
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
