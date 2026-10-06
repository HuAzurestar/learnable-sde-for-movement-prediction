"""Execute the actual case renderer, not a test-only copy of the browser logic."""

import json
from pathlib import Path
import subprocess

import pytest


HARNESS = r"""
const vm=require('vm'),assert=require('assert/strict'),crypto=require('crypto');
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};}
  append(...v){this.children.push(...v);}
  replaceChildren(...v){this.children=v;}
  setAttribute(k,v){this[k]=v;}
  querySelector(){return null;}
}
const requests=[],generated=[],saved=[],urls=[],selectors={},contents={};
const sourceId='a'.repeat(64), reference={manifest_id:'case-receipt-fixture'};
const result={spec_hash:'b'.repeat(64),cell_hash:'c'.repeat(64),protocol_hash:'d'.repeat(64),
  state_order:['x','y','vx','vy'],units:['m','m','m/s','m/s'],time_unit:'s',metrics:{error:2.5},
  forecast:{horizons:[1,2],samples:[[[0,0,1,2],[1,2,3,4]]],preview:{sample_ids:['s0'],generation_version:'fixture'}}};
const entries=[null,0,1].map(horizon_index=>{
  const horizon=horizon_index===null?null:result.forecast.horizons[horizon_index];
  const content='<svg xmlns="http://www.w3.org/2000/svg"><metadata>{"source_artifact_id":"'+sourceId+'"}</metadata><text>worker-'+horizon_index+'</text></svg>';
  const sha256=crypto.createHash('sha256').update(content).digest('hex');contents[sha256]=content;
  return {filename:'case-figure-'+sha256+'.svg',sha256,size_bytes:Buffer.byteLength(content),
    kind:'case',media_type:'image/svg+xml',horizon_index,horizon};
});
const index={schema_version:'pirc25-case-figure-index-v1',source_artifact_id:sourceId,
  spec_hash:result.spec_hash,cell_hash:result.cell_hash,protocol_hash:result.protocol_hash,
  computation_ref:reference,figures:entries};
const data={schema_version:'pirc25-case-view-v1',case_id:sourceId,result,figure_status:'AVAILABLE',
  figure_index:index,package:{schema_version:'pirc25-case-package-v1',case_id:sourceId,computation_ref:reference,
    figure_index:index,figures:entries.map(e=>({...e,artifact_id:e.sha256}))},
  computation_receipt:{schema_version:'pirc25-case-graph-receipt-v1',computation_ref:reference,
    cost:{arm_id:'affine',charged_ms:1234,unit:'slot-ms',basis:'measured-monotonic'}}};
let mode='normal',pending=[],denial=false;
const context={Blob,crypto:crypto.webcrypto,TextEncoder,console,Node:Element,
  URL:{createObjectURL:blob=>{urls.push(blob);return 'blob:fixture-'+urls.length;},revokeObjectURL:()=>{}},
  document:{createElement:tag=>new Element(tag),createElementNS:(_,tag)=>{generated.push(tag);return new Element(tag);}},
  text:(tag,v)=>Object.assign(new Element(tag),{textContent:String(v??'')}),table:()=>new Element('table'),
  selector:(id,_,values,changed)=>{const e=new Element('select');e.value=values[0][0];e.changed=changed;selectors[id]=e;return e;},
  el:id=>selectors[id],action:(label,callback)=>Object.assign(new Element('button'),{label,callback}),
  saveBlob:(blob,name)=>saved.push({blob,name}),showError:error=>{throw error;},
  // Controls expose the old client fallback; production implementations are extracted below.
  forecastPlot:()=>{generated.push('svg');return new Element('svg');},sourceFigure:()=>{saved.push({regenerated:true});},
  api:async(path,raw)=>{
    requests.push({path,raw});
    if(denial&&path.includes('?download=1'))throw new Error('UNAUTHORIZED_DATA');
    if(path==='/api/cases/'+sourceId)return data;
    if(path==='/api/artifacts/'+sourceId)return result;
    if(path==='/api/artifacts/'+sourceId+'?download=1')return new Blob([JSON.stringify(result)],{type:'application/json'});
    const id=path.split('/').pop().split('?')[0];let content=contents[id];assert.ok(content,'unknown artifact '+id);
    if(mode==='corrupt')content=content.replace('worker','tamper');
    const blob=new Blob([content],{type:mode==='media'?'text/html':'image/svg+xml; charset=utf-8'});
    if(mode==='race')return await new Promise(resolve=>pending.push(()=>resolve(blob)));
    return blob;
  }};
vm.createContext(context);vm.runInContext(RENDERERS,context);
const descendants=root=>[root,...root.children.flatMap(v=>v instanceof Element?descendants(v):[])];
const messages=root=>descendants(root).map(v=>v.textContent||'').join('\n');
const button=root=>descendants(root).find(v=>v.label==='Export case figure');
const image=root=>descendants(root).find(v=>v.tag==='img');
async function settle(){await new Promise(resolve=>setImmediate(resolve));}
"""


def run_script(body):
    source = (Path(__file__).resolve().parents[1] / 'experiments/pirc25/ui/app.js').read_text(encoding='utf-8')
    renderers = source[source.index('async function inspectCase('):source.index('function comparisonRows(')]
    # Share the actual existing byte/MIME/hash verifier when present.
    name = 'frozenFigureBlob' if 'async function frozenFigureBlob(' in source else 'frozenComparisonBlob'
    renderers += source[source.index('async function ' + name + '('):source.index('function showComparison(')]
    script = 'const RENDERERS=' + json.dumps(renderers) + ';\n' + HARNESS
    script += '\n(async()=>{' + body + '})().catch(e=>{console.error(e.stack);process.exitCode=1;});'
    executed = subprocess.run(['node'], input=script, capture_output=True, text=True, timeout=15)
    assert executed.returncode == 0, executed.stderr


def test_actual_case_ui_uses_saved_worker_bytes_and_fresh_export_permission():
    run_script(r"""
const root=new Element('root');await context.inspectCase(root,sourceId);
assert.ok(requests.some(r=>r.path==='/api/cases/'+sourceId),'declared case endpoint unused');
assert.ok(image(root),'frozen case image absent');assert.equal(generated.length,0,'client rendered a figure');
await button(root).callback();assert.equal(await saved[0].blob.text(),contents[entries[0].sha256]);
assert.equal(saved[0].name,entries[0].filename);
assert.ok(requests.some(r=>r.path==='/api/artifacts/'+entries[0].sha256+'?download=1'));
denial=true;await assert.rejects(button(root).callback(),/UNAUTHORIZED_DATA/);assert.equal(saved.length,1);
""")


def test_case_horizon_selector_uses_ordinal_and_preserves_saved_scores():
    run_script(r"""
const root=new Element('root');await context.inspectCase(root,sourceId);
const original=JSON.stringify(result);selectors['case-horizon'].value='1';await selectors['case-horizon'].changed();
assert.equal(image(root).dataset.artifactId,entries[2].sha256);
assert.equal(image(root).dataset.horizon,'2');assert.equal(image(root).dataset.horizonIndex,'1');
await button(root).callback();assert.equal(await saved[0].blob.text(),contents[entries[2].sha256]);
assert.equal(JSON.stringify(result),original);assert.equal(generated.length,0);
""")


@pytest.mark.parametrize('mode', ['corrupt', 'media'])
def test_invalid_case_bytes_cannot_be_previewed_or_downloaded(mode):
    run_script('mode=' + json.dumps(mode) + r""";
const root=new Element('root');await context.inspectCase(root,sourceId);
assert.equal(image(root),undefined);assert.match(messages(root),/unavailable|mismatch/i);
await assert.rejects(button(root).callback());assert.equal(saved.length,0);assert.equal(generated.length,0);
""")


@pytest.mark.parametrize('mutation', [
    "data.figure_status='UNAVAILABLE';delete data.figure_index;delete data.package",
    "data.figure_index.source_artifact_id='e'.repeat(64)",
    "data.package.figures[0].artifact_id='e'.repeat(64)",
    "data.figure_index.figures.push({...entries[0]})",
    "data.figure_index.figures[0].horizon=2",
    "data.computation_receipt.computation_ref={manifest_id:'wrong'}",
])
def test_missing_or_mismatched_case_evidence_cannot_trigger_client_fallback(mutation):
    run_script(mutation + r""";
const root=new Element('root');await context.inspectCase(root,sourceId);
assert.equal(image(root),undefined);assert.equal(generated.length,0);
assert.equal(requests.filter(r=>r.raw).length,0);assert.match(messages(root),/unavailable|mismatch/i);
await assert.rejects(button(root).callback());assert.equal(saved.length,0);
""")


def test_late_case_preview_cannot_replace_new_horizon_or_leak_blob_url():
    run_script(r"""
mode='race';const root=new Element('root');const first=context.inspectCase(root,sourceId);await settle();
selectors['case-horizon'].value='1';const second=selectors['case-horizon'].changed();await settle();
assert.equal(pending.length,2);pending[1]();await second;await settle();const current=image(root);assert.ok(current);
pending[0]();await first;assert.equal(image(root),current);assert.equal(urls.length,1);
assert.equal(await urls[0].text(),contents[entries[2].sha256]);
""")
