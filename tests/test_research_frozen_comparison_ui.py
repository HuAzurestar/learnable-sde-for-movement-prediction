"""Run the real browser renderer; frozen worker bytes are not a client recipe."""

import json
from pathlib import Path
import subprocess

import pytest


HARNESS = r"""
const vm = require('vm'), assert = require('assert/strict'), crypto = require('crypto');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; }
  append(...v) { this.children.push(...v); }
  replaceChildren(...v) { this.children = v; }
  setAttribute(k,v) { this[k] = v; }
}
const requests = [], generated = [], saved = [], urls = [], selectors = {};
const hash = 'a'.repeat(64), aggregateId = 'b'.repeat(64);
const svg = h => '<svg xmlns="http://www.w3.org/2000/svg"><metadata>{"aggregate_hash":"'+hash+'"}</metadata><text>worker-'+h+'</text></svg>';
const contents = {}, entries = [null,'1','2'].map(horizon => {
  const content = svg(horizon), sha256 = crypto.createHash('sha256').update(content).digest('hex');
  contents[sha256] = content;
  return {filename:'figure-'+sha256+'.svg',sha256,kind:'comparison',horizon,
    size_bytes:Buffer.byteLength(content),media_type:'image/svg+xml'};
});
const aggregate = {aggregate_hash:hash,qualification:'fixture',adjudication:null,computation_ref:null,
  comparisons:[],arms:['1','2'].map(horizon => ({arm_id:'affine',comparison_dimensions:{horizon},
    independent_n:1,status:'complete',metrics:{error:1},metric_units:{error:'m'},
    successful_cells:1,expected_cells:1,dispositions:{}}))};
const data = {aggregate,package:{aggregate_id:aggregateId,figures:entries.map(e=>({...e,artifact_id:e.sha256}))},
  figure_index:{schema_version:'pirc25-figure-index-v1',aggregate_hash:hash,compare_hash:null,
    computation_ref:null,figures:entries}};
let mode = 'normal', pending = [], denial = false;
const context = {Blob,crypto:crypto.webcrypto,TextEncoder,console,
  URL:{createObjectURL:blob=>{urls.push(blob);return 'blob:fixture-'+urls.length;},revokeObjectURL:()=>{}},
  document:{createElement:tag=>new Element(tag),createElementNS:(_,tag)=>{generated.push(tag);return new Element(tag);}},
  text:(tag,v)=>Object.assign(new Element(tag),{textContent:String(v??'')}),
  dimensionsLabel:v=>JSON.stringify(v||{}),table:()=>new Element('table'),
  selector:(id,_,values,changed)=>{const element=new Element('select');element.value=values[0][0];element.changed=changed;selectors[id]=element;return element;},
  el:id=>selectors[id],action:(label,callback)=>Object.assign(new Element('button'),{label,callback}),
  saveBlob:(blob,name)=>saved.push({blob,name}),showError:error=>{throw error;},
  api:async(path,raw)=>{
    requests.push({path,raw});
    if(denial && path.includes('?download=1')) throw new Error('UNAUTHORIZED_DATA');
    const id=path.split('/').pop().split('?')[0];
    if(id===aggregateId) return new Blob([JSON.stringify(aggregate)],{type:'application/json'});
    let content=contents[id];assert.ok(content,'unknown artifact '+id);
    if(mode==='corrupt') content=content.replace('worker','tamper');
    const blob=new Blob([content],{type:mode==='media'?'text/html':mode==='charset'?'image/svg+xml; charset=utf-8':'image/svg+xml'});
    if(mode==='race') return await new Promise(resolve=>pending.push(()=>resolve(blob)));
    return blob;
  }};
vm.createContext(context);vm.runInContext(RENDERERS,context);
const descendants = root => [root,...root.children.flatMap(v=>v instanceof Element?descendants(v):[])];
const messages = root => descendants(root).map(v=>v.textContent||'').join('\n');
const button = root => descendants(root).find(v=>v.label==='Export comparison figure');
const image = root => descendants(root).find(v=>v.tag==='img');
async function settle() {await new Promise(resolve=>setImmediate(resolve));}
"""


def run_script(body):
    source = (Path(__file__).resolve().parents[1] / "experiments/pirc25/ui/app.js").read_text(encoding="utf-8")
    source = source[source.index("function comparisonRows("):source.index("async function detail(")]
    script = "const RENDERERS=" + json.dumps(source) + ";\n" + HARNESS
    script += "\n(async()=>{" + body + "})().catch(e=>{console.error(e.stack);process.exitCode=1;});"
    result = subprocess.run(["node"], input=script, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_preview_and_export_use_exact_frozen_bytes_with_fresh_export_request():
    run_script(r"""
const root=new Element('root');await context.showComparison(root,data);await settle();
assert.ok(requests.some(r=>r.path==='/api/artifacts/'+entries[0].sha256 && r.raw),'frozen SVG never requested');
assert.ok(image(root),'no frozen image');assert.equal(generated.filter(v=>v==='svg').length,0);
await button(root).callback();assert.equal(await saved[0].blob.text(),contents[entries[0].sha256]);
assert.equal(saved[0].name,entries[0].filename);
assert.ok(requests.some(r=>r.path==='/api/artifacts/'+entries[0].sha256+'?download=1' && r.raw));
denial=true;await assert.rejects(button(root).callback(),/UNAUTHORIZED_DATA/);assert.equal(saved.length,1);
""")


def test_horizon_selection_fetches_matching_frozen_artifact_and_download():
    run_script(r"""
const root=new Element('root');await context.showComparison(root,data);await settle();
selectors['comparison-horizon'].value='2';await selectors['comparison-horizon'].changed();await settle();
assert.ok(requests.some(r=>r.path==='/api/artifacts/'+entries[2].sha256),'wrong horizon artifact');
await button(root).callback();assert.equal(await saved[0].blob.text(),contents[entries[2].sha256]);
assert.equal(generated.filter(v=>v==='svg').length,0);
""")


def test_real_service_svg_media_type_with_charset_is_accepted():
    run_script(r"""
mode='charset';const root=new Element('root');await context.showComparison(root,data);await settle();
assert.ok(image(root),'real service MIME parameters must not reject valid SVG');
await button(root).callback();assert.equal(await saved[0].blob.text(),contents[entries[0].sha256]);
""")


@pytest.mark.parametrize("mode", ["corrupt", "media"])
def test_invalid_response_never_becomes_a_preview_or_export(mode):
    run_script("mode=" + json.dumps(mode) + r""";
const root=new Element('root');await context.showComparison(root,data);await settle();
assert.equal(image(root),undefined);assert.equal(generated.filter(v=>v==='svg').length,0);
assert.match(messages(root),/unavailable|mismatch|invalid/i);
await assert.rejects(button(root).callback());assert.equal(saved.length,0);
""")


@pytest.mark.parametrize("mutation", [
    "delete data.figure_index",
    "data.figure_index.aggregate_hash='c'.repeat(64)",
    "data.package.figures[0].artifact_id='d'.repeat(64)",
    "data.figure_index.figures.push({...entries[0]})",
])
def test_absent_or_inconsistent_index_has_no_client_recomputation_fallback(mutation):
    run_script(mutation + r""";
const root=new Element('root');await context.showComparison(root,data);await settle();
assert.equal(image(root),undefined);assert.equal(generated.filter(v=>v==='svg').length,0);
assert.equal(requests.length,0);assert.match(messages(root),/unavailable|mismatch|invalid/i);
await assert.rejects(button(root).callback());assert.equal(saved.length,0);
""")


def test_late_preview_cannot_replace_the_new_horizon():
    run_script(r"""
mode='race';const root=new Element('root');const first=context.showComparison(root,data);await settle();
selectors['comparison-horizon'].value='2';const second=selectors['comparison-horizon'].changed();await settle();
assert.equal(pending.length,2,'both frozen preview requests must exist');
pending[1]();await second;await settle();const current=image(root);assert.ok(current);
pending[0]();await first;await settle();assert.equal(image(root),current);
assert.equal(urls.length,1,'stale preview must not retain a blob URL');
assert.equal(await urls[0].text(),contents[entries[2].sha256]);
""")
