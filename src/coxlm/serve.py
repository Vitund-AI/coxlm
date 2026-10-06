"""coxlm-serve: serve one checkpoint over HTTP.

    coxlm-serve --model path/to/model.pt --encoder Qwen/Qwen3.5-4B-Base --port 8000
    # open http://localhost:8000 for the demo page

Endpoints (Python stdlib HTTP server, no framework):

    GET  /              demo page (enter states and questions, see the calibrated answers)
    GET  /health        {"status": "ok", "model", "checkpoint"}
    GET  /v1/models     the served model id
    GET  /demos         the demo page's pre-written examples
    POST /v1/decide     native API: {"states": [...], "questions": {name: {type, instructions, options}}}
                        -> {"answers": [{name: {choice / p_yes / score, confidence, probabilities, ...}}]}
                        (what coxlm.connect() talks to; see coxlm/wire.py)
    POST /v1/systemone  System One compatible endpoint (see coxlm/systemone.py)
    POST /infer         the demo page's endpoint (question list x state list -> a matrix of answers,
                        plus the order / features / multiscore question types)
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import pathlib
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .order import order_items
from .schema import Schema, choice, multilabel, multiscore, questions, score, yesno
from .systemone import answer_systemone
from .wire import decide_response, schema_from_json

# torch is imported lazily (only to load the model and to time GPU work), so the server
# logic runs with any object that has decide(state, schema) and decide_batch(states, schema) -- e.g. a fake model
# in tests
MODEL = None
MODEL_NAME = "coxlm"  # reported by /v1/models and echoed in responses
CHECKPOINT = ""  # path of the loaded checkpoint, reported by /health
LOCK = threading.Lock()
ENCODER = ""  # backbone name, shown on the demo page
MAX_BATCH = 32  # states per forward pass (a long list is chunked so it cannot run out of memory)


def _sync():
    """Wait for queued GPU work, so timings measure the forward pass (a no-op without CUDA)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>coxlm</title><style>
*{box-sizing:border-box} body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
.wrap{max-width:1100px;margin:0 auto;padding:24px 18px 70px}
h1{font-size:20px;margin:0 0 3px} .sub{color:#8a93a6;margin:0;font-size:13px} .sub2{color:#8a93a6;margin:2px 0 0;font-size:12px}
.bench{background:#161922;border:1px solid #2b2f3a;border-radius:10px;padding:12px 14px;margin:28px 0 0}
.bench .bt{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8a93a6;margin-bottom:8px}
.bench .bt b{color:#e6e6e6} .bench table{width:100%;border-collapse:collapse;font-size:13px}
.bench th{text-align:right;color:#8a93a6;font-weight:500;font-size:11px;padding:3px 6px} .bench th:first-child{text-align:left}
.bench td{padding:4px 6px;border-top:1px solid #23262f} .bench td:first-child{color:#cbd2e0}
.bench td:not(:first-child){text-align:right;font-variant-numeric:tabular-nums} .bench .big{color:#22c55e;font-weight:600} .bench small{color:#5b6472}
.cols{display:flex;gap:16px;align-items:flex-start} .cols>section{flex:1;min-width:0}
.shead{display:flex;justify-content:space-between;align-items:center;margin:0 0 8px}
.shead h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#8a93a6;margin:0}
.add{background:#1e2330;border:1px solid #2b2f3a;color:#cbd2e0;border-radius:7px;padding:4px 10px;font:inherit;font-size:12px;cursor:pointer}
.add:hover{background:#252b3a}
.card{background:#161922;border:1px solid #2b2f3a;border-radius:9px;padding:10px;margin-bottom:8px;position:relative}
.card select,.card input,.card textarea{width:100%;background:#0f1115;border:1px solid #2b2f3a;color:#e6e6e6;border-radius:6px;padding:7px;font:inherit;margin-bottom:6px}
.card textarea{resize:vertical;min-height:112px} .card .row{display:flex;gap:6px} .card .row select{width:110px;flex:none} .card .row input{flex:1}
.card .rm{position:absolute;top:7px;right:8px;background:none;border:0;color:#5b6472;font-size:16px;cursor:pointer;padding:0;line-height:1;width:auto;margin:0}
.card .rm:hover{color:#f87171} .card:last-child{margin-bottom:0}
#go{margin:18px 0 0;background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:11px 26px;font:inherit;font-weight:600;cursor:pointer;font-size:15px}
#go:disabled{opacity:.5;cursor:wait}
#out{margin-top:22px}
.mwrap{overflow-x:auto;border:1px solid #2b2f3a;border-radius:10px}
table.m{border-collapse:collapse;width:100%;font-size:13px}
table.m th,table.m td{padding:8px 10px;text-align:left;border-bottom:1px solid #23262f;white-space:nowrap}
table.m thead th{background:#161922;color:#cbd2e0;font-weight:600;position:sticky;top:0;font-size:12px}
table.m thead th .badge{color:#5b6472;font-size:10px;text-transform:uppercase;margin-left:6px}
table.m tbody th{background:#12151c;color:#cbd2e0;font-variant-numeric:tabular-nums;font-weight:500;position:sticky;left:0}
.cell{cursor:pointer;border-radius:5px} .cell .v{font-weight:600} .cell .c{color:#8a93a6;font-size:11px;margin-left:6px;font-variant-numeric:tabular-nums}
.cell.trunc-row{opacity:.6}
.timing{margin-top:14px;color:#8a93a6;font-size:13px} .timing b{color:#3b82f6}
.det{background:#161922;border:1px solid #2b2f3a;border-radius:9px;padding:12px 14px;margin-top:12px}
.det h3{margin:0 0 8px;font-size:13px} .bar{display:flex;align-items:center;gap:8px;margin:5px 0}
.bar .n{width:150px;text-align:right;font-size:12px;color:#cbd2e0;overflow:hidden;text-overflow:ellipsis}
.bar .t{flex:1;background:#0f1115;border-radius:4px;height:18px;overflow:hidden} .bar .f{background:#3b82f6;height:100%} .bar.top .f{background:#22c55e}
.det ol.ord{margin:6px 0;padding-left:22px} .det ol.ord li{padding:2px 0}
.lvls{display:flex;flex-direction:column;gap:6px;margin:6px 0}
.lvl{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.lvl .ln{color:#5b6472;font-size:11px;width:16px;flex:none;text-align:right}
.chip{background:#1e2330;border:1px solid #2b2f3a;border-radius:6px;padding:3px 8px;font-size:12px}
.edges{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.edge{background:#12331f;border:1px solid #1e5233;border-radius:6px;padding:3px 8px;font-size:12px}
.edge.flex{background:#2a2330;border-color:#4a3a52} .edge em{color:#8a93a6;font-style:normal;margin-left:4px}
.edge.unres{background:#3a2323;border-color:#5a3a3a}
.bar .p{width:72px;font-size:12px;color:#cbd2e0;font-variant-numeric:tabular-nums}
.bar .mk{position:absolute;top:-2px;height:22px;width:3px;margin-left:-1px;background:#22c55e;border-radius:2px;box-shadow:0 0 0 1px #0f1115}
.warn{color:#f59e0b;font-size:12px;margin-top:8px} .err{color:#f87171}
.demorow{display:flex;align-items:center;gap:10px;margin:14px 0 4px}
.demorow label{font-size:12px;color:#8a93a6;text-transform:uppercase;letter-spacing:.04em}
.demorow select{background:#161922;border:1px solid #2b2f3a;color:#e6e6e6;border-radius:7px;padding:7px 10px;font:inherit;font-weight:600;cursor:pointer}
.demohint{font-size:12px;color:#5b6472}
</style></head><body><div class="wrap">
<h1>coxlm</h1>
<p class="sub">typed questions in, calibrated probability distributions out &mdash; one forward pass, not token generation &middot; backbone: __ENCODER__</p>
<p class="sub2">state up to ~__MAXTOK__ tokens</p>
<div class="demorow"><label>Demo</label><select id="demo" onchange="loadDemo(this.value)"></select><span class="demohint">picks a matching set of questions &amp; states &mdash; replaces what's below</span></div>
<div class="cols">
<section><div class="shead"><h2>Questions</h2><button class="add" onclick="addQ()">+ Question</button></div><div id="qlist"></div></section>
<section><div class="shead"><h2>States</h2><button class="add" onclick="addS()">+ State</button></div><div id="slist"></div></section>
</div>
<button id="go" onclick="run()">Run</button>
<div id="out"></div>
</div><script>
const $=s=>document.querySelector(s), ce=h=>{const d=document.createElement('div');d.innerHTML=h;return d.firstElementChild};
let LAST=null;
function addQ(t,q,o,mode,scale){
  const c=ce('<div class="card"><button class="rm" onclick="this.parentElement.remove()">&times;</button>'+
    '<div class="row"><select class="qtype"><option>choice</option><option>yesno</option><option>score</option><option value="features">features</option><option>multiscore</option><option>order</option></select>'+
    '<select class="qmode"><option value="score">by position</option><option value="pairwise">pairwise</option></select>'+
    '<input class="qtext" placeholder="question, e.g. How urgent is this?"></div>'+
    '<textarea class="qopts" rows="6" placeholder="options / features, one per line"></textarea>'+
    '<textarea class="qscale" rows="3" placeholder="scale, one per line — e.g. 1: poor"></textarea></div>');
  const qt=c.querySelector('.qtype'); qt.value=t||'choice';
  c.querySelector('.qtext').value=q||''; c.querySelector('.qopts').value=o||''; c.querySelector('.qmode').value=mode||'score'; c.querySelector('.qscale').value=scale||'';
  const sync=()=>{const tv=qt.value;
    c.querySelector('.qopts').style.display=tv==='yesno'?'none':'block';
    c.querySelector('.qmode').style.display=tv==='order'?'block':'none';
    c.querySelector('.qscale').style.display=tv==='multiscore'?'block':'none';
    c.querySelector('.qopts').placeholder=tv==='order'?'items to order, one per line':(tv==='multiscore'?'aspects to rate, one per line':(tv==='features'?'features, one per line':'options, one per line'));};
  qt.onchange=sync; sync(); $('#qlist').appendChild(c);
}
function addS(id,txt){
  const c=ce('<div class="card"><button class="rm" onclick="this.parentElement.remove()">&times;</button>'+
    '<input class="sid" placeholder="id (optional)"><textarea class="stext" rows="6" placeholder="state text..."></textarea></div>');
  c.querySelector('.sid').value=id||''; c.querySelector('.stext').value=txt||''; $('#slist').appendChild(c);
}
function gather(){
  const qs=[...document.querySelectorAll('#qlist .card')].map(c=>({type:c.querySelector('.qtype').value,
    question:c.querySelector('.qtext').value, options:c.querySelector('.qopts').value,
    mode:c.querySelector('.qmode').value, scale:c.querySelector('.qscale').value})).filter(q=>q.question||q.options);
  const ss=[...document.querySelectorAll('#slist .card')].map(c=>({id:c.querySelector('.sid').value||undefined,
    state:c.querySelector('.stext').value})).filter(s=>s.state.trim());
  return {questions:qs, states:ss};
}
function fmt(cell){
  if(cell.kind==='yesno'){const p=cell.p_yes, pos=p>=0.5; return {v:(pos?'yes':'no'), c:((pos?p:1-p)*100).toFixed(0)+'%', a:Math.abs(p-0.5)*2, pos:pos};}
  if(cell.kind==='order') return {v:cell.ordered_items.join(' › '), c:(cell.confidence*100).toFixed(0)+'%', a:cell.confidence, pos:true};
  if(cell.kind==='score') return {v:cell.score.toFixed(2), c:(cell.confidence*100).toFixed(0)+'%', a:cell.confidence, pos:true};
  return {v:cell.choice, c:(cell.confidence*100).toFixed(0)+'%', a:cell.confidence, pos:true};
}
function run(){
  $('#go').disabled=true; $('#out').innerHTML='<p class="timing">running...</p>';
  fetch('/infer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(gather())})
  .then(r=>r.json()).then(d=>{
    if(d.error){$('#out').innerHTML='<p class="err">'+d.error+'</p>';$('#go').disabled=false;return}
    LAST=d;
    let h='<div class="mwrap"><table class="m"><thead><tr><th></th>';
    for(const q of d.questions){const badge=q.type+(q.scale?(' '+q.scale[0]+'–'+q.scale[1]):''); h+='<th>'+q.label.replace(/</g,'&lt;')+'<span class="badge">'+badge+'</span></th>';}
    h+='</tr></thead><tbody>';
    d.results.forEach((row,ri)=>{
      h+='<tr'+(row.truncated?' class="trunc-row"':'')+'><th title="hash '+row.hash+'">'+String(row.id).replace(/</g,'&lt;')+(row.truncated?' &#9888;':'')+'</th>';
      row.cells.forEach((cell,ci)=>{
        const f=fmt(cell); const col=f.pos?'34,197,94':'148,163,184';
        h+='<td class="cell" style="background:rgba('+col+','+(f.a*0.32).toFixed(2)+')" onclick="expand('+ri+','+ci+')">'+
           '<span class="v">'+String(f.v).replace(/</g,'&lt;')+'</span><span class="c">'+f.c+'</span></td>';
      });
      h+='</tr>';
    });
    h+='</tbody></table></div>';
    h+='<p class="timing"><b>'+d.n_decisions+'</b> decisions ('+d.n_states+' states &times; '+d.n_questions+' questions) in <b>'+d.infer_ms.toFixed(0)+' ms</b> &middot; '+d.ms_per_state.toFixed(1)+' ms/state, one batched pass. Click any cell for its distribution.</p>';
    h+='<div id="det"></div>';
    $('#out').innerHTML=h; $('#go').disabled=false;
  }).catch(e=>{$('#out').innerHTML='<p class="err">'+e+'</p>';$('#go').disabled=false});
}
function expand(ri,ci){
  const cell=LAST.results[ri].cells[ci], q=LAST.questions[ci], row=LAST.results[ri];
  let h='<div class="det"><h3>'+String(row.id).replace(/</g,'&lt;')+' &middot; '+q.label.replace(/</g,'&lt;')+'</h3>';
  if(cell.kind==='order'){
    const esc=s=>String(s).replace(/</g,'&lt;'), lab=cell.items;
    if(cell.mode==='score'){
      h+='<p class="timing">recovered order &middot; score mode &middot; confidence '+(cell.confidence*100).toFixed(0)+'%</p>';
      h+='<ol class="ord">'; for(const it of cell.ordered_items) h+='<li>'+esc(it)+'</li>'; h+='</ol>';
      if(cell.positions){
        const es=cell.positions.map(p=>p.expected), lo=Math.min(...es), hi=Math.max(...es), sp=hi-lo||1;
        h+='<p class="timing" style="margin-top:10px">expected position per item (lower = earlier):</p>';
        for(const p of [...cell.positions].sort((a,b)=>a.expected-b.expected))
          h+='<div class="bar"><div class="n">'+esc(p.text)+'</div><div class="t"><div class="f" style="width:'+((p.expected-lo)/sp*100).toFixed(0)+'%"></div></div><div class="p">'+p.expected.toFixed(2)+'</div></div>';
      }
    } else {  // pairwise: the partial-order graph is the primary output
      let sub='pairwise mode';
      if(cell.consistency!=null) sub+=' &middot; consistency '+(cell.consistency*100).toFixed(0)+'%';
      if(cell.linear_extensions!=null) sub+=' &middot; '+cell.linear_extensions+' valid ordering'+(cell.linear_extensions===1?'':'s');
      if(cell.repaired_edges) sub+=' &middot; '+cell.repaired_edges+' edge'+(cell.repaired_edges===1?'':'s')+' repaired';
      if(cell.epsilon!=null) sub+=' &middot; ε='+cell.epsilon;
      h+='<p class="timing">partial order &middot; '+sub+'</p>';
      if(cell.levels){
        h+='<p class="timing" style="margin-top:8px">stages (items on one row can run in parallel):</p><div class="lvls">';
        cell.levels.forEach((g,li)=>{h+='<div class="lvl"><span class="ln">'+(li+1)+'</span>'+g.map(i=>'<span class="chip">'+esc(lab[i])+'</span>').join('')+'</div>';});
        h+='</div>';
      }
      if(cell.ranges){
        const n=lab.length, ex=cell.expected;
        h+='<p class="timing" style="margin-top:10px">feasible position of each item (band = allowed range, marker = most likely spot):</p>';
        // rows top-to-bottom in the inferred order; marker sits at the probability-weighted position
        const idx=ex?[...lab.keys()].sort((a,b)=>ex[a]-ex[b]):cell.order;
        for(const i of idx){const r=cell.ranges[i], fixed=r[0]===r[1];
          const left=(r[0]-1)/n*100, wid=(r[1]-r[0]+1)/n*100, mk=ex?((ex[i]-0.5)/n*100):null;
          h+='<div class="bar"><div class="n">'+esc(lab[i])+'</div><div class="t" style="position:relative">'+
             '<div class="f'+(fixed?' top':'')+'" style="opacity:'+(fixed?1:.3)+';margin-left:'+left+'%;width:'+wid+'%"></div>'+
             (mk!=null&&!fixed?'<div class="mk" style="left:'+mk+'%"></div>':'')+
             '</div><div class="p">'+(fixed?('#'+r[0]):('~'+ex[i].toFixed(1)+' ('+r[0]+'–'+r[1]+')'))+'</div></div>';}
      }
      if(cell.graph&&cell.graph.edges.length){
        h+='<p class="timing" style="margin-top:10px">direct precedences:</p><div class="edges">';
        for(const e of cell.graph.edges) h+='<span class="edge">'+esc(lab[e[0]])+' &rarr; '+esc(lab[e[1]])+' <em>'+(e[2]*100).toFixed(0)+'%</em></span>';
        h+='</div>';
      }
      if(cell.top_orders&&cell.top_orders.length){
        h+='<p class="timing" style="margin-top:10px">most likely orderings (share of all n! paths):</p><ol class="ord">';
        for(const t of cell.top_orders) h+='<li>'+t[0].map(i=>esc(lab[i])).join(' › ')+' <span class="c">'+(t[1]*100).toFixed(t[1]<0.01?2:0)+'%</span></li>';
        h+='</ol>';
      }
      if(cell.flexible_pairs&&cell.flexible_pairs.length){
        h+='<p class="timing" style="margin-top:10px">flexible (either order is fine):</p><div class="edges">';
        for(const e of cell.flexible_pairs) h+='<span class="edge flex">'+esc(lab[e[0]])+' &harr; '+esc(lab[e[1]])+' <em>'+(e[2]*100).toFixed(0)+'%</em></span>';
        h+='</div>';
      }
      if(cell.unresolved_pairs&&cell.unresolved_pairs.length){
        h+='<p class="timing" style="margin-top:10px">unresolved (model is inconsistent here, not indifferent — in a cyclic triple):</p><div class="edges">';
        for(const e of cell.unresolved_pairs) h+='<span class="edge unres">'+esc(lab[e[0]])+' &harr; '+esc(lab[e[1]])+' <em>'+(e[2]*100).toFixed(0)+'%</em></span>';
        h+='</div>';
      }
      if(cell.pairs){
        const O=cell.order;  // rows/cols in the recovered order -> coherent ranking reads upper-triangular
        h+='<details style="margin-top:12px"><summary class="timing" style="cursor:pointer">P( row before column ) — full matrix, in recovered order</summary><div class="mwrap" style="margin-top:8px"><table class="m"><thead><tr><th></th>';
        O.forEach((j,c)=>{h+='<th>'+(c+1)+'. '+esc(lab[j])+'</th>';});
        h+='</tr></thead><tbody>';
        O.forEach((i,r)=>{h+='<tr><th>'+(r+1)+'. '+esc(lab[i])+'</th>';
          O.forEach((j,c)=>{const v=cell.pairs[i][j];
            const below=c<r&&v!=null&&v>0.5;  // ranked against the recovered order: flag it
            h+=(v==null)?'<td style="color:#3a3f4b">&mdash;</td>':'<td style="background:rgba('+(below?'239,68,68':'34,197,94')+','+(Math.abs(v-0.5)*2*0.32).toFixed(2)+')"'+(below?' title="disagrees with the recovered order"':'')+'>'+(v*100).toFixed(0)+'%</td>';});
          h+='</tr>';});
        h+='</tbody></table><p class="timing" style="margin:6px 2px 0">Green above the diagonal = agrees with the order; red below = a pair ranked the other way (what lowers consistency).</p></div></details>';
      }
    }
    if(row.truncated) h+='<p class="warn">&#9888; this state was truncated to fit the context</p>';
    h+='</div>'; $('#det').innerHTML=h; $('#det').scrollIntoView({behavior:'smooth',block:'nearest'}); return;
  }
  const val={}; if(q.levels) for(const[v,lb]of q.levels) val[lb]=v;
  if(cell.kind==='score') h+='<p class="timing">expected score <b>'+cell.score.toFixed(2)+'</b>'+(q.scale?(' on '+q.scale[0]+'–'+q.scale[1]):'')+'</p>';
  const mx=Math.max(...Object.values(cell.probabilities));
  for(const[k,v]of Object.entries(cell.probabilities)){
    const tag=(k in val)?(' <span class="c">('+val[k]+')</span>'):'';
    h+='<div class="bar'+(v===mx?' top':'')+'"><div class="n">'+k.replace(/</g,'&lt;')+tag+'</div><div class="t"><div class="f" style="width:'+(v*100)+'%"></div></div><div class="p">'+(v*100).toFixed(1)+'%</div></div>';}
  if(cell.none!=null&&cell.none>0.01) h+='<p class="warn">none of these: '+(cell.none*100).toFixed(1)+'%</p>';
  if(row.truncated) h+='<p class="warn">&#9888; this state was truncated to fit the context</p>';
  h+='</div>'; $('#det').innerHTML=h; $('#det').scrollIntoView({behavior:'smooth',block:'nearest'});
}
// pre-written demos live in coxlm/demos.json (fetched below), each replacing the
// questions and states so the state always matches the question types
let DEMOS=[];
function loadDemo(idx){
  const d=DEMOS[idx]; if(!d) return;
  $('#qlist').innerHTML=''; $('#slist').innerHTML=''; $('#out').innerHTML=''; LAST=null;
  for(const q of (d.questions||[])) addQ(q.type,q.question||'',q.options||'',q.mode,q.scale);
  for(const s of (d.states||[])) addS(s.id||'',s.state||'');
}
fetch('/demos').then(r=>r.json()).then(list=>{
  if(!Array.isArray(list)){$('#demo').innerHTML='<option>demos.json: '+((list&&list.error)||'invalid')+'</option>';return}
  DEMOS=list; const sel=$('#demo'); sel.innerHTML='';
  DEMOS.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d.name;sel.appendChild(o);});
  if(DEMOS.length) loadDemo(0);
}).catch(e=>{$('#demo').innerHTML='<option>failed to load demos.json</option>';});
document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')run()});
</script></body></html>"""


def parse_options(text):
    opts, desc = [], {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ": " in line:
            name, d = line.split(": ", 1)
            opts.append(name.strip()); desc[name.strip()] = d.strip()
        else:
            opts.append(line)
    return opts, desc


_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*:\s*(.+)$")


def parse_score(text):
    """Score levels, one per line, low -> high. A line 'value: label' gives that level an
    explicit numeric value (the expected score is then read out in those units, and scores
    from different fields can be combined). Plain labels fall back to 1..k. Mixing is an error."""
    labels, values = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NUM.match(line)
        if m:
            values.append(float(m.group(1))); labels.append(m.group(2).strip())
        else:
            values.append(None); labels.append(line)
    have = [v is not None for v in values]
    if any(have) and not all(have):
        raise ValueError("score levels: give a 'value: label' on every line, or on none")
    return labels, (values if all(have) and labels else None)


def _score_field(options_text, question, name=None):
    from dataclasses import replace
    labels, values = parse_score(options_text)
    if len(labels) < 2:
        raise ValueError("score needs at least two levels (one per line)")
    f = score(tuple(labels), instructions=question, values=values)
    return replace(f, name=name) if name is not None else f


def build_schema(q):
    typ = q.get("type", "choice")
    question = q.get("question", "") or "answer"
    if typ == "yesno":
        return questions(answer=yesno(question))
    if typ == "features":
        # each line is a whole feature, taken verbatim (colons are part of the feature, not a delimiter)
        feats = [l.strip() for l in q.get("options", "").splitlines() if l.strip()]
        if not feats:
            raise ValueError("give at least one feature (one per line)")
        return multilabel(feats, question=question or "Is this feature present in the text?")
    if typ == "multiscore":
        aspects = [l.strip() for l in q.get("options", "").splitlines() if l.strip()]
        slabels, svalues = parse_score(q.get("scale", ""))
        if len(slabels) < 2:
            raise ValueError("multi-score needs a scale of at least two levels (one per line)")
        if not aspects:
            raise ValueError("multi-score needs at least one aspect (one per line)")
        return multiscore(aspects, slabels, question=question or "Rate this aspect on the scale.", values=svalues)
    if typ == "score":
        return questions(answer=_score_field(q.get("options", ""), question))
    opts, desc = parse_options(q.get("options", ""))
    if len(opts) < 2:
        raise ValueError("give at least two options (one per line)")
    return questions(answer=choice({o: desc.get(o, "") for o in opts} if desc else tuple(opts), instructions=question))


def _one_field(typ, question, options_text, name):
    from dataclasses import replace
    if typ == "yesno":
        return replace(yesno(question or "answer"), name=name)
    if typ == "features":
        feats = [l.strip() for l in options_text.splitlines() if l.strip()]
        if not feats:
            raise ValueError("features question needs at least one feature")
        # a features question expands to several yes/no fields; caller flattens
        return multilabel(feats, question=question or "Is this feature present in the text?")
    if typ == "score":
        return _score_field(options_text, question, name=name)
    opts, desc = parse_options(options_text)
    if len(opts) < 2:
        raise ValueError(f"question {question!r} needs at least two options")
    crit = {o: desc.get(o, "") for o in opts} if desc else tuple(opts)
    return replace(choice(crit, instructions=question), name=name)


def build_fields(qlist):
    """A list of {type, question, options} -> a Schema (one field per question; a features
    question expands to several) plus per-field metadata for the results matrix."""
    from dataclasses import replace
    fields, meta, seen = [], [], set()
    for i, qi in enumerate(qlist):
        typ = qi.get("type", "choice")
        label = qi.get("question", "") or f"question {i + 1}"
        if typ == "order":
            items = [l.strip() for l in qi.get("options", "").splitlines() if l.strip()]
            meta.append({"name": None, "label": label, "type": "order", "kind": "order",
                         "items": items, "mode": qi.get("mode", "score")})
            continue
        base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:28] or f"q{i + 1}"
        if typ == "features":
            # header shows just the feature NAME (text before the first ": "); the model
            # still reads the whole line, so any "name: description" clarifies the judgment
            feats = [l.strip() for l in qi.get("options", "").splitlines() if l.strip()]
            subs = list(_one_field(typ, label, qi.get("options", ""), base))
            labels = [("Feature: " + f.split(": ", 1)[0].strip())[:48] for f in feats]
        elif typ == "multiscore":
            # one score field per aspect, all on the shared scale; header shows the aspect name
            aspects = [l.strip() for l in qi.get("options", "").splitlines() if l.strip()]
            slabels, svalues = parse_score(qi.get("scale", ""))
            if len(slabels) < 2:
                raise ValueError("multi-score needs a scale of at least two levels (one per line)")
            if not aspects:
                raise ValueError("multi-score needs at least one aspect (one per line)")
            subs = list(multiscore(aspects, slabels, question=label or "Rate this aspect on the scale.", values=svalues))
            labels = [("Aspect: " + a.split(": ", 1)[0].strip())[:48] for a in aspects]
        else:
            subs = [_one_field(typ, label, qi.get("options", ""), base)]
            labels = [label]
        for f, dlabel in zip(subs, labels):
            nm = f.name
            k = 2
            while nm in seen:
                nm = f"{f.name}_{k}"; k += 1
            seen.add(nm)
            fields.append(replace(f, name=nm))
            m = {"name": nm, "label": dlabel, "type": typ, "kind": f.kind}
            if f.kind == "score":
                vals = list(f.values) if f.values is not None else [i + 1 for i in range(len(f.options))]
                m["scale"] = [vals[0], vals[-1]]
                m["levels"] = [[vals[i], f.options[i]] for i in range(len(f.options))]
            meta.append(m)
    return (Schema(tuple(fields)) if fields else None), meta


def _fields(ans):
    return [{"name": name, "kind": a.kind, "probabilities": a.probabilities, "confidence": a.confidence,
             "choice": a.choice, "score": a.score, "p_yes": a.p_yes, "none": a.none} for name, a in ans.items()]


def _tok_count(text):
    tok = getattr(MODEL, "tokenizer", None)
    return len(tok(text, add_special_tokens=False)["input_ids"]) if tok is not None else len(text.split())


def _max_tokens():
    return max(16, getattr(MODEL, "max_length", 2048) - 2)  # the packed encoder truncates the state to this


def infer_matrix(q):
    """questions[] x states[] -> a matrix of calibrated answers, one batched pass."""
    schema, qmeta = build_fields(q["questions"])
    raw = q.get("states") if isinstance(q.get("states"), list) else [q.get("state", "")]
    parsed = [(it.get("id"), it.get("state", "")) if isinstance(it, dict) else (None, it) for it in raw]
    texts = [t for _, t in parsed]
    cap = _max_tokens()
    sync = _sync
    has_fields = schema is not None and len(list(schema)) > 0
    answers = []
    with LOCK:
        sync(); t0 = time.perf_counter()
        if has_fields:
            for i in range(0, len(texts), MAX_BATCH):
                answers.extend(MODEL.decide_batch(texts[i:i + MAX_BATCH], schema))
        else:
            answers = [{} for _ in texts]
        sync(); infer_ms = (time.perf_counter() - t0) * 1000
    order_ms = 0.0
    results = []
    for idx, ((rid, text), ans) in enumerate(zip(parsed, answers)):
        cells = []
        for m in qmeta:
            if m["type"] == "order":
                sync(); _o0 = time.perf_counter()
                with LOCK:
                    r = order_items(MODEL, m["items"], mode=m.get("mode", "score"), instructions=m["label"], context=text)
                sync(); order_ms += (time.perf_counter() - _o0) * 1000
                # the model reads the full item text (name + any description); the graph shows
                # just the name (before the first ": "), unless that would collide
                disp = [it.split(": ", 1)[0].strip() for it in m["items"]]
                if len(set(disp)) < len(disp):
                    disp = list(m["items"])
                positions = [{**p, "text": disp[p["item"]]} for p in (r.get("positions") or [])] or r.get("positions")
                cells.append({"kind": "order", "order": r["order"], "ordered_items": [disp[i] for i in r["order"]],
                              "items": disp, "confidence": r["confidence"], "positions": positions,
                              "pairs": r.get("pairs"), "consistency": r.get("consistency"), "mode": r["mode"],
                              "graph": r.get("graph"), "ranges": r.get("ranges"), "expected": r.get("expected"), "levels": r.get("levels"),
                              "flexible_pairs": r.get("flexible_pairs"), "unresolved_pairs": r.get("unresolved_pairs"),
                              "linear_extensions": r.get("linear_extensions"), "epsilon": r.get("epsilon"),
                              "top_orders": r.get("top_orders"), "repaired_edges": r.get("repaired_edges")})
                continue
            a = ans[m["name"]]
            cells.append({"kind": a.kind, "choice": a.choice, "p_yes": a.p_yes, "score": a.score,
                          "confidence": a.confidence, "probabilities": a.probabilities, "none": a.none})
        results.append({"id": rid if rid is not None else idx, "index": idx,
                        "hash": hashlib.sha1(text.encode()).hexdigest()[:8],
                        "truncated": _tok_count(text) > cap, "cells": cells})
    n_dec = len(results) * len(qmeta)
    return {"questions": qmeta, "results": results, "infer_ms": round(infer_ms + order_ms, 1),
            "n_states": len(results), "n_questions": len(qmeta), "n_decisions": n_dec,
            "ms_per_state": round(infer_ms / max(len(results), 1), 2)}


def infer(q):
    if isinstance(q.get("questions"), list):
        return infer_matrix(q)
    schema = build_schema(q)
    cap = _max_tokens()
    sync = _sync

    if "states" in q:
        # batch: same questions over many states. Each item is a string, or {"id": ..., "state": ...}.
        items = q["states"]
        parsed = [(it.get("id"), it.get("state", "")) if isinstance(it, dict) else (None, it) for it in items]
        texts = [t for _, t in parsed]
        answers = []
        with LOCK:
            sync(); t0 = time.perf_counter()
            for i in range(0, len(texts), MAX_BATCH):  # chunk so a huge list can't OOM
                answers.extend(MODEL.decide_batch(texts[i:i + MAX_BATCH], schema))
            sync(); infer_ms = (time.perf_counter() - t0) * 1000
        results = []
        for idx, ((rid, text), ans) in enumerate(zip(parsed, answers)):
            n = _tok_count(text)
            results.append({"id": rid if rid is not None else idx, "index": idx,
                            "hash": hashlib.sha1(text.encode()).hexdigest()[:8],
                            "truncated": n > cap, "state_tokens": n, "fields": _fields(ans)})
        return {"results": results, "count": len(results), "infer_ms": round(infer_ms, 1),
                "ms_per_state": round(infer_ms / max(len(results), 1), 2)}

    state_text = q.get("state", "")
    n_state = _tok_count(state_text)
    with LOCK:
        sync(); t0 = time.perf_counter()
        ans = MODEL.decide(state_text, schema)
        sync(); infer_ms = (time.perf_counter() - t0) * 1000
    return {"fields": _fields(ans), "infer_ms": round(infer_ms, 1), "truncated": n_state > cap, "state_tokens": n_state, "cap": cap}


def decide(req):
    """POST /v1/decide: {"states": [...], "questions": {name: spec}} -> {"answers": [{name: answer}]}."""
    if not isinstance(req, dict):
        raise ValueError("request body must be a JSON object")
    states = req.get("states")
    if not isinstance(states, list):
        raise ValueError("'states' must be a list (each a string, object or array)")
    schema = schema_from_json(req.get("questions"))
    answers = []
    with LOCK:
        _sync(); t0 = time.perf_counter()
        for i in range(0, len(states), MAX_BATCH):
            answers.extend(MODEL.decide_batch(states[i:i + MAX_BATCH], schema))
        _sync(); infer_ms = (time.perf_counter() - t0) * 1000
    return decide_response(answers, MODEL_NAME, infer_ms)


def _page():
    return (PAGE.replace("__ENCODER__", html.escape(ENCODER or "unknown"))
                .replace("__MAXTOK__", str(_max_tokens())))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, headers=None):
        self.send_response(code); self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        if self.path.startswith("/demos"):
            # read fresh each request so editing demos.json shows up on a page reload, no restart
            try:
                with open(pathlib.Path(__file__).with_name("demos.json"), encoding="utf-8") as fh:
                    self._send(200, fh.read(), "application/json")
            except Exception as e:  # noqa: BLE001
                self._send(200, json.dumps({"error": str(e)}), "application/json")
            return
        if self.path.rstrip("/") == "/health":  # liveness + which checkpoint is loaded (game harnesses read it)
            self._send(200, json.dumps({"status": "ok", "model": MODEL_NAME, "checkpoint": CHECKPOINT}), "application/json")
            return
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, json.dumps({"object": "list",
                                        "data": [{"id": MODEL_NAME, "object": "model"}]}),
                       "application/json")
            return
        self._send(200, _page(), "text/html; charset=utf-8")

    def do_POST(self):
        try:
            self._post()
        finally:  # give back cached activation memory (long states), so the GPU can be shared
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _post(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) or b"{}"
        if self.path.rstrip("/") == "/v1/decide":  # native typed API (coxlm.connect)
            try:
                self._send(200, json.dumps(decide(json.loads(raw))), "application/json")
            except ValueError as e:  # bad request shape (includes malformed JSON)
                self._send(422, json.dumps({"error": {"message": str(e), "type": "invalid_request"}}), "application/json")
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"error": {"message": str(e), "type": "internal"}}), "application/json")
            return
        # System One compatibility endpoint
        if self.path.startswith("/v1/systemone"):
            rid = uuid.uuid4().hex
            hdr = {"x-request-id": rid}
            try:
                req = json.loads(raw)
                resp = answer_systemone(req, infer_matrix, model_name=MODEL_NAME)
                self._send(200, json.dumps(resp), "application/json", hdr)
            except ValueError as e:  # bad request shape -> 422, as the contract specifies
                self._send(422, json.dumps({"error": {"message": str(e), "type": "invalid_request"}}),
                           "application/json", hdr)
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"error": {"message": str(e), "type": "internal"}}),
                           "application/json", hdr)
            return
        # the demo page's endpoint
        try:
            q = json.loads(raw)
            self._send(200, json.dumps(infer(q)), "application/json")
        except Exception as e:  # noqa: BLE001
            self._send(200, json.dumps({"error": str(e)}), "application/json")


def default_model_name(encoder: str) -> str:
    base = os.path.basename(encoder.rstrip("/")).lower()
    return "coxlm-" + re.sub(r"-base$", "", base)


def serve(model, port: int = 8000, host: str = "0.0.0.0", model_name: str | None = None,
          encoder: str = "", checkpoint: str = "") -> ThreadingHTTPServer:
    """Build the HTTP server around an already-loaded model (call .serve_forever() on the result)."""
    global MODEL, MODEL_NAME, CHECKPOINT, ENCODER
    MODEL = model
    ENCODER = encoder
    CHECKPOINT = checkpoint
    MODEL_NAME = model_name or (default_model_name(encoder) if encoder else "coxlm")
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="coxlm-serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint path (model.pt)")
    ap.add_argument("--encoder", default=None, help="backbone, e.g. Qwen/Qwen3.5-4B-Base (default: recorded in the checkpoint)")
    ap.add_argument("--dtype", default=None, help="backbone dtype: bf16 / fp16 / fp32 (default: bf16, as models are trained and evaluated)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-length", type=int, default=2048, help="state token budget")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available)")
    ap.add_argument("--model-name", default=None, help="id reported by /v1/models and echoed in responses (default: from encoder)")
    args = ap.parse_args(argv)
    from .local import load

    print(f"loading {args.model}...", flush=True)
    model = load(args.model, encoder=args.encoder, device=args.device, dtype=args.dtype, max_length=args.max_length)
    httpd = serve(model, port=args.port, host=args.host, model_name=args.model_name,
                  encoder=model.encoder_name, checkpoint=args.model)
    infer({"state": "warmup", "question": "warm?", "type": "yesno", "options": ""})  # compile kernels before the first request
    print(f"ready -> http://localhost:{args.port}  (model {MODEL_NAME}, format {model.prompt_format}, device {model.device})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
