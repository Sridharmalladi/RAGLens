'use strict';

const ANSWER_CLAMP_CHARS = 340;

// Config metadata the UI needs to build cards. Mirrors config.py CONFIG_*.
const CONFIGS = [
  { id: 1, name: 'No RAG',          sub: 'training data only', color: '#6B7280' },
  { id: 2, name: 'Dense RAG',       sub: 'BGE-small · FAISS',  color: '#60A5FA' },
  { id: 3, name: 'Hybrid RAG',      sub: 'Dense + BM25',       color: '#FBBF24' },
  { id: 4, name: 'Hybrid + Rerank', sub: '+ cross-encoder',    color: '#34D399' },
];
const NUM_CONFIGS = CONFIGS.length;

let CFG = null;                 // /api/config payload
let selectedModels = [];       // model ids the user has picked
let runModels = [];            // snapshot of models for the in-flight comparison
let suggestionPool = [];       // full question pool from the backend

let _answers = {};             // `${mi}-${cid}` -> full answer text
let _scoreMap = {};            // `${mi}-${cid}` -> scores object
let _chunks = {};             // `${mi}-${cid}` -> retrieved chunk list
let _pool = {};               // `${mi}-${cid}` -> rerank pool (config 4)
let _latency = {};            // `${mi}-${cid}` -> seconds
let _pinned = [];             // up to 2 card keys pinned for the answer diff
let _lastQuery = '';
let _lastRunAt = null;

let _monitorData = null;
let _charts = {};
let _hiddenModels = new Set();  // monitoring model ids toggled off

// Retrieval score label per config. Config 2 reports raw L2 distance (lower is
// closer); 3 a normalised 0–1 hybrid score; 4 a cross-encoder rerank score.
const SCORE_LABEL = { 2: 'L2', 3: 'hybrid', 4: 'rerank' };

// ── Theme ─────────────────────────────────────────────────────────────
function _resolveTheme() {
  const saved = localStorage.getItem('raglens-theme');
  if (saved === 'light' || saved === 'dark') return saved;
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function _applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('raglens-theme', theme); } catch (_) {}
  if (_monitorData) _renderAllCharts(_monitorData);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  _applyTheme(current === 'dark' ? 'light' : 'dark');
}

// ── App config ────────────────────────────────────────────────────────
async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    CFG = await res.json();
  } catch (_) {
    CFG = {
      models: [{ id: 'openai/gpt-oss-20b', label: 'gpt-oss-20b', tag: 'fast' }],
      default_model: 'openai/gpt-oss-20b',
      max_models: 1,
      suggestions: [],
    };
  }
  selectedModels = [CFG.default_model];
  suggestionPool = CFG.suggestions || [];
  renderModelPicker();
  shuffleSuggestions();
}

// ── Model picker (multi-select) ───────────────────────────────────────
function renderModelPicker() {
  const host = document.getElementById('model-picker');
  host.innerHTML = '';
  for (const m of CFG.models) {
    const on = selectedModels.includes(m.id);
    const btn = document.createElement('button');
    btn.className = 'model-tab' + (on ? ' active' : '');
    btn.dataset.model = m.id;
    btn.innerHTML =
      `<span class="tab-check" aria-hidden="true"></span>${m.label}` +
      (m.tag ? ` <span class="model-tag">${m.tag}</span>` : '');
    btn.onclick = () => toggleModel(m.id);
    host.appendChild(btn);
  }
  _updatePickerHint();
}

function toggleModel(id) {
  const on = selectedModels.includes(id);
  if (on) {
    if (selectedModels.length === 1) return;           // keep at least one
    selectedModels = selectedModels.filter(x => x !== id);
  } else {
    if (selectedModels.length >= (CFG.max_models || 1)) { _flashPickerHint(); return; }
    // preserve CFG.models order
    const order = CFG.models.map(m => m.id);
    selectedModels = order.filter(x => x === id || selectedModels.includes(x));
  }
  renderModelPicker();
}

function _updatePickerHint() {
  const el = document.getElementById('picker-hint');
  if (!el) return;
  const max = CFG.max_models || 1;
  el.textContent = CFG.models.length > 1
    ? `${selectedModels.length} of ${max} selected · runs 4 configs each`
    : '';
}

function _flashPickerHint() {
  const el = document.getElementById('picker-hint');
  if (!el) return;
  el.textContent = `Up to ${CFG.max_models} models at a time`;
  el.classList.add('flash');
  setTimeout(() => { el.classList.remove('flash'); _updatePickerHint(); }, 1400);
}

// ── Suggested questions (shuffled) ────────────────────────────────────
function _shuffled(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function shuffleSuggestions() {
  const host = document.getElementById('suggestion-chips');
  if (!host) return;
  const picks = _shuffled(suggestionPool).slice(0, 5);
  host.innerHTML = '';
  for (const q of picks) {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.textContent = q;
    btn.onclick = () => setQuery(btn);
    host.appendChild(btn);
  }
}

function setQuery(btn) {
  const input = document.getElementById('query-input');
  input.value = btn.textContent.trim();
  input.focus();
}

// ── View toggle (grid / stack) ────────────────────────────────────────
function setView(mode) {
  const groups = document.getElementById('results-groups');
  document.getElementById('btn-grid').classList.toggle('active', mode !== 'stack');
  document.getElementById('btn-stack').classList.toggle('active', mode === 'stack');
  groups.classList.toggle('groups-stack', mode === 'stack');
}

// ── Card + group construction ─────────────────────────────────────────
const COPY_SVG =
  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
  '<rect x="9" y="9" width="13" height="13" rx="2"/>' +
  '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';

const PIN_SVG =
  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
  '<rect x="3" y="4" width="7" height="16" rx="1"/><rect x="14" y="4" width="7" height="16" rx="1"/></svg>';

function _cardHTML(mi, c) {
  const k = `${mi}-${c.id}`;
  return `
    <div class="config-card" id="card-${k}" style="--cc:${c.color}">
      <div class="card-header">
        <span class="config-dot" style="background:${c.color}"></span>
        <span class="config-name">${c.name}</span>
        <span class="config-sub">${c.sub}</span>
        <div class="card-header-right">
          <span class="latency-pill" id="latency-${k}"></span>
          <button class="pin-btn" id="pin-${k}" title="Pin for answer diff (max 2)">${PIN_SVG}</button>
          <button class="copy-btn" id="copy-${k}" title="Copy answer">${COPY_SVG}</button>
          <span class="card-status" id="status-${k}"></span>
        </div>
      </div>
      <div class="card-answer" id="answer-${k}">
        <div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line"></div>
      </div>
      <button class="expand-btn" id="expand-${k}" style="display:none">Show more</button>
      <div class="card-scores" id="scores-${k}"></div>
      <div class="card-sources" id="sources-${k}"></div>
      <div class="card-context" id="context-${k}"></div>
    </div>`;
}

function _modelMeta(id) {
  return (CFG.models || []).find(m => m.id === id) || { id, label: id, tag: '' };
}

function buildGroups(models) {
  const host = document.getElementById('results-groups');
  host.innerHTML = '';
  models.forEach((mid, mi) => {
    const meta = _modelMeta(mid);
    const group = document.createElement('div');
    group.className = 'model-group';
    group.innerHTML =
      `<div class="model-group-head">
         <span class="mg-name">${meta.label}</span>
         ${meta.tag ? `<span class="mg-tag">${meta.tag}</span>` : ''}
         ${models.length > 1 ? `<span class="mg-count">4 configs</span>` : ''}
       </div>
       <div class="results-grid">${CONFIGS.map(c => _cardHTML(mi, c)).join('')}</div>`;
    host.appendChild(group);
  });

  // wire per-card buttons
  models.forEach((_, mi) => {
    CONFIGS.forEach(c => {
      const k = `${mi}-${c.id}`;
      document.getElementById(`copy-${k}`).onclick = () => copyAnswer(k);
      document.getElementById(`expand-${k}`).onclick = () => toggleExpand(k);
      document.getElementById(`pin-${k}`).onclick = () => togglePin(k);
    });
  });

  // fresh DOM: drop any pins and hide the derived panels
  _pinned = [];
  _renderDiff();
  const vd = document.getElementById('run-verdict');
  if (vd) { vd.style.display = 'none'; vd.innerHTML = ''; }
}

// ── Answer expand / collapse ──────────────────────────────────────────
function _maybeClamp(k) {
  const el = document.getElementById(`answer-${k}`);
  const btn = document.getElementById(`expand-${k}`);
  const txt = _answers[k] || '';
  if (txt.length > ANSWER_CLAMP_CHARS) {
    el.classList.add('clamped');
    btn.textContent = 'Show more';
    btn.style.display = 'block';
  } else {
    el.classList.remove('clamped');
    btn.style.display = 'none';
  }
}

function toggleExpand(k) {
  const el = document.getElementById(`answer-${k}`);
  const btn = document.getElementById(`expand-${k}`);
  if (el.classList.contains('clamped')) {
    el.classList.remove('clamped');
    btn.textContent = 'Show less';
  } else {
    el.classList.add('clamped');
    btn.textContent = 'Show more';
  }
}

// ── Copy answer ───────────────────────────────────────────────────────
function copyAnswer(k) {
  const text = _answers[k];
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById(`copy-${k}`);
    btn.classList.add('copied');
    btn.title = 'Copied';
    setTimeout(() => { btn.classList.remove('copied'); btn.title = 'Copy answer'; }, 1800);
  });
}

// ── Render one result card ────────────────────────────────────────────
function renderResult(result) {
  const mi = runModels.indexOf(result.model);
  if (mi < 0) return;
  const k = `${mi}-${result.config_id}`;
  const card = document.getElementById(`card-${k}`);
  if (!card) return;
  card.classList.add('visible');

  const status = document.getElementById(`status-${k}`);

  if (result.error) {
    document.getElementById(`answer-${k}`).innerHTML =
      `<div class="card-error">⚠ ${result.error}</div>`;
    status.textContent = '✗';
    status.className = 'card-status error';
    card.classList.add('complete');
    return;
  }

  _latency[k] = result.latency;
  _chunks[k] = result.chunks || [];
  _pool[k] = result.rerank_pool || [];

  if (result.latency != null) {
    const lp = document.getElementById(`latency-${k}`);
    lp.textContent = result.latency.toFixed(2) + 's';
    lp.className = 'latency-pill visible';
  }

  const costEl = document.getElementById(`cost-${k}`);
  if (costEl) costEl.outerHTML = '';
  const costText = _fmtCost(result.tokens, result.cost);
  if (costText) {
    document.getElementById(`latency-${k}`).insertAdjacentHTML(
      'afterend', `<span class="cost-pill" id="cost-${k}">${costText}</span>`);
  }

  const answerEl = document.getElementById(`answer-${k}`);
  const fullText = result.answer || '(no answer)';
  _answers[k] = fullText;
  answerEl.innerHTML = _renderMarkdown(fullText);
  _maybeClamp(k);

  status.textContent = '✓';
  status.className = 'card-status done';
  card.classList.add('complete');

  const sources = result.sources || [];
  const sourcesEl = document.getElementById(`sources-${k}`);
  if (sources.length) {
    sourcesEl.innerHTML = sources.slice(0, 5).map(s =>
      `<span class="source-chip" title="${s}">${s}</span>`
    ).join('');
  }

  _renderContext(k, result.config_id);
  _syncPins();

  const scoresEl = document.getElementById(`scores-${k}`);
  if (!scoresEl.innerHTML.trim()) {
    scoresEl.innerHTML = `<span class="score-pending">Scoring…</span>`;
  }
}

// ── Cost / token pill ────────────────────────────────────────────────
function _fmtCost(tokens, cost) {
  const bits = [];
  const t = tokens || {};
  const tot = (t.prompt_tokens || 0) + (t.completion_tokens || 0);
  if (tot > 0) bits.push(tot >= 1000 ? (tot / 1000).toFixed(1) + 'k tok' : tot + ' tok');
  if (cost != null && cost > 0) {
    bits.push(cost < 0.01 ? '$' + cost.toFixed(4) : '$' + cost.toFixed(3));
  }
  return bits.join(' · ');
}

// ── Retrieved-context inspector ──────────────────────────────────────
function _fmtScore(v) {
  if (v == null || isNaN(v)) return '–';
  return Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(1);
}

function _renderContext(k, cid) {
  const host = document.getElementById(`context-${k}`);
  if (!host) return;
  const chunks = _chunks[k] || [];
  const pool = _pool[k] || [];
  if (!chunks.length && !pool.length) { host.innerHTML = ''; return; }

  const label = SCORE_LABEL[cid] || 'score';
  const scoreKey = cid === 4 ? 'rerank_score' : 'retrieval_score';

  const row = (c, i, isPool) => {
    const sc = isPool ? c.rerank_score : c[scoreKey];
    const kept = isPool ? !!c.kept : true;
    const txt = c.text || '';
    return `<div class="ctx-row${kept ? '' : ' ctx-drop'}">
      <span class="ctx-rank">${i + 1}</span>
      <div class="ctx-body">
        <div class="ctx-meta">
          <span class="ctx-src" title="${_esc(c.source || '?')}">${_esc(c.source || '?')}</span>
          <span class="ctx-score">${label} ${_fmtScore(sc)}</span>
          ${isPool ? `<span class="ctx-tag ${kept ? 'keep' : 'drop'}">${kept ? 'kept' : 'dropped'}</span>` : ''}
        </div>
        <div class="ctx-text">${_esc(txt)}${txt.length >= 598 ? '…' : ''}</div>
      </div>
    </div>`;
  };

  if (cid === 4 && pool.length) {
    const keptN = pool.filter(c => c.kept).length;
    host.innerHTML = `<details class="ctx">
      <summary>Rerank pool · ${pool.length} candidates, ${keptN} kept</summary>
      <div class="ctx-list">${pool.map((c, i) => row(c, i, true)).join('')}</div>
    </details>`;
  } else {
    host.innerHTML = `<details class="ctx">
      <summary>Retrieved context · ${chunks.length} chunk${chunks.length === 1 ? '' : 's'}</summary>
      <div class="ctx-list">${chunks.map((c, i) => row(c, i, false)).join('')}</div>
    </details>`;
  }
}

// ── Render score event ────────────────────────────────────────────────
function updateScores(event) {
  const mi = runModels.indexOf(event.model);
  if (mi < 0) return;
  const k = `${mi}-${event.config_id}`;
  const scores = event.scores || {};
  const el = document.getElementById(`scores-${k}`);
  if (!el) return;

  _scoreMap[k] = scores;

  const METRICS = [
    ['faithfulness',      'Faithful'],
    ['answer_relevancy',  'Relevancy'],
    ['context_precision', 'Precision'],
  ];

  const reasons = scores.reasons || {};

  const rows = METRICS
    .filter(([f]) => scores[f] != null)
    .map(([f, label]) => {
      const v = scores[f];
      const pct = Math.round(v * 100);
      const col = v >= 0.75 ? 'var(--green)' : v >= 0.5 ? 'var(--amber)' : 'var(--red)';
      const why = reasons[f]
        ? `<div class="metric-reason">${_esc(reasons[f])}</div>` : '';
      return `<div class="metric-row">
        <span class="metric-label">${label}</span>
        <div class="metric-track"><div class="metric-fill" style="width:${pct}%;background:${col}"></div></div>
        <span class="metric-val" style="color:${col}">${v.toFixed(2)}</span>
      </div>${why}`;
    });

  const ungrounded = (scores.unsupported || []).length
    ? `<div class="ungrounded-note">
         <span class="ug-head">Not grounded in retrieved context</span>
         <ul>${scores.unsupported.map(s => `<li>${_esc(s)}</li>`).join('')}</ul>
       </div>`
    : '';

  el.innerHTML = (rows.length
    ? rows.join('')
    : `<span class="score-note">No retrieval, so faithfulness does not apply</span>`
  ) + ungrounded;

  _highlightBest(mi);  // idempotent; only paints once the whole group is scored
  _maybeRenderVerdict();
}

// ── Post-run verdict strip ──────────────────────────────────────────
function _avgScore(s) {
  if (!s) return null;
  const v = [s.faithfulness, s.answer_relevancy, s.context_precision].filter(x => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

function _verdictForModel(mi, mid) {
  let bestId = -1, bestAvg = -1;
  for (const c of CONFIGS) {
    const a = _avgScore(_scoreMap[`${mi}-${c.id}`]);
    if (a != null && a > bestAvg) { bestAvg = a; bestId = c.id; }
  }
  if (bestId < 1) return null;

  const base = _avgScore(_scoreMap[`${mi}-1`]);
  const bestC = CONFIGS.find(c => c.id === bestId);

  let fastId = -1, fastLat = Infinity, latSum = 0, latN = 0;
  for (const c of CONFIGS) {
    const l = _latency[`${mi}-${c.id}`];
    if (l == null) continue;
    latSum += l; latN++;
    if (l < fastLat) { fastLat = l; fastId = c.id; }
  }

  return {
    modelLabel: _modelMeta(mid).label,
    bestName: bestC.name, bestColor: bestC.color, bestAvg,
    delta: base != null ? bestAvg - base : null,
    fastName: (CONFIGS.find(c => c.id === fastId) || {}).name || '–',
    fastLat: isFinite(fastLat) ? fastLat : 0,
    avgLat: latN ? latSum / latN : 0,
  };
}

function _maybeRenderVerdict() {
  const host = document.getElementById('run-verdict');
  if (!host || !runModels.length) return;
  if (Object.keys(_scoreMap).length < runModels.length * NUM_CONFIGS) return;

  const cards = runModels.map((mid, mi) => _verdictForModel(mi, mid)).filter(Boolean);
  if (!cards.length) { host.style.display = 'none'; return; }

  let overall = '';
  if (cards.length > 1) {
    const top = cards.slice().sort((a, b) => b.bestAvg - a.bestAvg)[0];
    overall = `<div class="vd-overall">Strongest overall — <b>${_esc(top.modelLabel)}</b> · ${_esc(top.bestName)} at ${top.bestAvg.toFixed(2)} avg</div>`;
  }

  host.innerHTML = overall + `<div class="vd-grid">` + cards.map(c => {
    const d = c.delta;
    const deltaHtml = d == null ? ''
      : `<div class="vd-delta ${d >= 0 ? 'up' : 'down'}">${d >= 0 ? '+' : ''}${d.toFixed(2)} vs No RAG</div>`;
    return `<div class="vd-card" style="--cc:${c.bestColor}">
      <div class="vd-model">${_esc(c.modelLabel)}</div>
      <div class="vd-line"><span class="vd-win">${_esc(c.bestName)}</span> wins · <span class="vd-avg">${c.bestAvg.toFixed(2)} avg</span></div>
      ${deltaHtml}
      <div class="vd-lat">fastest ${_esc(c.fastName)} ${c.fastLat.toFixed(2)}s · avg ${c.avgLat.toFixed(2)}s</div>
    </div>`;
  }).join('') + `</div>`;
  host.style.display = 'block';
}

// ── Lightweight markdown renderer ─────────────────────────────────────
// Handles bold / italic / inline-code, plus graceful fallbacks for lists,
// headings, and the stray markdown table a model sometimes returns.
function _esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function _inline(s) {
  return s
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>')
    .replace(/(^|[^*])\*(?!\s)([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g,         '<code>$1</code>');
}
function _renderMarkdown(text) {
  const lines = String(text).replace(/\r/g, '').split('\n');
  const out = [];
  let para = [];
  const flush = () => { if (para.length) { out.push('<p>' + para.join(' ') + '</p>'); para = []; } };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flush(); continue; }

    // table divider row: | --- | :--: |
    if (/^\|?[\s:|-]+\|[\s:|-]*$/.test(line) && line.includes('-')) continue;
    // table row: collapse cells into a readable line
    if (/^\|.*\|/.test(line)) {
      flush();
      const cells = line.replace(/^\||\|$/g, '').split('|')
        .map(c => _inline(_esc(c.trim()))).filter(Boolean);
      if (cells.length) out.push('<p class="md-row">' + cells.join('<span class="md-mid">·</span>') + '</p>');
      continue;
    }
    // heading -> bold line
    const h = line.match(/^#{1,6}\s+(.*)$/);
    if (h) { flush(); out.push('<p><strong>' + _inline(_esc(h[1])) + '</strong></p>'); continue; }
    // list item -> bullet line
    const li = line.match(/^([-*+]|\d+[.)])\s+(.*)$/);
    if (li) { flush(); out.push('<p class="md-li">' + _inline(_esc(li[2])) + '</p>'); continue; }

    para.push(_inline(_esc(line)));
  }
  flush();
  return out.join('') || '<p></p>';
}

// ── Highlight the strongest config in one model group ────────────────
// Every config is eligible. Each is scored on the average of the metrics it
// has (config 1 only has relevancy, which is fair: on an off-corpus question
// the no-retrieval answer really can be the best one). The badge only shows
// once every config in the group has a score and the leader clears a floor,
// so a group of weak answers gets no winner.
function _highlightBest(mi) {
  const groupCards = CONFIGS.map(c => document.getElementById(`card-${mi}-${c.id}`)).filter(Boolean);
  groupCards.forEach(card => {
    card.classList.remove('best');
    card.querySelector('.best-badge')?.remove();
  });

  let scored = 0, bestId = -1, bestAvg = -1;
  for (const c of CONFIGS) {
    const s = _scoreMap[`${mi}-${c.id}`];
    if (!s) continue;
    const vals = [s.faithfulness, s.answer_relevancy, s.context_precision].filter(v => v != null);
    if (!vals.length) continue;
    scored++;
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    if (avg > bestAvg) { bestAvg = avg; bestId = c.id; }
  }

  if (scored < CONFIGS.length || bestId < 1 || bestAvg < 0.55) return;

  const card = document.getElementById(`card-${mi}-${bestId}`);
  if (!card) return;
  card.classList.add('best');
  const badge = document.createElement('span');
  badge.className = 'best-badge';
  badge.innerHTML = `
    <svg width="9" height="9" viewBox="0 0 12 12" fill="currentColor">
      <path d="M6 1l1.5 3 3.5.5-2.5 2.5.6 3.5L6 9l-3.1 1.5.6-3.5L1 4.5l3.5-.5z"/>
    </svg>
    Best`;
  card.querySelector('.card-header-right').prepend(badge);
}

// ── Pin two cards, diff their answers ───────────────────────────────
function _syncPins() {
  document.querySelectorAll('.config-card').forEach(card => {
    const k = card.id.replace('card-', '');
    const on = _pinned.includes(k);
    card.classList.toggle('pin-on', on);
    document.getElementById(`pin-${k}`)?.classList.toggle('pinned', on);
  });
}

function togglePin(k) {
  const i = _pinned.indexOf(k);
  if (i >= 0) _pinned.splice(i, 1);
  else {
    if (_pinned.length >= 2) _pinned.shift();
    _pinned.push(k);
  }
  _syncPins();
  _renderDiff();
}

function clearPins() {
  _pinned = [];
  _syncPins();
  _renderDiff();
}

function _kLabel(k) {
  const [mi, cid] = k.split('-').map(Number);
  const c = CONFIGS.find(x => x.id === cid);
  return `${c ? c.name : 'Config ' + cid} · ${_modelMeta(runModels[mi]).label}`;
}

// LCS word diff. Tokens keep their trailing whitespace so output re-spaces.
function _wordDiff(aStr, bStr) {
  const cap = 1200;
  const a = String(aStr).split(/(\s+)/).slice(0, cap);
  const b = String(bStr).split(/(\s+)/).slice(0, cap);
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);

  const out = [];
  const push = (tag, s) => { if (s) out.push(tag ? `<${tag}>${_esc(s)}</${tag}>` : _esc(s)); };
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { push('', a[i]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { push('del', a[i]); i++; }
    else { push('ins', b[j]); j++; }
  }
  while (i < n) push('del', a[i++]);
  while (j < m) push('ins', b[j++]);
  return out.join('');
}

function _renderDiff() {
  const panel = document.getElementById('diff-panel');
  if (!panel) return;
  if (_pinned.length < 2) { panel.style.display = 'none'; panel.innerHTML = ''; return; }
  const [ka, kb] = _pinned;
  panel.style.display = 'block';
  panel.innerHTML = `
    <div class="diff-head">
      <span class="diff-title">Answer diff</span>
      <span class="diff-legend"><del>${_esc(_kLabel(ka))}</del> <ins>${_esc(_kLabel(kb))}</ins></span>
      <button class="diff-close" onclick="clearPins()">Clear</button>
    </div>
    <div class="diff-body">${_wordDiff(_answers[ka] || '', _answers[kb] || '')}</div>`;
}

// ── Export the run ─────────────────────────────────────────────────
function _download(name, text, mime) {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function _buildJSON() {
  return {
    query: _lastQuery,
    ran_at: _lastRunAt,
    models: runModels,
    results: runModels.flatMap((mid, mi) => CONFIGS.map(c => {
      const k = `${mi}-${c.id}`;
      return {
        model: mid,
        config_id: c.id,
        config_name: c.name,
        answer: _answers[k] || null,
        latency_s: _latency[k] ?? null,
        scores: _scoreMap[k] || null,
        chunks: _chunks[k] || [],
        rerank_pool: _pool[k] || [],
      };
    })),
    verdict: runModels.map((mid, mi) => _verdictForModel(mi, mid)).filter(Boolean),
  };
}

function _buildMarkdown() {
  const L = [];
  L.push(`# RAGLens run`, '');
  L.push(`**Query:** ${_lastQuery}`);
  L.push(`**When:** ${_lastRunAt}`);
  L.push(`**Models:** ${runModels.join(', ')}`, '');

  runModels.forEach((mid, mi) => {
    const v = _verdictForModel(mi, mid);
    L.push(`## ${_modelMeta(mid).label}`, '');
    if (v) {
      const d = v.delta == null ? '' : ` (${v.delta >= 0 ? '+' : ''}${v.delta.toFixed(2)} vs No RAG)`;
      L.push(`**Verdict:** ${v.bestName} wins at ${v.bestAvg.toFixed(2)} avg${d}. Fastest ${v.fastName} ${v.fastLat.toFixed(2)}s, avg ${v.avgLat.toFixed(2)}s.`, '');
    }
    CONFIGS.forEach(c => {
      const k = `${mi}-${c.id}`;
      const s = _scoreMap[k] || {};
      L.push(`### ${c.name} — ${c.sub}`);
      if (_latency[k] != null) L.push(`- Latency: ${_latency[k].toFixed(2)}s`);
      const sc = ['faithfulness', 'answer_relevancy', 'context_precision']
        .filter(f => s[f] != null)
        .map(f => `${f} ${s[f].toFixed(2)}${s.reasons && s.reasons[f] ? ` (${s.reasons[f]})` : ''}`);
      if (sc.length) L.push(`- Scores: ${sc.join(' · ')}`);
      if ((s.unsupported || []).length) {
        L.push(`- Not grounded in context:`);
        s.unsupported.forEach(u => L.push(`  - ${u}`));
      }
      const srcs = [...new Set((_chunks[k] || []).map(x => x.source).filter(Boolean))];
      if (srcs.length) L.push(`- Sources: ${srcs.join(', ')}`);
      L.push('', (_answers[k] || '_(no answer)_').trim(), '');
    });
  });
  return L.join('\n');
}

function exportRun() {
  if (!runModels.length || !_lastRunAt) return;
  const stamp = _lastRunAt.replace(/[:.]/g, '-');
  _download(`raglens-${stamp}.md`, _buildMarkdown(), 'text/markdown');
  _download(`raglens-${stamp}.json`, JSON.stringify(_buildJSON(), null, 2), 'application/json');
}

// ── Main comparison runner ────────────────────────────────────────────
async function runComparison() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;
  if (!selectedModels.length) return;

  runModels = selectedModels.slice();
  _answers = {}; _scoreMap = {};
  _chunks = {}; _pool = {}; _latency = {}; _pinned = [];
  _lastQuery = query;
  _lastRunAt = new Date().toISOString();
  document.getElementById('export-btn').style.display = 'inline-flex';

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
         style="animation:spin 0.9s linear infinite">
      <circle cx="12" cy="12" r="10" stroke-opacity="0.2"/>
      <path d="M12 2a10 10 0 0 1 10 10"/>
    </svg>
    Running…`;

  document.getElementById('results-section').style.display = 'block';
  buildGroups(runModels);
  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const resp = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, models: runModels }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let ev;
        try { ev = JSON.parse(line.slice(6)); } catch (_) { continue; }

        if (ev.type === 'start') {
          if (Array.isArray(ev.models) && ev.models.length) {
            runModels = ev.models;
            buildGroups(runModels);
          }
        } else if (ev.error) {
          throw new Error(ev.error);
        } else if (ev.type === 'score') {
          updateScores(ev);
        } else {
          renderResult(ev);
        }
      }
    }
  } catch (err) {
    console.error('Compare failed:', err);
    runModels.forEach((_, mi) => {
      CONFIGS.forEach(c => {
        const k = `${mi}-${c.id}`;
        const a = document.getElementById(`answer-${k}`);
        if (a) a.innerHTML = `<div class="card-error">⚠ Request failed. ${err.message}</div>`;
        const s = document.getElementById(`status-${k}`);
        if (s) { s.textContent = '✗'; s.className = 'card-status error'; }
        document.getElementById(`card-${k}`)?.classList.add('visible', 'complete');
      });
    });
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M3 2l10 6-10 6V2z"/></svg>
      Run comparison`;
    shuffleSuggestions();
  }
}

// ── Monitoring ────────────────────────────────────────────────────────
async function loadMonitoring() {
  try {
    const res = await fetch('/api/monitoring');
    const data = await res.json();
    _monitorData = data;

    document.getElementById('last-run').textContent = data.last_run ? _fmtAgo(data.last_run) : 'not yet';
    document.getElementById('next-run').textContent = data.next_run || 'not scheduled';

    const driftEl = document.getElementById('drift-banner');
    if (data.alerts && data.alerts.length) {
      driftEl.style.display = 'block';
      driftEl.innerHTML = data.alerts.map(a => {
        const label = (data.models.find(m => m.id === a.model) || {}).label || a.model;
        return `⚠ ${label} · ${a.config_name}: faithfulness down ${(a.drop * 100).toFixed(1)}% versus the prior window`;
      }).join('<br>');
    } else {
      driftEl.style.display = 'none';
    }

    const noData = document.getElementById('no-data-msg');
    const hasData = data.has_data && data.series && data.series.length;
    document.getElementById('latest-wrap').style.display = hasData ? 'block' : 'none';
    if (!hasData) {
      noData.style.display = 'block';
      document.getElementById('mon-models').innerHTML = '';
      for (const id in _charts) { _charts[id].destroy(); delete _charts[id]; }
      return;
    }
    noData.style.display = 'none';
    renderMonModelToggles(data);
    _renderAllCharts(data);
    _renderLatestMatrix(data);
  } catch (err) {
    console.error('Monitoring load failed:', err);
  }
}

function renderMonModelToggles(data) {
  const host = document.getElementById('mon-models');
  if (!data.models || data.models.length < 2) { host.innerHTML = ''; return; }
  host.innerHTML = '<span class="field-label">Show</span>';
  for (const m of data.models) {
    const on = !_hiddenModels.has(m.id);
    const btn = document.createElement('button');
    btn.className = 'model-tab' + (on ? ' active' : '');
    btn.innerHTML = `<span class="tab-check" aria-hidden="true"></span>${m.label}`;
    btn.onclick = () => {
      if (_hiddenModels.has(m.id)) _hiddenModels.delete(m.id);
      else if (data.models.length - _hiddenModels.size > 1) _hiddenModels.add(m.id);
      renderMonModelToggles(data);
      _renderAllCharts(data);
      _renderLatestMatrix(data);
    };
    host.appendChild(btn);
  }
}

function _modelOrder(data) {
  return (data.models || []).map(m => m.id);
}
function _dashFor(idx) {
  return [[], [6, 4], [2, 3], [4, 2, 1, 2]][idx] || [3, 3];
}

function _chartColors() {
  const light = document.documentElement.getAttribute('data-theme') === 'light';
  return {
    tick:     light ? '#94a3b8' : '#38404f',
    grid:     light ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.04)',
    tooltip:  light ? '#ffffff' : '#07090f',
    ttBorder: light ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)',
    ttTitle:  light ? '#0f172a' : '#e2e8f0',
    ttBody:   light ? '#475569' : '#7d8a9a',
    legend:   light ? '#94a3b8' : '#38404f',
  };
}

function _visibleSeries(data) {
  return data.series.filter(s => !_hiddenModels.has(s.model));
}

function _renderAllCharts(data) {
  const order = _modelOrder(data);
  const vis = _visibleSeries(data);
  const allTs = [...new Set(vis.flatMap(s => s.points.map(p => p.ts.slice(5, 16))))].sort();
  const C = _chartColors();
  const multiModel = (data.models || []).length > 1;

  const CHART_DEFS = [
    { id: 'chart-faithfulness', field: 'faithfulness',     yMin: 0, yMax: 1,
      series: vis.filter(s => s.config_id !== 1), yLabel: 'Score (0 to 1)' },
    { id: 'chart-relevancy',    field: 'answer_relevancy', yMin: 0, yMax: 1,
      series: vis, yLabel: 'Score (0 to 1)' },
    { id: 'chart-latency',      field: 'latency',          yMin: null, yMax: null,
      series: vis, yLabel: 'Seconds' },
  ];

  for (const def of CHART_DEFS) {
    if (_charts[def.id]) { _charts[def.id].destroy(); delete _charts[def.id]; }

    const datasets = def.series.map(s => {
      const mi = order.indexOf(s.model);
      const dash = multiModel ? _dashFor(mi < 0 ? 0 : mi) : [];
      return {
        label:            multiModel ? `${s.model_label} · ${s.config_name}` : s.config_name,
        data:             s.points.map(p => ({ x: p.ts.slice(5, 16), y: p[def.field] })),
        borderColor:      s.color,
        backgroundColor:  s.color + '18',
        borderWidth:      2,
        borderDash:       dash,
        tension:          0.4,
        pointRadius:      3,
        pointHoverRadius: 5,
        pointStyle:       mi === 1 ? 'rectRot' : 'circle',
        pointBackgroundColor: s.color,
        spanGaps:         true,
        fill:             false,
      };
    });

    _charts[def.id] = new Chart(document.getElementById(def.id), {
      type: 'line',
      data: { labels: allTs, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: C.legend,
              font: { size: 11, family: 'Inter' },
              boxWidth: 10, boxHeight: 10, padding: 14,
              usePointStyle: true, pointStyle: 'circle',
            },
          },
          tooltip: {
            backgroundColor: C.tooltip, borderColor: C.ttBorder, borderWidth: 1,
            titleColor: C.ttTitle, bodyColor: C.ttBody, padding: 12, cornerRadius: 8,
            callbacks: {
              label: ctx => {
                const v = ctx.parsed.y;
                return ` ${ctx.dataset.label}: ${v != null ? v.toFixed(3) : 'no data'}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: C.tick, font: { size: 10 }, maxTicksLimit: 7, maxRotation: 0 },
            grid:  { color: C.grid },
          },
          y: {
            min: def.yMin, max: def.yMax,
            title: { display: true, text: def.yLabel, color: C.tick, font: { size: 10 } },
            ticks: { color: C.tick, font: { size: 10 } },
            grid:  { color: C.grid },
          },
        },
      },
    });
  }
}

// ── Latest-scores matrix ─────────────────────────────────────────────
function _renderLatestMatrix(data) {
  const host = document.getElementById('latest-matrix');
  const models = (data.models || []).filter(m => !_hiddenModels.has(m.id));
  if (!models.length) { host.innerHTML = ''; return; }

  const last = {};  // `${model}-${cid}` -> latest point
  for (const s of data.series) {
    if (!s.points.length) continue;
    last[`${s.model}-${s.config_id}`] = s.points[s.points.length - 1];
  }

  const cell = (mid, cid) => {
    const p = last[`${mid}-${cid}`];
    if (!p) return '<td class="lm-cell lm-empty">no run</td>';
    const bits = [];
    if (p.faithfulness != null) bits.push(`<span class="lm-k">F</span>${p.faithfulness.toFixed(2)}`);
    if (p.answer_relevancy != null) bits.push(`<span class="lm-k">R</span>${p.answer_relevancy.toFixed(2)}`);
    if (p.latency != null) bits.push(`<span class="lm-lat">${p.latency.toFixed(2)}s</span>`);
    return `<td class="lm-cell">${bits.join('<span class="lm-sep">·</span>')}</td>`;
  };

  const head = ['<th class="lm-corner"></th>']
    .concat(models.map(m => `<th class="lm-model">${m.label}</th>`)).join('');
  const body = CONFIGS.map(c =>
    `<tr><th class="lm-config"><span class="lm-dot" style="background:${c.color}"></span>${c.name}</th>` +
    models.map(m => cell(m.id, c.id)).join('') + '</tr>'
  ).join('');

  host.innerHTML =
    `<table class="lm-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
     <div class="lm-legend"><span class="lm-k">F</span> faithfulness
       <span class="lm-k">R</span> relevancy · trailing value is latency</div>`;
}

// ── Helpers ───────────────────────────────────────────────────────────
function _fmtAgo(iso) {
  try {
    const ts = iso.endsWith('Z') ? iso : iso + 'Z';
    const mins = Math.round((Date.now() - new Date(ts).getTime()) / 60000);
    if (mins < 0) return 'just now';
    return mins < 60 ? `${mins}m ago` : `${Math.floor(mins / 60)}h ${mins % 60}m ago`;
  } catch (_) { return iso; }
}

// ── Warmup poller ─────────────────────────────────────────────────────
async function _checkReady() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    return d.status === 'ok' && d.corpus_ready === true;
  } catch (_) { return false; }
}

async function _initWarmup() {
  const banner = document.getElementById('warmup-banner');
  const btn = document.getElementById('run-btn');
  if (await _checkReady()) return;

  banner.style.display = 'flex';
  btn.disabled = true;
  btn.title = 'Waiting for the backend';

  const iv = setInterval(async () => {
    if (await _checkReady()) {
      clearInterval(iv);
      banner.style.display = 'none';
      btn.disabled = false;
      btn.title = '';
    }
  }, 3000);
}

// ── Keyboard ──────────────────────────────────────────────────────────
function _onQueryKey(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    runComparison();
  }
}

// ── Init ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  _applyTheme(_resolveTheme());
  document.getElementById('query-input').addEventListener('keydown', _onQueryKey);
  loadConfig();
  _initWarmup();
  loadMonitoring();
});
