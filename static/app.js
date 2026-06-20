'use strict';

const NUM_CONFIGS = 4;
const ANSWER_CLAMP_CHARS = 340;

let selectedModel = 'llama-3.1-8b-instant';

// Per-run state
let _fullAnswers = {};       // config_id → full answer text
let _scoreMap   = {};        // config_id → {faithfulness, answer_relevancy, context_precision}
let _scoresIn   = 0;         // how many score events received
let _answersIn  = 0;

// Monitoring state
let _monitorData = null;
let _charts      = {};

// ── Model selector ────────────────────────────────────────────────
function selectModel(btn) {
  document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  selectedModel = btn.dataset.model;
}

// ── Suggested query chip ──────────────────────────────────────────
function setQuery(btn) {
  document.getElementById('query-input').value = btn.textContent.trim();
  document.getElementById('query-input').focus();
}

// ── View toggle (grid / stack) ────────────────────────────────────
function setView(mode) {
  const grid = document.getElementById('results-grid');
  const btnGrid  = document.getElementById('btn-grid');
  const btnStack = document.getElementById('btn-stack');
  if (mode === 'stack') {
    grid.classList.add('stack');
    btnStack.classList.add('active');
    btnGrid.classList.remove('active');
  } else {
    grid.classList.remove('stack');
    btnGrid.classList.add('active');
    btnStack.classList.remove('active');
  }
}

// ── Answer expand / collapse ──────────────────────────────────────
function _maybeClamp(id) {
  const el  = document.getElementById(`answer-${id}`);
  const btn = document.getElementById(`expand-${id}`);
  const txt = _fullAnswers[id] || '';
  if (txt.length > ANSWER_CLAMP_CHARS) {
    el.classList.add('clamped');
    btn.textContent = 'Show more';
    btn.style.display = 'block';
  } else {
    el.classList.remove('clamped');
    btn.style.display = 'none';
  }
}

function toggleExpand(id) {
  const el  = document.getElementById(`answer-${id}`);
  const btn = document.getElementById(`expand-${id}`);
  if (el.classList.contains('clamped')) {
    el.classList.remove('clamped');
    btn.textContent = 'Show less';
  } else {
    el.classList.add('clamped');
    btn.textContent = 'Show more';
  }
}

// ── Copy answer to clipboard ──────────────────────────────────────
function copyAnswer(id) {
  const text = _fullAnswers[id];
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById(`copy-${id}`);
    btn.classList.add('copied');
    btn.title = 'Copied!';
    setTimeout(() => { btn.classList.remove('copied'); btn.title = 'Copy answer'; }, 1800);
  });
}

// ── Reset cards to skeleton state ─────────────────────────────────
function resetCards() {
  _fullAnswers = {}; _scoreMap = {}; _scoresIn = 0; _answersIn = 0;
  document.getElementById('results-section').style.display = 'block';

  for (let i = 1; i <= NUM_CONFIGS; i++) {
    const card = document.getElementById(`card-${i}`);
    card.classList.remove('visible', 'complete', 'best');
    card.style.boxShadow = '';

    // Remove best badge if present
    const badge = card.querySelector('.best-badge');
    if (badge) badge.remove();

    document.getElementById(`answer-${i}`).innerHTML =
      `<div class="skeleton-line"></div>
       <div class="skeleton-line short"></div>
       <div class="skeleton-line"></div>`;
    document.getElementById(`answer-${i}`).classList.remove('clamped');

    const status = document.getElementById(`status-${i}`);
    status.textContent = ''; status.className = 'card-status';

    const lp = document.getElementById(`latency-${i}`);
    lp.textContent = ''; lp.className = 'latency-pill';

    document.getElementById(`expand-${i}`).style.display = 'none';
    document.getElementById(`scores-${i}`).innerHTML = '';
    document.getElementById(`sources-${i}`).innerHTML = '';
    document.getElementById(`copy-${i}`).classList.remove('copied');
  }
}

// ── Show "Scoring…" spinner after all answers arrive ──────────────
function markScoringPending() {
  for (let i = 1; i <= NUM_CONFIGS; i++) {
    const el = document.getElementById(`scores-${i}`);
    if (el && !el.innerHTML.trim()) {
      el.innerHTML = `<span class="score-pending">Scoring…</span>`;
    }
  }
}

// ── Render one result card ────────────────────────────────────────
function renderResult(result) {
  const id   = result.config_id;
  const card = document.getElementById(`card-${id}`);
  card.classList.add('visible');

  const status = document.getElementById(`status-${id}`);

  if (result.error) {
    document.getElementById(`answer-${id}`).innerHTML =
      `<div class="card-error">⚠ ${result.error}</div>`;
    status.textContent = '✗'; status.className = 'card-status error';
    card.classList.add('complete');
    return;
  }

  // Latency badge
  if (result.latency != null) {
    const lp = document.getElementById(`latency-${id}`);
    lp.textContent = result.latency.toFixed(2) + 's';
    lp.className = 'latency-pill visible';
  }

  // Answer (with optional clamp)
  const answerEl = document.getElementById(`answer-${id}`);
  const fullText = result.answer || '(no answer)';
  _fullAnswers[id] = fullText;
  answerEl.textContent = fullText;
  _maybeClamp(id);

  status.textContent = '✓'; status.className = 'card-status done';
  card.classList.add('complete');

  // Sources
  const sources   = result.sources || [];
  const sourcesEl = document.getElementById(`sources-${id}`);
  if (sources.length) {
    sourcesEl.innerHTML = sources.slice(0, 5).map(s =>
      `<span class="source-chip" title="${s}">${s}</span>`
    ).join('');
  }

  _answersIn++;
  if (_answersIn === NUM_CONFIGS) markScoringPending();
}

// ── Render score event ────────────────────────────────────────────
function updateScores(event) {
  const id     = event.config_id;
  const scores = event.scores || {};
  const el     = document.getElementById(`scores-${id}`);
  if (!el) return;

  _scoreMap[id] = scores;
  _scoresIn++;

  const METRICS = [
    ['faithfulness',      'Faithful'],
    ['answer_relevancy',  'Relevancy'],
    ['context_precision', 'Precision'],
  ];

  const rows = METRICS
    .filter(([f]) => scores[f] != null)
    .map(([f, label]) => {
      const v   = scores[f];
      const pct = Math.round(v * 100);
      const col = v >= 0.75 ? 'var(--green)' : v >= 0.5 ? 'var(--amber)' : 'var(--red)';
      return `<div class="metric-row">
        <span class="metric-label">${label}</span>
        <div class="metric-track"><div class="metric-fill" style="width:${pct}%;background:${col}"></div></div>
        <span class="metric-val" style="color:${col}">${v.toFixed(2)}</span>
      </div>`;
    });

  el.innerHTML = rows.length
    ? rows.join('')
    : `<span class="score-note">No retrieval — faithfulness not applicable</span>`;

  if (_scoresIn === NUM_CONFIGS) _highlightBest();
}

// ── Highlight the best-scoring config ────────────────────────────
function _highlightBest() {
  let bestId = -1, bestAvg = -1;

  for (let id = 1; id <= NUM_CONFIGS; id++) {
    const s = _scoreMap[id];
    if (!s) continue;
    const vals = [s.faithfulness, s.answer_relevancy, s.context_precision].filter(v => v != null);
    if (!vals.length) continue;
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    if (avg > bestAvg) { bestAvg = avg; bestId = id; }
  }

  if (bestId < 1) return;
  const card = document.getElementById(`card-${bestId}`);
  card.classList.add('best');

  const badge = document.createElement('span');
  badge.className = 'best-badge';
  badge.innerHTML = `
    <svg width="9" height="9" viewBox="0 0 12 12" fill="currentColor">
      <path d="M6 1l1.5 3 3.5.5-2.5 2.5.6 3.5L6 9l-3.1 1.5.6-3.5L1 4.5l3.5-.5z"/>
    </svg>
    Best`;
  const hdr = card.querySelector('.card-header-right');
  hdr.prepend(badge);
}

// ── Main comparison runner ────────────────────────────────────────
async function runComparison() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
         style="animation:spin 0.9s linear infinite">
      <circle cx="12" cy="12" r="10" stroke-opacity="0.2"/>
      <path d="M12 2a10 10 0 0 1 10 10"/>
    </svg>
    Running…`;

  resetCards();
  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const resp = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, model: selectedModel }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader  = resp.body.getReader();
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
        try {
          const ev = JSON.parse(line.slice(6));
          ev.type === 'score' ? updateScores(ev) : renderResult(ev);
        } catch (_) {}
      }
    }
  } catch (err) {
    console.error('Compare failed:', err);
    for (let i = 1; i <= NUM_CONFIGS; i++) {
      document.getElementById(`answer-${i}`).innerHTML =
        `<div class="card-error">⚠ Request failed — ${err.message}</div>`;
      document.getElementById(`status-${i}`).textContent = '✗';
      document.getElementById(`card-${i}`).classList.add('visible', 'complete');
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M3 2l10 6-10 6V2z"/></svg>
      Run 4 configs`;
  }
}

// ── Monitoring — load and render all 3 charts ─────────────────────
async function loadMonitoring() {
  try {
    const res  = await fetch('/api/monitoring');
    const data = await res.json();
    _monitorData = data;

    document.getElementById('last-run').textContent = data.last_run ? _fmtAgo(data.last_run) : '—';
    document.getElementById('next-run').textContent = data.next_run || '—';

    const driftEl = document.getElementById('drift-banner');
    if (data.alerts && data.alerts.length) {
      driftEl.style.display = 'block';
      driftEl.innerHTML = data.alerts.map(a =>
        `⚠ ${a.config_name}: faithfulness dropped ${(a.drop * 100).toFixed(1)}% in the last 24 h`
      ).join('<br>');
    } else {
      driftEl.style.display = 'none';
    }

    const noData = document.getElementById('no-data-msg');
    if (!data.has_data || !data.series || !data.series.length) {
      noData.style.display = 'block';
      return;
    }
    noData.style.display = 'none';
    _renderAllCharts(data);
  } catch (err) {
    console.error('Monitoring load failed:', err);
  }
}

function _renderAllCharts(data) {
  const allTs = [...new Set(
    data.series.flatMap(s => s.points.map(p => p.ts.slice(5, 16)))
  )].sort();

  const CHART_DEFS = [
    {
      id:          'chart-faithfulness',
      field:       'faithfulness',
      yLabel:      'Score (0–1)',
      yMin: 0, yMax: 1,
      series:      data.series.filter(s => s.config_id !== 1),
    },
    {
      id:          'chart-relevancy',
      field:       'answer_relevancy',
      yLabel:      'Score (0–1)',
      yMin: 0, yMax: 1,
      series:      data.series,
    },
    {
      id:          'chart-latency',
      field:       'latency',
      yLabel:      'Seconds',
      yMin: null, yMax: null,
      series:      data.series,
    },
  ];

  for (const def of CHART_DEFS) {
    if (_charts[def.id]) { _charts[def.id].destroy(); delete _charts[def.id]; }

    const datasets = def.series.map(s => ({
      label:           s.config_name,
      data:            s.points.map(p => ({ x: p.ts.slice(5, 16), y: p[def.field] })),
      borderColor:     s.color,
      backgroundColor: s.color + '15',
      borderWidth:     2,
      tension:         0.35,
      pointRadius:     4,
      pointHoverRadius: 6,
      spanGaps:        true,
      fill:            false,
    }));

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
              color: '#64748b', font: { size: 11, family: 'Inter' },
              boxWidth: 10, boxHeight: 10, padding: 14,
              usePointStyle: true, pointStyle: 'circle',
            },
          },
          tooltip: {
            backgroundColor: '#0c0e14',
            borderColor: 'rgba(255,255,255,0.08)',
            borderWidth: 1,
            titleColor: '#e2e8f0',
            bodyColor: '#8b95a8',
            padding: 10,
            callbacks: {
              label: ctx => {
                const v = ctx.parsed.y;
                return ` ${ctx.dataset.label}: ${v != null ? v.toFixed(3) : '—'}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#404859', font: { size: 10 }, maxTicksLimit: 7, maxRotation: 0 },
            grid:  { color: 'rgba(255,255,255,0.03)' },
          },
          y: {
            min: def.yMin, max: def.yMax,
            title: { display: true, text: def.yLabel, color: '#404859', font: { size: 10 } },
            ticks: { color: '#404859', font: { size: 10 } },
            grid:  { color: 'rgba(255,255,255,0.03)' },
          },
        },
      },
    });
  }
}

// ── Helpers ───────────────────────────────────────────────────────
function _fmtAgo(iso) {
  try {
    // Server stores UTC without 'Z' — append it so Date parses correctly
    const ts   = iso.endsWith('Z') ? iso : iso + 'Z';
    const mins = Math.round((Date.now() - new Date(ts).getTime()) / 60000);
    if (mins < 0) return 'just now';
    return mins < 60 ? `${mins}m ago` : `${Math.floor(mins / 60)}h ${mins % 60}m ago`;
  } catch (_) { return iso; }
}

// ── Warmup poller ─────────────────────────────────────────────────
async function _checkReady() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    return d.status === 'ok' && d.corpus_ready === true;
  } catch (_) { return false; }
}

async function _initWarmup() {
  const banner = document.getElementById('warmup-banner');
  const btn    = document.getElementById('run-btn');
  const ready  = await _checkReady();
  if (ready) return;

  banner.style.display = 'flex';
  btn.disabled = true;
  btn.title = 'Waiting for backend…';

  const iv = setInterval(async () => {
    if (await _checkReady()) {
      clearInterval(iv);
      banner.style.display = 'none';
      btn.disabled = false;
      btn.title = '';
    }
  }, 3000);
}

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  _initWarmup();
  loadMonitoring();
});
