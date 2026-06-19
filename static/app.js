'use strict';

// ── Selected generation model ─────────────────────────────────────────
let selectedModel = 'llama-3.1-8b-instant';

function selectModel(btn) {
  document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  selectedModel = btn.dataset.model;
}

// ── Config colours matching Python config ─────────────────────────────
const CONFIG_COLORS = {
  1: '#6B7280',
  2: '#60A5FA',
  3: '#FBBF24',
  4: '#34D399',
};

// ── Score colour by value ─────────────────────────────────────────────
function scoreColor(v) {
  if (v === null || v === undefined) return '#475569';
  if (v >= 0.75) return '#34D399';
  if (v >= 0.5)  return '#FBBF24';
  return '#F87171';
}

// ── Set query from chip click ─────────────────────────────────────────
function setQuery(btn) {
  document.getElementById('query-input').value = btn.textContent.trim();
  document.getElementById('query-input').focus();
}

// ── Reset cards to loading state ──────────────────────────────────────
function resetCards() {
  document.getElementById('results-section').style.display = 'block';
  for (let i = 1; i <= 4; i++) {
    const card = document.getElementById(`card-${i}`);
    card.classList.remove('visible', 'complete');

    document.getElementById(`answer-${i}`).innerHTML =
      `<div class="skeleton-line"></div>
       <div class="skeleton-line short"></div>
       <div class="skeleton-line"></div>`;
    document.getElementById(`answer-${i}`).classList.remove('filled');

    const status = document.getElementById(`status-${i}`);
    status.textContent = '●';
    status.className = 'card-status';

    document.getElementById(`scores-${i}`).innerHTML = '';
    document.getElementById(`sources-${i}`).innerHTML = '';
  }
}

// ── Show "scoring…" placeholder once answers are all in ───────────────
function markScoringPending() {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`scores-${i}`);
    if (!el.innerHTML.trim()) {
      el.innerHTML = `<span class="score-pending">Scoring…</span>`;
    }
  }
}

// ── Render a completed answer into its card ───────────────────────────
function renderResult(result) {
  const id = result.config_id;
  const card   = document.getElementById(`card-${id}`);
  const status = document.getElementById(`status-${id}`);

  card.classList.add('visible');

  if (result.error) {
    document.getElementById(`answer-${id}`).innerHTML =
      `<div class="card-error">⚠ ${result.error}</div>`;
    status.textContent = '✗';
    status.className = 'card-status error';
    card.classList.add('complete');
    return;
  }

  // Answer text
  const answerEl = document.getElementById(`answer-${id}`);
  answerEl.textContent = result.answer || '(no answer)';
  answerEl.classList.add('filled');

  // Status dot
  status.textContent = '✓';
  status.className = 'card-status done';
  card.classList.add('complete');

  // Sources
  const sources = result.sources || [];
  const sourcesEl = document.getElementById(`sources-${id}`);
  if (sources.length) {
    sourcesEl.innerHTML = sources.slice(0, 4).map(s =>
      `<span class="source-chip" title="${s}">${s}</span>`
    ).join('');
  }
}

// ── Update score badges from a score event ────────────────────────────
function updateScores(event) {
  const id = event.config_id;
  const scores = event.scores || {};
  const scoresEl = document.getElementById(`scores-${id}`);
  if (!scoresEl) return;

  const METRICS = [
    ['faithfulness',      'Faithfulness'],
    ['answer_relevancy',  'Relevancy'],
    ['context_precision', 'Precision'],
  ];

  const badges = METRICS
    .filter(([field]) => scores[field] !== null && scores[field] !== undefined)
    .map(([field, label]) => {
      const v = scores[field];
      const color = scoreColor(v);
      return `<div class="score-badge">
        <span class="score-label">${label}</span>
        <span class="score-val" style="color:${color}">${v.toFixed(2)}</span>
      </div>`;
    });

  if (badges.length) {
    scoresEl.innerHTML = badges.join('');
  } else {
    // Config 1 — no context, no faithfulness/precision
    scoresEl.innerHTML = `<span class="score-note">No retrieval — faithfulness N/A</span>`;
  }
}

// ── Main comparison runner ────────────────────────────────────────────
async function runComparison() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
         style="animation:spin 1s linear infinite">
      <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
      <path d="M12 2a10 10 0 0 1 10 10" stroke-opacity="1"/>
    </svg>
    Running…`;

  resetCards();
  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

  let answersReceived = 0;

  try {
    const response = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, model: selectedModel }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader  = response.body.getReader();
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
          const event = JSON.parse(line.slice(6));
          if (event.type === 'score') {
            updateScores(event);
          } else {
            renderResult(event);
            answersReceived++;
            if (answersReceived === 4) {
              // All answers are in; scoring phase begins on the server
              markScoringPending();
            }
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    console.error('Comparison failed:', err);
    for (let i = 1; i <= 4; i++) {
      document.getElementById(`answer-${i}`).innerHTML =
        `<div class="card-error">⚠ Request failed — ${err.message}</div>`;
      document.getElementById(`status-${i}`).textContent = '✗';
      document.getElementById(`card-${i}`).classList.add('visible', 'complete');
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M3 2l10 6-10 6V2z"/></svg>
      Run 4 configs`;
  }
}

// ── Monitoring charts ─────────────────────────────────────────────────
let faithChart   = null;
let latencyChart = null;

function buildChartOptions(yLabel, yMin, yMax) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'bottom',
        labels: { color: '#94A3B8', font: { size: 11, family: 'Inter' }, boxWidth: 12, padding: 16 },
      },
      tooltip: {
        backgroundColor: '#111827',
        borderColor: 'rgba(255,255,255,0.07)',
        borderWidth: 1,
        titleColor: '#F1F5F9',
        bodyColor: '#94A3B8',
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
        ticks: { color: '#475569', font: { size: 10 }, maxTicksLimit: 7, maxRotation: 0 },
        grid:  { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        min: yMin,
        max: yMax,
        title: { display: true, text: yLabel, color: '#475569', font: { size: 10 } },
        ticks: { color: '#475569', font: { size: 10 } },
        grid:  { color: 'rgba(255,255,255,0.04)' },
      },
    },
  };
}

async function loadMonitoring() {
  try {
    const res  = await fetch('/api/monitoring');
    const data = await res.json();

    // Meta row
    document.getElementById('last-run').textContent = data.last_run ? fmtAgo(data.last_run) : '—';
    document.getElementById('next-run').textContent = data.next_run || '—';

    // Drift alerts
    const driftBanner = document.getElementById('drift-banner');
    if (data.alerts && data.alerts.length) {
      driftBanner.style.display = 'block';
      driftBanner.innerHTML = data.alerts.map(a =>
        `⚠ ${a.config_name}: faithfulness dropped ${(a.drop * 100).toFixed(1)}% in 24h`
      ).join('<br>');
    } else {
      driftBanner.style.display = 'none';
    }

    if (!data.has_data || !data.series || !data.series.length) {
      document.getElementById('no-data-msg').style.display = 'block';
      return;
    }
    document.getElementById('no-data-msg').style.display = 'none';

    // Build Chart.js datasets from server series
    // Faithfulness — skip Config 1 (no retrieval, always null)
    const faithSeries = data.series.filter(s => s.config_id !== 1);
    const faithDatasets = faithSeries.map(s => ({
      label: s.config_name,
      data:  s.points.map(p => ({ x: p.ts.slice(5, 16), y: p.faithfulness })),
      borderColor:     s.color,
      backgroundColor: s.color + '22',
      tension: 0.35, pointRadius: 4, pointHoverRadius: 6, spanGaps: true,
    }));

    // Answer relevancy — all 4 configs
    const relDatasets = data.series.map(s => ({
      label: s.config_name,
      data:  s.points.map(p => ({ x: p.ts.slice(5, 16), y: p.answer_relevancy })),
      borderColor:     s.color,
      backgroundColor: s.color + '22',
      tension: 0.35, pointRadius: 4, pointHoverRadius: 6, spanGaps: true,
    }));

    // Latency — all 4 configs
    const latDatasets = data.series.map(s => ({
      label: s.config_name,
      data:  s.points.map(p => ({ x: p.ts.slice(5, 16), y: p.latency })),
      borderColor:     s.color,
      backgroundColor: s.color + '22',
      tension: 0.35, pointRadius: 4, pointHoverRadius: 6, spanGaps: true,
    }));

    // X labels: union of all timestamps across all series
    const allTs = [...new Set(
      data.series.flatMap(s => s.points.map(p => p.ts.slice(5, 16)))
    )].sort();

    // Destroy old charts before recreating
    if (faithChart)   { faithChart.destroy();   faithChart   = null; }
    if (latencyChart) { latencyChart.destroy(); latencyChart = null; }

    faithChart = new Chart(document.getElementById('faith-chart'), {
      type: 'line',
      data: { labels: allTs, datasets: faithDatasets },
      options: buildChartOptions('Faithfulness (0–1)', 0, 1),
    });

    latencyChart = new Chart(document.getElementById('latency-chart'), {
      type: 'line',
      data: { labels: allTs, datasets: latDatasets },
      options: buildChartOptions('Latency (seconds)'),
    });

  } catch (err) {
    console.error('Monitoring load failed:', err);
  }
}

function fmtAgo(iso) {
  try {
    const delta = (Date.now() - new Date(iso).getTime()) / 1000 / 60;
    if (delta < 60) return `${Math.floor(delta)}m ago`;
    return `${Math.floor(delta / 60)}h ${Math.floor(delta % 60)}m ago`;
  } catch (_) { return iso; }
}

// ── Init ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadMonitoring();
});
