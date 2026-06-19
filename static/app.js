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
  if (v >= 0.8) return '#34D399';
  if (v >= 0.6) return '#FBBF24';
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

    const answer = document.getElementById(`answer-${i}`);
    answer.innerHTML = `
      <div class="skeleton-line"></div>
      <div class="skeleton-line short"></div>
      <div class="skeleton-line"></div>`;
    answer.classList.remove('filled');

    const status = document.getElementById(`status-${i}`);
    status.textContent = '●';
    status.className = 'card-status';

    document.getElementById(`scores-${i}`).innerHTML = '';
    document.getElementById(`sources-${i}`).innerHTML = '';
  }
}

// ── Render a completed result into its card ───────────────────────────
function renderResult(result) {
  const id = result.config_id;
  const card  = document.getElementById(`card-${id}`);
  const status = document.getElementById(`status-${id}`);

  // Animate card in (stagger by config_id order)
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

  // Scores
  const scores = result.scores || {};
  const hasScores = Object.values(scores).some(v => v !== null && v !== undefined);
  const scoresEl = document.getElementById(`scores-${id}`);
  if (hasScores) {
    const metrics = [
      ['faithfulness', 'Faithfulness'],
      ['answer_relevancy', 'Relevancy'],
      ['context_precision', 'Precision'],
    ];
    scoresEl.innerHTML = metrics.map(([field, label]) => {
      const v = scores[field];
      const display = v !== null && v !== undefined ? v.toFixed(2) : '—';
      const color = scoreColor(v);
      return `
        <div class="score-badge">
          <span class="score-label">${label}</span>
          <span class="score-val" style="color:${color}">${display}</span>
        </div>`;
    }).join('');
  } else if (id !== 1) {
    scoresEl.innerHTML = `<span class="score-note">Scores run hourly in monitoring ↓</span>`;
  }

  // Sources
  const sources = result.sources || [];
  const sourcesEl = document.getElementById(`sources-${id}`);
  if (sources.length) {
    sourcesEl.innerHTML = sources.slice(0, 4).map(s =>
      `<span class="source-chip" title="${s}">${s}</span>`
    ).join('');
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

  // Scroll to results smoothly
  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const response = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, model: selectedModel }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const result = JSON.parse(line.slice(6));
            renderResult(result);
          } catch (_) {}
        }
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
let faithChart = null;
let latencyChart = null;

function buildChartOptions(label, yMin, yMax) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: { color: '#94A3B8', font: { size: 11, family: 'Inter' }, boxWidth: 12, padding: 16 },
      },
      tooltip: {
        backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.07)', borderWidth: 1,
        titleColor: '#F1F5F9', bodyColor: '#94A3B8',
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(3) ?? '—'}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#475569', font: { size: 10 }, maxTicksLimit: 8 },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        min: yMin, max: yMax,
        ticks: { color: '#475569', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
    },
  };
}

async function loadMonitoring() {
  try {
    const res = await fetch('/api/monitoring');
    const data = await res.json();

    // Meta
    document.getElementById('last-run').textContent = data.last_run
      ? fmtAgo(data.last_run) : '—';
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

    if (!data.rows || !data.rows.length) {
      document.getElementById('no-data-msg').style.display = 'block';
      return;
    }
    document.getElementById('no-data-msg').style.display = 'none';

    // Group by config name
    const byConfig = {};
    for (const row of data.rows) {
      const n = row.config_name;
      if (!byConfig[n]) byConfig[n] = { ts: [], faith: [], latency: [] };
      byConfig[n].ts.push(row.timestamp.slice(0, 16));
      byConfig[n].faith.push(row.faithfulness);
      byConfig[n].latency.push(row.latency_s);
    }

    const colorMap = {
      'No RAG': CONFIG_COLORS[1],
      'Dense RAG': CONFIG_COLORS[2],
      'Hybrid RAG': CONFIG_COLORS[3],
      'Hybrid + Rerank': CONFIG_COLORS[4],
    };

    const faithDatasets = Object.entries(byConfig).map(([name, d]) => ({
      label: name,
      data: d.faith,
      borderColor: colorMap[name] || '#818CF8',
      backgroundColor: (colorMap[name] || '#818CF8') + '22',
      tension: 0.3, pointRadius: 3,
    }));

    const latDatasets = Object.entries(byConfig).map(([name, d]) => ({
      label: name,
      data: d.latency,
      borderColor: colorMap[name] || '#818CF8',
      backgroundColor: (colorMap[name] || '#818CF8') + '22',
      tension: 0.3, pointRadius: 3,
    }));

    const allTs = [...new Set(Object.values(byConfig).flatMap(d => d.ts))].sort();

    // Destroy existing charts
    if (faithChart) { faithChart.destroy(); faithChart = null; }
    if (latencyChart) { latencyChart.destroy(); latencyChart = null; }

    faithChart = new Chart(
      document.getElementById('faith-chart'),
      { type: 'line', data: { labels: allTs, datasets: faithDatasets },
        options: buildChartOptions('Faithfulness', 0, 1) }
    );

    latencyChart = new Chart(
      document.getElementById('latency-chart'),
      { type: 'line', data: { labels: allTs, datasets: latDatasets },
        options: buildChartOptions('Latency (s)') }
    );

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
