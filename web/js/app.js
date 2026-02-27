/**
 * Health utility explorer — frontend
 * Loads config (YAML), collects form data, POSTs to /api/calculate, and displays results + charts.
 */

const form = document.getElementById('qaly-form');
const resultsEl = document.getElementById('results');
const resultsContent = document.getElementById('results-content');
const submitBtn = document.getElementById('submit-btn');

let config = null;
let contributionChart = null;
let scenarioChart = null;
let cumulativeChart = null;
let tornadoChart = null;
let scenarioCumulativeChart = null;
let scenarioTornadoChart = null;

async function loadConfig() {
  try {
    const res = await fetch('config.yaml');
    if (!res.ok) throw new Error(`Failed to load config: ${res.status}`);
    const text = await res.text();
    const jsyaml = window.jsyaml || (typeof jsyaml !== 'undefined' ? jsyaml : null);
    if (jsyaml) {
      config = jsyaml.load(text);
    } else {
      config = JSON.parse(text);
    }
    return config;
  } catch (e) {
    console.warn('Config load failed, using defaults:', e.message);
    config = getDefaultConfig();
    return config;
  }
}

function getDefaultConfig() {
  return {
    app: {
      title: 'Health utility explorer',
      subtitle: 'Compare your profile to population data and see how factors affect a utility-style score and quality-adjusted years.',
    },
    utility: { min_score: 0.1, max_score: 1.0, benchmark_age: 80 },
  };
}

function getFormData() {
  const data = {};
  form.querySelectorAll('input, select').forEach((el) => {
    const name = el.name;
    if (!name) return;
    if (el.type === 'number') {
      const v = el.value.trim();
      data[name] = v === '' ? null : Number(v);
    } else {
      data[name] = el.value;
    }
  });
  return data;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

/**
 * Render numeric result and warnings, then draw charts if data present.
 */
function renderResults(result) {
  const u = config?.utility || getDefaultConfig().utility;
  const minU = u.min_score ?? 0.1;
  const maxU = u.max_score ?? 1.0;
  const utilityScore = result.utility_score != null ? result.utility_score : 0;
  const clamped = Math.max(minU, Math.min(maxU, utilityScore));
  const qaly = result.qaly != null ? result.qaly : 0;
  const remainingYears = result.remaining_years != null ? result.remaining_years : 0;
  const warnings = result.warnings || [];
  const contributions = result.contributions || {};
  const scenarioQaly = result.scenario_qaly_scores || {};

  let html = '';
  if (result.error) {
    html += `<p class="error">${escapeHtml(result.error)}</p>`;
  }
  html += `
    <p><strong>Utility score:</strong> <span class="utility-score">${clamped.toFixed(4)}</span></p>
    <p><strong>Remaining years to age ${u.benchmark_age ?? 80}:</strong> ${remainingYears}</p>
    <p><strong>Estimated quality-adjusted years:</strong> <span class="qaly-value">${qaly}</span></p>
    <p class="result-explainer">Utility score: 0 = worst health, 1 = full health. The value above is <strong>estimated</strong> for comparison and exploration only—not for clinical or reimbursement use. Remaining years is counted to the benchmark age above, not life expectancy.</p>
  `;
  if (warnings.length > 0) {
    html += `<div class="warnings"><strong>Notes</strong><ul>${warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join('')}</ul></div>`;
  }
  if (Object.keys(contributions).length > 0) {
    const n = result.contribution_labels?.length ?? Object.keys(contributions).length;
    const chartHeight = Math.max(200, n * 28);
    html += `<div class="chart-container chart-container-contributions" style="height:${chartHeight}px"><h3>Utility score contribution breakdown</h3><p class="chart-legend">Green = raises your score, red = lowers it.</p><canvas id="chart-contributions" aria-label="Contribution breakdown"></canvas></div>`;
  }
  if (Object.keys(scenarioQaly).length > 0) {
    html += `<div class="chart-container"><h3>Scenario comparison (quality-adjusted years)</h3><p class="chart-legend">Best case: favorable values from population. Worst case: unfavorable values. Middle cases: mixed.</p><canvas id="chart-scenarios" aria-label="Scenario comparison"></canvas></div>`;
  }
  if (result.cumulative_utility_values?.length > 0) {
    html += `<div class="chart-container"><h3>Cumulative utility score progression</h3><canvas id="chart-cumulative" aria-label="Cumulative utility progression"></canvas></div>`;
  }
  if (result.tornado_values?.length > 0) {
    const nT = result.tornado_labels?.length ?? result.tornado_values.length;
    const hT = Math.max(200, nT * 28);
    html += `<div class="chart-container chart-container-contributions" style="height:${hT}px"><h3>Parameter sensitivity (tornado)</h3><canvas id="chart-tornado" aria-label="Tornado chart"></canvas></div>`;
  }
  if (result.scenario_cumulative_series && Object.keys(result.scenario_cumulative_series).length > 0) {
    html += `<div class="chart-container"><h3>Cumulative utility by scenario</h3><p class="chart-legend">Best case: favorable values from population. Worst case: unfavorable values. Middle cases: mixed.</p><canvas id="chart-scenario-cumulative" aria-label="Scenario cumulative progression"></canvas></div>`;
  }
  if (result.scenario_tornado_series?.length > 0) {
    const nSt = result.scenario_tornado_parameters?.length ?? 0;
    const hSt = Math.max(200, nSt * 28);
    html += `<div class="chart-container chart-container-contributions" style="height:${hSt}px"><h3>Scenario comparison (contributions)</h3><p class="chart-legend">Best case: favorable values from population. Worst case: unfavorable values. Middle cases: mixed.</p><canvas id="chart-scenario-tornado" aria-label="Scenario tornado"></canvas></div>`;
  }

  resultsContent.innerHTML = html;
  resultsEl.classList.remove('hidden');
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Draw charts after DOM is updated
  if (contributionChart) {
    contributionChart.destroy();
    contributionChart = null;
  }
  if (scenarioChart) {
    scenarioChart.destroy();
    scenarioChart = null;
  }
  if (cumulativeChart) {
    cumulativeChart.destroy();
    cumulativeChart = null;
  }
  if (tornadoChart) {
    tornadoChart.destroy();
    tornadoChart = null;
  }
  if (scenarioCumulativeChart) {
    scenarioCumulativeChart.destroy();
    scenarioCumulativeChart = null;
  }
  if (scenarioTornadoChart) {
    scenarioTornadoChart.destroy();
    scenarioTornadoChart = null;
  }
  if (typeof Chart !== 'undefined' && Object.keys(contributions).length > 0) {
    // Use ordered arrays from API when present so labels and bars always align
    const labels = Array.isArray(result.contribution_labels) && Array.isArray(result.contribution_values)
      ? result.contribution_labels
      : Object.keys(contributions);
    const values = Array.isArray(result.contribution_values) && result.contribution_values.length === labels.length
      ? result.contribution_values
      : labels.map((l) => result.contributions[l]);
    const colors = values.map((v) => (v >= 0 ? 'rgba(25, 135, 84, 0.8)' : 'rgba(220, 53, 69, 0.8)'));
    const ctx = document.getElementById('chart-contributions');
    if (ctx) {
      contributionChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{ label: 'Contribution', data: values, backgroundColor: colors }],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            title: { display: false },
          },
          scales: {
            x: {
              title: { display: true, text: 'Contribution value' },
              ticks: { precision: 3 },
            },
            y: {
              ticks: {
                autoSkip: false,
                maxTicksLimit: labels.length,
              },
            },
          },
        },
      });
    }
  }
  if (typeof Chart !== 'undefined' && Object.keys(scenarioQaly).length > 0) {
    const labels = Object.keys(scenarioQaly).map((k) => k.replace(/_/g, ' '));
    const values = Object.values(scenarioQaly);
    const colors = ['#198754', '#dc3545', '#fd7e14', '#6f42c1'];
    const ctx = document.getElementById('chart-scenarios');
    if (ctx) {
      scenarioChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{ label: 'Quality-adjusted years', data: values, backgroundColor: colors }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: { display: false },
            title: { display: false },
          },
          scales: {
            y: {
              title: { display: true, text: 'Quality-adjusted years' },
              beginAtZero: true,
            },
          },
        },
      });
    }
  }
  // Cumulative utility progression (line)
  if (typeof Chart !== 'undefined' && result.cumulative_utility_values?.length > 0) {
    const cumValues = result.cumulative_utility_values;
    const labels = result.cumulative_utility_labels || cumValues.map((_, i) => (i === 0 ? 'Initial' : `Step ${i}`));
    const cumMin = Math.min(...cumValues);
    const cumMax = Math.max(...cumValues);
    const cumRange = cumMax - cumMin || 0.1;
    const cumPad = cumRange * 0.1;
    const ctx = document.getElementById('chart-cumulative');
    if (ctx) {
      cumulativeChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
          labels,
          datasets: [{ label: 'Utility score', data: result.cumulative_utility_values, borderColor: '#0d6efd', backgroundColor: 'rgba(13, 110, 253, 0.1)', fill: true, tension: 0.2 }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              ticks: {
                autoSkip: false,
                maxRotation: 90,
                minRotation: 45,
              },
            },
            y: {
              title: { display: true, text: 'Utility score' },
              min: cumMin - cumPad,
              max: cumMax + cumPad,
            },
          },
        },
      });
    }
  }
  // Tornado (parameter sensitivity) – horizontal bar
  if (typeof Chart !== 'undefined' && result.tornado_values?.length > 0) {
    const labels = result.tornado_labels || [];
    const values = result.tornado_values;
    const colors = values.map((v) => (v >= 0 ? 'rgba(25, 135, 84, 0.8)' : 'rgba(220, 53, 69, 0.8)'));
    const ctx = document.getElementById('chart-tornado');
    if (ctx) {
      tornadoChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{ label: 'Contribution', data: values, backgroundColor: colors }],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { title: { display: true, text: 'Contribution value' }, ticks: { precision: 3 } },
            y: { ticks: { autoSkip: false, maxTicksLimit: labels.length } },
          },
        },
      });
    }
  }
  // Scenario cumulative progression (multiple lines)
  if (typeof Chart !== 'undefined' && result.scenario_cumulative_series && Object.keys(result.scenario_cumulative_series).length > 0) {
    const xLabels = result.scenario_cumulative_labels || [];
    const series = result.scenario_cumulative_series;
    const allScenarioValues = Object.values(series).flat();
    const scMin = Math.min(...allScenarioValues);
    const scMax = Math.max(...allScenarioValues);
    const scRange = scMax - scMin || 0.1;
    const scPad = scRange * 0.1;
    const scenarioColors = { best_case: '#198754', worst_case: '#dc3545', middle_case_1: '#fd7e14', middle_case_2: '#6f42c1' };
    const datasets = Object.entries(series).map(([key, values]) => ({
      label: key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
      data: values,
      borderColor: scenarioColors[key] || '#6c757d',
      backgroundColor: 'transparent',
      tension: 0.2,
    }));
    const ctx = document.getElementById('chart-scenario-cumulative');
    if (ctx) {
      scenarioCumulativeChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: { labels: xLabels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: { legend: { display: true } },
          scales: {
            x: {
              ticks: {
                autoSkip: false,
                maxRotation: 90,
                minRotation: 45,
              },
            },
            y: {
              title: { display: true, text: 'Utility score' },
              min: scMin - scPad,
              max: scMax + scPad,
            },
          },
        },
      });
    }
  }
  // Scenario tornado (grouped horizontal bars)
  if (typeof Chart !== 'undefined' && result.scenario_tornado_series?.length > 0) {
    const params = result.scenario_tornado_parameters || [];
    const series = result.scenario_tornado_series;
    const colors = ['#198754', '#dc3545', '#fd7e14', '#6f42c1'];
    const datasets = series.map((s, i) => ({
      label: s.scenario,
      data: s.values,
      backgroundColor: colors[i % colors.length],
    }));
    const ctx = document.getElementById('chart-scenario-tornado');
    if (ctx) {
      scenarioTornadoChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: { labels: params, datasets },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: true } },
          scales: {
            x: { title: { display: true, text: 'Contribution value' }, ticks: { precision: 3 } },
            y: { ticks: { autoSkip: false, maxTicksLimit: params.length } },
          },
        },
      });
    }
  }
}

async function handleSubmit(e) {
  e.preventDefault();
  const formData = getFormData();

  submitBtn.disabled = true;
  resultsContent.innerHTML = '<p>Calculating… (this may take a moment)</p>';
  resultsEl.classList.remove('hidden');

  const apiUrl = 'http://127.0.0.1:5000/api/calculate';
  try {
    const res = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });
    const result = await res.json();
    if (!res.ok) {
      renderResults({ ...result, error: result.error || `HTTP ${res.status}` });
      return;
    }
    renderResults(result);
  } catch (err) {
    resultsContent.innerHTML = `
      <p class="error">Error: ${escapeHtml(err.message)}</p>
      <p><strong>Do this:</strong></p>
      <ol>
        <li>In a terminal, run: <code>python server.py</code> (from the project folder)</li>
        <li>In the browser, open <a href="http://127.0.0.1:5000">http://127.0.0.1:5000</a> (use this link or type it in the address bar)</li>
        <li>Then click Calculate again</li>
      </ol>
      <p>If you opened this page by another URL (e.g. Live Server or file), the backend cannot be reached. Always use <a href="http://127.0.0.1:5000">http://127.0.0.1:5000</a>.</p>
    `;
  } finally {
    submitBtn.disabled = false;
  }
}

function applyConfigToPage() {
  if (!config?.app) return;
  const title = document.getElementById('app-title');
  const subtitle = document.getElementById('app-subtitle');
  if (title && config.app.title) title.textContent = config.app.title;
  if (subtitle && config.app.subtitle) subtitle.textContent = config.app.subtitle;
}

async function init() {
  await loadConfig();
  applyConfigToPage();
  form.addEventListener('submit', handleSubmit);
}

init();
