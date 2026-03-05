/**
 * SmartContainer Risk Engine — Frontend JavaScript
 */

/* ── State ─────────────────────────────────────────────────────────────────── */
let allPredictions  = [];
let filteredPredictions = [];
let currentPage     = 1;
const PAGE_SIZE     = 50;
let sortCol         = 'Risk_Score';
let sortDir         = 'desc';
let pollInterval    = null;

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const runBtn        = document.getElementById('run-btn');
const progressWrap  = document.getElementById('progress-wrap');
const progressFill  = document.getElementById('progress-fill');
const progressLabel = document.getElementById('progress-label');
const statTotal     = document.getElementById('stat-total');
const statCritical  = document.getElementById('stat-critical');
const statMedium    = document.getElementById('stat-medium');
const statLow       = document.getElementById('stat-low');
const tableBody     = document.getElementById('table-body');
const searchInput   = document.getElementById('search-input');
const filterSelect  = document.getElementById('filter-select');
const dashImg       = document.getElementById('dashboard-img');
const dashSection   = document.getElementById('dashboard-section');
const paginationInfo = document.getElementById('pagination-info');
const prevBtn       = document.getElementById('prev-btn');
const nextBtn       = document.getElementById('next-btn');
const modal         = document.getElementById('modal-overlay');
const modalBody     = document.getElementById('modal-body');
const toastContainer = document.getElementById('toast-container');

/* ── Initialise ────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  checkStatus();
  loadStats();
  loadPredictions();
  setInterval(checkStatus, 10000);
});

/* ── Status & Pipeline ─────────────────────────────────────────────────────── */
async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    updateStatusUI(data);
  } catch (e) { /* silent */ }
}

function updateStatusUI(data) {
  if (data.status === 'running') {
    setRunBtnState('running');
    showProgress(data.progress || 0, data.message || 'Processing…');
  } else if (data.status === 'completed') {
    setRunBtnState('idle');
    hideProgress();
    if (data.dashboard_exists) showDashboard();
  } else if (data.status === 'error') {
    setRunBtnState('idle');
    hideProgress();
    showToast('Pipeline error: ' + (data.message || 'Unknown error'), 'error');
  } else {
    setRunBtnState('idle');
  }
}

async function runPipeline() {
  if (runBtn.disabled) return;
  setRunBtnState('running');
  showProgress(0, 'Starting pipeline…');

  try {
    const res = await fetch('/api/run-pipeline', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to start pipeline');
    }
    showToast('Pipeline started! This may take a few minutes.', 'info');
    startPolling();
  } catch (e) {
    setRunBtnState('idle');
    hideProgress();
    showToast(e.message, 'error');
  }
}

function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      updateStatusUI(data);

      if (data.status === 'completed') {
        clearInterval(pollInterval);
        pollInterval = null;
        showToast('✅ Pipeline completed successfully!', 'success');
        await loadStats();
        await loadPredictions();
        showDashboard();
      } else if (data.status === 'error') {
        clearInterval(pollInterval);
        pollInterval = null;
        showToast('❌ Pipeline error: ' + (data.message || ''), 'error');
      }
    } catch (e) { /* silent */ }
  }, 2000);
}

function setRunBtnState(state) {
  if (state === 'running') {
    runBtn.disabled = true;
    runBtn.innerHTML = '<span class="spinner"></span> Running…';
  } else {
    runBtn.disabled = false;
    runBtn.innerHTML = '▶ Run Pipeline';
  }
}

function showProgress(pct, label) {
  progressWrap.style.display = 'block';
  progressFill.style.width   = pct + '%';
  progressLabel.textContent  = label + ' (' + pct + '%)';
}
function hideProgress() { progressWrap.style.display = 'none'; }

/* ── Stats ─────────────────────────────────────────────────────────────────── */
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const d   = await res.json();
    if (!d.total) return;
    animateCounter(statTotal,    d.total);
    animateCounter(statCritical, d.critical);
    animateCounter(statMedium,   d.medium_risk);
    animateCounter(statLow,      d.low_risk);
  } catch (e) { /* silent */ }
}

function animateCounter(el, target) {
  const start = parseInt(el.textContent.replace(/,/g, '')) || 0;
  const dur   = 800;
  const step  = 16;
  const steps = dur / step;
  let current = start;
  const inc   = (target - start) / steps;
  const timer = setInterval(() => {
    current += inc;
    if ((inc >= 0 && current >= target) || (inc < 0 && current <= target)) {
      current = target;
      clearInterval(timer);
    }
    el.textContent = Math.round(current).toLocaleString();
  }, step);
}

/* ── Predictions Table ─────────────────────────────────────────────────────── */
async function loadPredictions() {
  try {
    const res  = await fetch('/api/predictions?limit=10000');
    const data = await res.json();
    if (!data.data || data.data.length === 0) return;
    allPredictions = data.data;
    applyFilters();
  } catch (e) { /* silent */ }
}

function applyFilters() {
  const query  = (searchInput.value || '').toLowerCase().trim();
  const level  = filterSelect.value;

  filteredPredictions = allPredictions.filter(p => {
    const matchLevel  = !level || p.Risk_Level === level;
    const matchSearch = !query  ||
      p.Container_ID.toLowerCase().includes(query) ||
      (p.Origin_Country || '').toLowerCase().includes(query) ||
      (p.Destination_Port || '').toLowerCase().includes(query) ||
      (p.Explanation_Summary || '').toLowerCase().includes(query);
    return matchLevel && matchSearch;
  });

  sortData();
  currentPage = 1;
  renderTable();
}

function sortData() {
  filteredPredictions.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va < vb) return sortDir === 'asc' ? -1 : 1;
    if (va > vb) return sortDir === 'asc' ? 1 : -1;
    return 0;
  });
}

function sortByColumn(col) {
  if (sortCol === col) {
    sortDir = sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    sortCol = col;
    sortDir = col === 'Risk_Score' ? 'desc' : 'asc';
  }
  document.querySelectorAll('th').forEach(th => {
    th.classList.remove('sorted');
    const icon = th.querySelector('.sort-icon');
    if (icon) icon.textContent = '↕';
  });
  const activeTh = document.querySelector(`th[data-col="${col}"]`);
  if (activeTh) {
    activeTh.classList.add('sorted');
    const icon = activeTh.querySelector('.sort-icon');
    if (icon) icon.textContent = sortDir === 'asc' ? '↑' : '↓';
  }
  sortData();
  renderTable();
}

function renderTable() {
  const total  = filteredPredictions.length;
  const start  = (currentPage - 1) * PAGE_SIZE;
  const end    = Math.min(start + PAGE_SIZE, total);
  const page   = filteredPredictions.slice(start, end);

  paginationInfo.textContent =
    total === 0 ? 'No results' :
    `Showing ${(start + 1).toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()}`;

  prevBtn.disabled = currentPage === 1;
  nextBtn.disabled = end >= total;

  if (page.length === 0) {
    tableBody.innerHTML = `
      <tr><td colspan="6">
        <div class="empty-state">
          <div class="icon">📭</div>
          <p>No predictions found. Run the pipeline first.</p>
        </div>
      </td></tr>`;
    return;
  }

  tableBody.innerHTML = page.map(p => {
    const level   = p.Risk_Level;
    const cls     = level === 'Critical' ? 'critical' : level === 'Medium Risk' ? 'medium' : '';
    const badge   = level === 'Critical' ? 'badge-critical' :
                    level === 'Medium Risk' ? 'badge-medium' : 'badge-low';
    const scoreColor = level === 'Critical' ? '#e74c3c' :
                       level === 'Medium Risk' ? '#f39c12' : '#2ecc71';
    const score   = p.Risk_Score.toFixed(1);

    return `<tr class="${cls}" onclick="showContainerDetail('${escHtml(p.Container_ID)}')">
      <td>${escHtml(p.Container_ID)}</td>
      <td>
        <div class="score-bar-wrap">
          <span style="width:42px;text-align:right;font-weight:600;">${score}</span>
          <div class="score-bar"><div class="score-bar-fill" style="width:${p.Risk_Score}%;background:${scoreColor}"></div></div>
        </div>
      </td>
      <td><span class="badge ${badge}">${escHtml(level)}</span></td>
      <td>${escHtml(p.Origin_Country || '—')}</td>
      <td>${escHtml(p.Destination_Port || '—')}</td>
      <td style="max-width:280px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${escHtml(p.Explanation_Summary || '')}">${escHtml(p.Explanation_Summary || '—')}</td>
    </tr>`;
  }).join('');
}

function prevPage() { if (currentPage > 1) { currentPage--; renderTable(); } }
function nextPage() {
  if (currentPage * PAGE_SIZE < filteredPredictions.length) { currentPage++; renderTable(); }
}

/* ── Container Detail Modal ───────────────────────────────────────────────── */
async function showContainerDetail(id) {
  try {
    const res  = await fetch('/api/predictions/' + encodeURIComponent(id));
    if (!res.ok) throw new Error('Not found');
    const p    = await res.json();
    const level = p.Risk_Level || '';
    const badge = level === 'Critical' ? 'badge-critical' :
                  level === 'Medium Risk' ? 'badge-medium' : 'badge-low';

    modalBody.innerHTML = `
      <div class="modal-header">
        <div>
          <h2 style="font-size:1.3rem;margin-bottom:6px">Container ${escHtml(String(p.Container_ID))}</h2>
          <span class="badge ${badge}">${escHtml(level)}</span>
        </div>
        <button class="modal-close" onclick="closeModal()">✕</button>
      </div>
      ${detailRow('Risk Score', `<strong style="font-size:1.6rem">${Number(p.Risk_Score).toFixed(1)}</strong> / 100`)}
      ${detailRow('Risk Level', `<span class="badge ${badge}">${escHtml(level)}</span>`)}
      ${detailRow('Origin Country', escHtml(p.Origin_Country || '—'))}
      ${detailRow('Destination Port', escHtml(p.Destination_Port || '—'))}
      ${detailRow('Explanation', escHtml(p.Explanation_Summary || '—'))}
    `;
    modal.classList.remove('hidden');
  } catch (e) {
    showToast('Could not load container details.', 'error');
  }
}

function detailRow(label, value) {
  return `<div class="detail-row"><div class="detail-label">${label}</div><div class="detail-value">${value}</div></div>`;
}

function closeModal() { modal.classList.add('hidden'); }
modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });

/* ── Dashboard Image ──────────────────────────────────────────────────────── */
function showDashboard() {
  if (!dashSection) return;
  dashSection.style.display = 'block';
  dashImg.src = '/api/dashboard-image?t=' + Date.now();
}

/* ── Download ──────────────────────────────────────────────────────────────── */
function downloadFile(type) {
  const urls = {
    predictions: '/api/download/predictions',
    dashboard:   '/api/download/dashboard',
    report:      '/api/report',
  };
  const url = urls[type];
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = type === 'predictions' ? 'predictions.csv' : type === 'dashboard' ? 'summary_report.png' : 'report.html';
  a.click();
}

/* ── Toast ─────────────────────────────────────────────────────────────────── */
function showToast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${escHtml(msg)}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

/* ── Utility ───────────────────────────────────────────────────────────────── */
function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
