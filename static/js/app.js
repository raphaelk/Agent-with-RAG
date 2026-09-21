/**
 * Agent with RAG Frontend Controller
 * Manages 4 views, API integrations, polling, charts, and modal dialogs.
 */

// Application State
let currentTab = 'chat';
let activeConversationId = null;
let currentEmbeddingModel = '';
let targetEmbeddingModelToSwitch = '';
let requestsChart = null;
let tokensChart = null;
let modelMaxTokensMap = {};
let allEventsCache = [];

// DOM Ready Initialization
document.addEventListener('DOMContentLoaded', () => {
  initHealthPolling();
  loadModelsList();
  loadSkillsDropdown();
  loadIngestData();
  fetchTelemetryData();
  fetchLogsData();
});

// ----------------- Tab Navigation -----------------
function switchTab(tabName) {
  currentTab = tabName;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-page').forEach(page => page.classList.remove('active'));

  const activeBtn = document.getElementById(`tab-btn-${tabName}`);
  const activePage = document.getElementById(`page-${tabName}`);
  if (activeBtn) activeBtn.classList.add('active');
  if (activePage) activePage.classList.add('active');

  if (tabName === 'ingest') {
    loadIngestData();
  } else if (tabName === 'telemetry') {
    fetchTelemetryData();
  } else if (tabName === 'audit') {
    fetchLogsData();
  }
}

// ----------------- Health & Status Polling -----------------
function initHealthPolling() {
  checkHealth();
  setInterval(checkHealth, 5000);
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const dot = document.getElementById('status-indicator');
    const text = document.getElementById('status-agent-text');

    if (data.status === 'healthy') {
      dot.className = 'status-dot';
      text.textContent = 'Online';
      text.style.color = 'var(--accent-emerald)';
    } else {
      dot.className = 'status-dot degraded';
      text.textContent = 'Degraded';
      text.style.color = 'var(--accent-amber)';
    }
  } catch (err) {
    const dot = document.getElementById('status-indicator');
    const text = document.getElementById('status-agent-text');
    dot.className = 'status-dot offline';
    text.textContent = 'Offline';
    text.style.color = 'var(--accent-rose)';
  }
}

// ----------------- Shutdown Modal Flow -----------------
function openShutdownModal() {
  document.getElementById('shutdown-confirm-input').value = '';
  document.getElementById('btn-confirm-shutdown').disabled = true;
  openModal('modal-shutdown');
}

function validateShutdownConfirm(val) {
  const btn = document.getElementById('btn-confirm-shutdown');
  btn.disabled = (val.trim() !== 'Shutdown the service');
}

async function executeShutdown() {
  const confirmInput = document.getElementById('shutdown-confirm-input').value;
  try {
    const res = await fetch('/api/shutdown', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirm_text: confirmInput })
    });
    const data = await res.json();
    closeModal('modal-shutdown');
    alert(data.message || 'Shutdown command executed.');
    document.getElementById('status-agent-text').textContent = 'Terminated';
    document.getElementById('status-indicator').className = 'status-dot offline';
  } catch (e) {
    alert('Shutdown executed. Server process is terminating.');
  }
}

// ----------------- PAGE 1: Chat & Synthesis -----------------
async function loadModelsList() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const select = document.getElementById('chat-model-select');
    select.innerHTML = '';

    modelMaxTokensMap = {};
    (data.models || []).forEach(m => {
      modelMaxTokensMap[m.id] = m.max_tokens || 4096;
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.name;
      select.appendChild(opt);
    });

    if (data.last_custom_endpoint) {
      document.getElementById('chat-custom-endpoint').value = data.last_custom_endpoint;
    }

    const defaultModel = data.default_model || 'gemma-4-26b-a4b-it';
    if (defaultModel && Array.from(select.options).some(o => o.value === defaultModel)) {
      select.value = defaultModel;
    }
    onModelChange();
  } catch (e) {
    console.error('Failed to load model list:', e);
  }
}

function clampMaxTokens(el) {
  const select = document.getElementById('chat-model-select');
  const selectedModel = select ? select.value : 'gemma-4-26b-a4b-it';
  const modelMaxLimit = modelMaxTokensMap[selectedModel] || 8192;
  el.max = modelMaxLimit;
  const val = parseInt(el.value, 10);
  if (val > modelMaxLimit) {
    el.value = modelMaxLimit;
  }
}

function onModelChange() {
  const select = document.getElementById('chat-model-select');
  const selectedModel = select.value;
  const customBox = document.getElementById('custom-endpoint-box');
  const maxTokensInput = document.getElementById('chat-max-tokens');

  if (selectedModel === 'custom') {
    customBox.style.display = 'flex';
  } else {
    customBox.style.display = 'none';
  }

  const modelMaxLimit = modelMaxTokensMap[selectedModel] || 8192;
  maxTokensInput.max = modelMaxLimit;
  if (parseInt(maxTokensInput.value, 10) > modelMaxLimit) {
    maxTokensInput.value = modelMaxLimit;
  }
}

function clampMaxTurns(input) {
  let val = parseInt(input.value, 10);
  if (isNaN(val) || val < 1) val = 1;
  if (val > 10) val = 10;
  input.value = val;
}

function onSkillsModeChange() {
  const select = document.getElementById('chat-skills-select');
  const thresholdBox = document.getElementById('skill-threshold-box');
  if (!select || !thresholdBox) return;
  if (select.value === 'vector_store') {
    thresholdBox.style.display = 'flex';
  } else {
    thresholdBox.style.display = 'none';
  }
}

async function loadSkillsDropdown() {
  const select = document.getElementById('chat-skills-select');
  if (!select) return;
  try {
    const res = await fetch('/api/skills/list');
    const json = await res.json();
    if (json.status === 'success' && json.skills) {
      select.innerHTML = `
        <option value="vector_store" selected>Vector Store</option>
        <option value="llm_selected">LLM Selected</option>
      `;
      json.skills.forEach(skill => {
        const opt = document.createElement('option');
        opt.value = skill.folder_name;
        opt.textContent = skill.name;
        select.appendChild(opt);
      });
    }
  } catch (err) {
    console.error('Failed to load skills dropdown:', err);
  }
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const query = input.value.trim();
  if (!query) return;

  const model = document.getElementById('chat-model-select').value;
  const temperature = parseFloat(document.getElementById('chat-temperature').value) || 0.7;
  const modelMaxLimit = modelMaxTokensMap[model] || 8192;
  let maxTokens = parseInt(document.getElementById('chat-max-tokens').value, 10) || 2048;
  if (maxTokens > modelMaxLimit) {
    maxTokens = modelMaxLimit;
    document.getElementById('chat-max-tokens').value = modelMaxLimit;
  }
  const maxRagChunks = parseInt(document.getElementById('rag-max-chunks').value, 10) || 5;
  const customEndpoint = document.getElementById('chat-custom-endpoint').value;

  const agentSelect = document.getElementById('chat-agent-select');
  const agentType = agentSelect ? agentSelect.value : 'custom';

  const maxTurnsInput = document.getElementById('chat-max-turns');
  const maxTurns = maxTurnsInput ? Math.min(Math.max(1, parseInt(maxTurnsInput.value, 10) || 3), 10) : 3;
  const skillsSelect = document.getElementById('chat-skills-select');
  const skillsMode = skillsSelect ? skillsSelect.value : 'vector_store';
  const skillThresholdInput = document.getElementById('chat-skill-threshold');
  const skillThreshold = skillThresholdInput ? (parseFloat(skillThresholdInput.value) || 0.2) : 0.2;
  const docThresholdInput = document.getElementById('doc-threshold');
  const docThreshold = docThresholdInput ? (parseFloat(docThresholdInput.value) || 0.3) : 0.3;

  // Use a new conversation ID for each question per specification
  const questionConvId = `conv-${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 7)}`;

  // Append user bubble
  appendChatMessage('user', query);
  input.value = '';

  // Show temporary loading indicator
  const loadingBubble = appendChatMessage('agent', 'Thinking and retrieving evidence...');
  const sendBtn = document.getElementById('btn-send-chat');
  sendBtn.disabled = true;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        agent_type: agentType,
        model: model,
        temperature: temperature,
        max_tokens: maxTokens,
        max_rag_chunks: maxRagChunks,
        custom_endpoint: customEndpoint,
        conversation_id: questionConvId,
        skills_mode: skillsMode,
        skill_threshold: skillThreshold,
        doc_threshold: docThreshold,
        max_turns: maxTurns
      })
    });

    const json = await res.json();
    loadingBubble.remove();

    if (json.status === 'success') {
      const respData = json.data;
      const tokensMeta = respData.tokens ? ` | Tokens: ${respData.tokens.input} in, ${respData.tokens.output} out` : '';

      appendChatMessage(
        'agent',
        respData.response || respData.answer || '',
        `Conv ID: ${respData.conversation_id} | Agent: ${respData.agent_type === 'google_adk' ? 'Google ADK LlmAgent' : 'Custom Agent'} | Model: ${respData.model_used || model}${tokensMeta} | Latency: ${respData.latency_ms}ms`,
        respData.steps || []
      );

      // Render retrieved context evidence
      renderRetrievedEvidence(respData.evidence || respData.retrieved_evidence || []);
    } else {
      appendChatMessage('agent', `Error: ${json.message}`);
    }
  } catch (err) {
    loadingBubble.remove();
    appendChatMessage('agent', `Network error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
  }
}

function appendChatMessage(sender, text, metaText = '', steps = []) {
  const history = document.getElementById('chat-history');
  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-message ${sender}`;

  const textDiv = document.createElement('div');
  textDiv.className = 'message-content';
  textDiv.textContent = text;
  msgDiv.appendChild(textDiv);

  // Add response logs box with Show Logs button, component bubbles, and collapsible step logs
  if (steps && steps.length > 0) {
    const box = document.createElement('div');
    box.className = 'response-logs-box';

    const header = document.createElement('div');
    header.className = 'logs-box-header';

    const bubblesRow = document.createElement('div');
    bubblesRow.className = 'component-bubbles-row';

    steps.forEach((step) => {
      const bubble = document.createElement('div');
      bubble.className = 'component-bubble';
      bubble.innerHTML = `
        <span class="bubble-icon">${escapeHtml(step.icon || '⚙️')}</span>
        <span class="bubble-name">${escapeHtml(step.component || step.step_name)}</span>
        <span class="bubble-time">⏱️ ${step.elapsed_ms}ms</span>
      `;
      bubblesRow.appendChild(bubble);
    });

    const toggleBtn = document.createElement('button');
    toggleBtn.type = 'button';
    toggleBtn.className = 'btn-show-logs';
    toggleBtn.textContent = 'Show Logs';
    toggleBtn.onclick = function() { toggleShowLogs(this); };

    header.appendChild(bubblesRow);
    header.appendChild(toggleBtn);
    box.appendChild(header);

    // Collapsible container to expand and show full content of the step including logs
    const collapsible = document.createElement('div');
    collapsible.className = 'logs-collapsible-content';
    collapsible.style.display = 'none';

    steps.forEach((step) => {
      const stepCard = document.createElement('div');
      stepCard.className = 'step-detail-bubble';

      let logsHtml = '';
      if (step.logs && step.logs.length > 0) {
        logsHtml = step.logs.map(log => `
          <div class="step-log-item">
            <div class="log-meta">
              <span class="evidence-badge" style="font-size: 0.7rem; padding: 1px 5px;">[${escapeHtml(log.call_type || 'event')}]</span>
              <strong>${escapeHtml(log.event_type || '')}</strong>:
              <span>${escapeHtml(log.invoker || '')} ➔ ${escapeHtml(log.recipient || log.target || '')}</span>
              <span style="color: var(--text-muted); font-size: 0.72rem;">(${escapeHtml(log.timestamp ? log.timestamp.split('T')[1].slice(0, 8) : '')})</span>
            </div>
            <div class="log-desc">${escapeHtml(log.description || '')}</div>
            <pre class="log-payload-json">${escapeHtml(JSON.stringify(log.payload, null, 2))}</pre>
          </div>
        `).join('');
      } else {
        logsHtml = `<div style="color: var(--text-muted); font-size: 0.8rem; padding: 4px;">No detailed log records captured for this step.</div>`;
      }

      stepCard.innerHTML = `
        <div class="step-detail-header">
          <div class="step-detail-title">
            <span class="step-detail-icon">${escapeHtml(step.icon || '⚙️')}</span>
            <strong>${escapeHtml(step.component || step.step_name)}</strong>
          </div>
          <span class="step-detail-time">⏱️ ${step.elapsed_ms}ms</span>
        </div>
        <div class="step-bubble-scroll-area">
          <div class="step-summary"><strong>Summary:</strong> ${escapeHtml(step.summary || '')}</div>
          <div class="step-logs-wrapper">
            <div class="step-logs-header">📋 Step Logs &amp; Payloads:</div>
            ${logsHtml}
          </div>
        </div>
      `;
      collapsible.appendChild(stepCard);
    });

    box.appendChild(collapsible);
    msgDiv.appendChild(box);
  }

  if (metaText) {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'meta';
    metaDiv.textContent = metaText;
    msgDiv.appendChild(metaDiv);
  }

  history.appendChild(msgDiv);
  history.scrollTop = history.scrollHeight;
  return msgDiv;
}

function toggleShowLogs(btn) {
  const box = btn.closest('.response-logs-box');
  if (!box) return;
  const content = box.querySelector('.logs-collapsible-content');
  if (!content) return;
  const isHidden = content.style.display === 'none';
  content.style.display = isHidden ? 'flex' : 'none';
  btn.textContent = isHidden ? 'Hide Logs' : 'Show Logs';
  btn.classList.toggle('active', isHidden);
}

function renderRetrievedEvidence(evidenceList) {
  const container = document.getElementById('evidence-container');
  container.innerHTML = '';

  if (!evidenceList || evidenceList.length === 0) {
    container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.88rem; padding: 8px;">No relevant context chunks or skills matched above threshold.</div>';
    return;
  }

  // Group all results by the document per SPECIFICATION.md:
  // "Display the contents of the information retrieved from the vector store.
  //  - Include results from the skills vector store and the documents vector store.
  //  - Group the results by the documents"
  const groups = {};
  evidenceList.forEach(item => {
    let docName = item.document_name || item.details?.document_name;
    if (!docName) {
      if (item.title && item.title.startsWith('Doc: ')) {
        docName = item.title.replace('Doc: ', '').split(' (')[0];
      } else if (item.title && item.title.startsWith('Skill: ')) {
        const sName = item.title.replace('Skill: ', '').split(' (')[0];
        docName = `${item.details?.folder_name || sName}/SKILL.md`;
      } else if (item.details?.folder_name) {
        docName = `${item.details.folder_name}/SKILL.md`;
      } else if (item.details?.name) {
        docName = `${item.details.name}/SKILL.md`;
      } else {
        docName = item.details?.source || 'Context Document';
      }
    }

    if (!groups[docName]) groups[docName] = [];
    groups[docName].push(item);
  });

  for (const [docName, items] of Object.entries(groups)) {
    const groupHeader = document.createElement('div');
    groupHeader.className = 'evidence-group-title';
    groupHeader.textContent = `▶ Document: ${docName} (${items.length} ${items.length === 1 ? 'item' : 'items'})`;
    container.appendChild(groupHeader);

    items.forEach(ev => {
      const card = document.createElement('div');
      card.className = 'evidence-card';

      const isSkillStore = ev.source_type === 'skill_vector_store' || ev.store === 'skills' || (ev.step && ev.step.toLowerCase().includes('skill'));
      const storeLabel = isSkillStore ? 'Skills Vector Store' : 'Documents Vector Store';
      const storeBadgeStyle = isSkillStore
        ? 'background: rgba(139, 92, 246, 0.2); color: #c4b5fd;'
        : 'background: rgba(16, 185, 129, 0.15); color: #34d399;';

      const scoreText = (ev.score !== undefined && ev.score !== null)
        ? `Score: ${typeof ev.score === 'number' ? ev.score.toFixed(3) : ev.score}`
        : '';

      card.innerHTML = `
        <div class="evidence-header">
          <strong>${escapeHtml(ev.title || 'Evidence Chunk')}</strong>
          <div style="display: flex; gap: 6px; align-items: center;">
            <span class="evidence-badge" style="${storeBadgeStyle} font-size: 0.72rem;">${storeLabel}</span>
            ${scoreText ? `<span class="evidence-badge">${scoreText}</span>` : ''}
          </div>
        </div>
        <div class="evidence-body">${escapeHtml(ev.content || '')}</div>
      `;
      container.appendChild(card);
    });
  }
}

function clearEvidenceView() {
  document.getElementById('evidence-container').innerHTML =
    '<div style="color: var(--text-muted); font-size: 0.9rem; margin-top: 12px;">Evidence view cleared. Subsequent chat inquiries will populate here.</div>';
}

// ----------------- PAGE 2: Vector DB Ingestion -----------------
async function loadIngestData() {
  loadVectorStorageStatus();
  loadOllamaEmbeddingCatalog();
}

async function loadVectorStorageStatus() {
  try {
    const res = await fetch('/api/vector/status');
    const data = await res.json();

    document.getElementById('stat-chunks-count').textContent = data.total_chunks || 0;
    document.getElementById('stat-docs-count').textContent = data.total_documents || 0;
    document.getElementById('stat-db-size').textContent = `${data.db_size_mb || 0} MB`;

    const spinner = document.getElementById('ingest-progress-spinner');
    spinner.style.display = data.is_ingesting ? 'inline' : 'none';

    // Populate Ingested Docs Table
    const tbody = document.getElementById('ingested-docs-tbody');
    tbody.innerHTML = '';
    const docs = data.documents || [];

    if (docs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No documents ingested yet.</td></tr>';
    } else {
      docs.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${escapeHtml(d.name)}</strong></td>
          <td>${d.chunks_count}</td>
          <td>${d.total_characters.toLocaleString()}</td>
          <td style="text-align: right;">
            <button class="btn-danger btn-sm" style="padding: 4px 10px; font-size: 0.8rem;" onclick="deleteDocument('${escapeHtml(d.name)}')">Delete</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (e) {
    console.error('Error fetching vector status:', e);
  }
}

async function loadOllamaEmbeddingCatalog() {
  try {
    const res = await fetch('/api/ollama/models');
    const data = await res.json();
    currentEmbeddingModel = data.current_model || '';

    const select = document.getElementById('embedder-select');
    select.innerHTML = '';

    const tbody = document.getElementById('embedding-catalog-tbody');
    tbody.innerHTML = '';

    (data.models || []).forEach(m => {
      // Populate Dropdown
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name} ${m.is_active ? '(Active)' : ''}`;
      if (m.is_active) opt.selected = true;
      select.appendChild(opt);

      // Populate Catalog Table
      const tr = document.createElement('tr');
      let badgeClass = 'badge-pull';
      if (m.status === 'Active') badgeClass = 'badge-active';
      else if (m.status === 'Installed') badgeClass = 'badge-installed';

      tr.innerHTML = `
        <td><strong>${escapeHtml(m.name)}</strong></td>
        <td>${m.dimensions}</td>
        <td>${m.context_window}</td>
        <td>${m.size}</td>
        <td style="color: var(--text-secondary);">${escapeHtml(m.description || '')}</td>
        <td><span class="${badgeClass}">${m.status}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error('Error loading embedding catalog:', e);
  }
}

function onEmbedderSelectChange(newModel) {
  if (newModel === currentEmbeddingModel) return;
  targetEmbeddingModelToSwitch = newModel;

  // Open stern warning modal
  document.getElementById('model-switch-confirm-input').value = '';
  document.getElementById('btn-confirm-switch-model').disabled = true;
  openModal('modal-switch-model');
}

function cancelModelSwitch() {
  closeModal('modal-switch-model');
  // Revert select dropdown to active model
  document.getElementById('embedder-select').value = currentEmbeddingModel;
}

function validateModelSwitchConfirm(val) {
  const btn = document.getElementById('btn-confirm-switch-model');
  btn.disabled = (val.trim() !== 'Delete Data and Switch');
}

async function executeModelSwitch() {
  closeModal('modal-switch-model');
  const btn = document.getElementById('btn-confirm-switch-model');
  btn.disabled = true;

  try {
    const res = await fetch('/api/ollama/switch-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: targetEmbeddingModelToSwitch,
        confirm_text: 'Delete Data and Switch'
      })
    });
    const data = await res.json();
    if (data.status === 'success') {
      alert(`Model switched to ${targetEmbeddingModelToSwitch}. Skills database re-indexed.`);
      loadIngestData();
    } else {
      alert(`Switch failed: ${data.message}`);
    }
  } catch (e) {
    alert(`Error: ${e.message}`);
  }
}

function toggleIngestInputMode() {
  const mode = document.getElementById('ingest-source-type').value;
  const input = document.getElementById('ingest-target-input');
  if (mode === 'url') {
    input.placeholder = 'https://en.wikipedia.org/wiki/Artificial_intelligence';
  } else {
    input.placeholder = 'sample_docs or /path/to/documents';
  }
}

function setIngestUrl(url) {
  const input = document.getElementById('ingest-target-input');
  const typeSelect = document.getElementById('ingest-source-type');
  input.value = url;
  if (url.startsWith('http')) {
    typeSelect.value = 'url';
  } else {
    typeSelect.value = 'local';
  }
}

function toggleAdvancedChunking() {
  const card = document.getElementById('advanced-chunking-card');
  const icon = document.getElementById('advanced-toggle-icon');
  if (card.classList.contains('show')) {
    card.classList.remove('show');
    icon.textContent = '▶';
  } else {
    card.classList.add('show');
    icon.textContent = '▼';
  }
}

async function populateVectorDatabase() {
  const target = document.getElementById('ingest-target-input').value.trim();
  const type = document.getElementById('ingest-source-type').value;
  const chunkSize = parseInt(document.getElementById('chunk-size-input').value, 10) || 500;
  const overlap = parseInt(document.getElementById('chunk-overlap-input').value, 10) || 100;

  if (!target) {
    alert('Please specify a URL or local directory/file path to ingest.');
    return;
  }

  const btn = document.getElementById('btn-populate-db');
  btn.disabled = true;
  btn.textContent = 'Ingesting & Vectorizing...';

  try {
    const res = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target: target,
        type: type,
        chunk_size: chunkSize,
        overlap: overlap
      })
    });
    const data = await res.json();
    if (data.status === 'success') {
      alert(`Ingestion Successful! Added ${data.added_chunks} new non-duplicated chunks (${data.total_characters} characters).`);
      loadVectorStorageStatus();
    } else {
      alert(`Ingestion Failed: ${data.message}`);
    }
  } catch (e) {
    alert(`Ingestion error: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Populate Vector Database';
  }
}

async function resetDatabase() {
  if (!confirm('Are you sure you want to reset the document vector database? All document chunks will be deleted.')) {
    return;
  }
  try {
    const res = await fetch('/api/vector/reset', { method: 'POST' });
    const data = await res.json();
    alert(data.message || 'Database reset.');
    loadVectorStorageStatus();
  } catch (e) {
    alert(`Reset error: ${e.message}`);
  }
}

async function deleteDocument(docName) {
  if (!confirm(`Are you sure you want to delete document "${docName}" from the database? All its chunks will be deleted.`)) {
    return;
  }
  try {
    const res = await fetch('/api/vector/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document_name: docName })
    });
    const data = await res.json();
    if (data.status === 'success') {
      loadVectorStorageStatus();
    } else {
      alert(`Failed to delete document: ${data.message || 'Unknown error'}`);
    }
  } catch (e) {
    alert(`Error deleting document: ${e.message}`);
  }
}

async function updateSkillsDatabase() {
  const btn = document.getElementById('btn-update-skills');
  btn.disabled = true;
  btn.textContent = 'Scanning skills/...';

  try {
    const res = await fetch('/api/skills/update', { method: 'POST' });
    const data = await res.json();
    const info = data.data || {};
    alert(`Skills Database Updated! Newly loaded: ${info.loaded_skills?.length || 0}, Skipped existing: ${info.skipped_skills?.length || 0}`);
  } catch (e) {
    alert(`Error updating skills: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Update Skills Database';
  }
}

// ----------------- PAGE 3: Telemetry -----------------
function onTimeRangeChange() {
  const range = document.getElementById('chart-timerange').value;
  const customBox = document.getElementById('custom-date-range-box');
  if (range === 'Custom') {
    customBox.style.display = 'flex';
  } else {
    customBox.style.display = 'none';
  }
  fetchTelemetryData();
}

async function fetchTelemetryData() {
  const modelFilter = document.getElementById('telemetry-model-filter').value;
  const interval = document.getElementById('chart-interval').value;
  const timeRange = document.getElementById('chart-timerange').value;
  const customStart = document.getElementById('custom-start-date').value;
  const customEnd = document.getElementById('custom-end-date').value;

  let url = `/api/telemetry?model=${encodeURIComponent(modelFilter)}&interval=${encodeURIComponent(interval)}&time_range=${encodeURIComponent(timeRange)}`;
  if (timeRange === 'Custom' && customStart && customEnd) {
    url += `&custom_start=${encodeURIComponent(customStart)}&custom_end=${encodeURIComponent(customEnd)}`;
  }

  try {
    const res = await fetch(url);
    const data = await res.json();

    // Update Model Filter options
    const modelSelect = document.getElementById('telemetry-model-filter');
    const existingModels = new Set(Array.from(modelSelect.options).map(o => o.value));
    (data.models_used || []).forEach(m => {
      if (!existingModels.has(m)) {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        modelSelect.appendChild(opt);
      }
    });

    // Update KPI Tiles
    const totals = data.totals || {};
    document.getElementById('kpi-total-prompts').textContent = (totals.total_prompts || 0).toLocaleString();
    document.getElementById('kpi-total-responses').textContent = (totals.total_responses || 0).toLocaleString();
    document.getElementById('kpi-total-errors').textContent = (totals.total_errors || 0).toLocaleString();
    document.getElementById('kpi-input-tokens').textContent = (totals.total_input_tokens || 0).toLocaleString();
    document.getElementById('kpi-output-tokens').textContent = (totals.total_output_tokens || 0).toLocaleString();

    // Render Charts
    renderTelemetryCharts(data.charts || {});
  } catch (e) {
    console.error('Error loading telemetry:', e);
  }
}

function renderTelemetryCharts(chartData) {
  const labels = chartData.labels || [];
  const prompts = chartData.prompts || [];
  const responses = chartData.responses || [];
  const errors = chartData.errors || [];
  const inTokens = chartData.input_tokens || [];
  const outTokens = chartData.output_tokens || [];

  // Chart 1: Prompts, Responses, Errors
  const ctx1 = document.getElementById('chart-requests').getContext('2d');
  if (requestsChart) requestsChart.destroy();

  requestsChart = new Chart(ctx1, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Prompts',
          data: prompts,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          tension: 0.3,
          fill: true
        },
        {
          label: 'Responses',
          data: responses,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          tension: 0.3,
          fill: true
        },
        {
          label: 'Errors',
          data: errors,
          borderColor: '#f43f5e',
          backgroundColor: 'rgba(244, 63, 94, 0.1)',
          tension: 0.3,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#9ca3af' } }
      },
      scales: {
        x: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
      }
    }
  });

  // Chart 2: Input and Output Tokens
  const ctx2 = document.getElementById('chart-tokens').getContext('2d');
  if (tokensChart) tokensChart.destroy();

  tokensChart = new Chart(ctx2, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Input Tokens',
          data: inTokens,
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          tension: 0.3,
          fill: true
        },
        {
          label: 'Output Tokens',
          data: outTokens,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          tension: 0.3,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#9ca3af' } }
      },
      scales: {
        x: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
      }
    }
  });
}

// ----------------- PAGE 4: Audit Log & Event -----------------
async function fetchLogsData(selectedConvId = null) {
  let url = '/api/logs';
  if (selectedConvId) {
    url += `?conversation_id=${encodeURIComponent(selectedConvId)}`;
  }

  try {
    const res = await fetch(url);
    const data = await res.json();

    // KPI Metrics
    const stats = data.statistics || {};
    document.getElementById('kpi-log-prompts').textContent = (stats.total_user_prompts || 0).toLocaleString();
    document.getElementById('kpi-log-model-calls').textContent = (stats.total_model_calls || 0).toLocaleString();
    document.getElementById('kpi-log-embeds').textContent = (stats.total_ollama_embeds || 0).toLocaleString();
    document.getElementById('kpi-log-latency').textContent = `${stats.avg_latency_ms || 0} ms`;

    // Table 1: Conversations
    const convTbody = document.getElementById('conversations-tbody');
    convTbody.innerHTML = '';
    const convs = data.conversations || [];
    document.getElementById('conv-total-count').textContent = `Total Conversations: ${convs.length}`;

    if (convs.length === 0) {
      convTbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No conversation logs recorded yet.</td></tr>';
    } else {
      convs.forEach(c => {
        const tr = document.createElement('tr');
        const isSelected = (c.conversation_id === data.selected_conversation_id);
        if (isSelected) tr.className = 'selected';

        const localTimeStr = new Date(c.timestamp).toLocaleString();
        tr.innerHTML = `
          <td>${localTimeStr}</td>
          <td><code>${escapeHtml(c.conversation_id)}</code></td>
          <td>${escapeHtml((c.user_query || '').slice(0, 45))}${c.user_query?.length > 45 ? '...' : ''}</td>
          <td>${escapeHtml((c.agent_response || '').slice(0, 45))}${c.agent_response?.length > 45 ? '...' : ''}</td>
          <td><span class="evidence-badge">${escapeHtml(c.agent_type || 'Custom Agent')}</span></td>
          <td>${c.total_events}</td>
        `;
        tr.style.cursor = 'pointer';
        tr.onclick = () => {
          document.querySelectorAll('#conversations-tbody tr').forEach(r => r.classList.remove('selected'));
          tr.classList.add('selected');
          fetchLogsData(c.conversation_id);
        };
        convTbody.appendChild(tr);
      });
    }

    // Table 2: Events
    const eventsTbody = document.getElementById('events-tbody');
    eventsTbody.innerHTML = '';
    const events = data.events || [];
    allEventsCache = events;

    document.getElementById('events-table-title').textContent =
      `Events for Conversation for ${data.selected_conversation_id || 'None'}`;

    if (events.length === 0) {
      eventsTbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No events logged for this conversation.</td></tr>';
    } else {
      events.forEach((ev, idx) => {
        const tr = document.createElement('tr');
        const localTimeAndDate = new Date(ev.timestamp).toLocaleString();
        const callTypeTag = ev.call_type ? `<span style="font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; margin-left: 6px; background: ${ev.call_type === 'response' ? 'rgba(16, 185, 129, 0.15); color: #34d399;' : 'rgba(99, 102, 241, 0.15); color: #818cf8;'}">${escapeHtml(ev.call_type)}</span>` : '';
        tr.innerHTML = `
          <td>${localTimeAndDate}</td>
          <td><span class="evidence-badge">${escapeHtml(ev.event_type)}</span>${callTypeTag}</td>
          <td>${escapeHtml(ev.invoker)}</td>
          <td>${escapeHtml(ev.target || ev.recipient || '')}</td>
          <td>${escapeHtml(ev.description || '')}</td>
        `;
        tr.style.cursor = 'pointer';
        tr.onclick = () => openJsonModal(idx);
        eventsTbody.appendChild(tr);
      });
    }

  } catch (e) {
    console.error('Error loading logs:', e);
  }
}

function openJsonModal(eventIndex) {
  const ev = allEventsCache[eventIndex];
  if (!ev) return;

  const callTypeStr = ev.call_type ? ` [${ev.call_type.toUpperCase()}]` : '';
  document.getElementById('json-modal-title').textContent = `${ev.event_type}${callTypeStr} (${ev.invoker} ➔ ${ev.target || ev.recipient})`;
  document.getElementById('json-modal-content').textContent = JSON.stringify(ev, null, 2);
  openModal('modal-json-detail');
}

function openClearLogsModal() {
  openModal('modal-clear-logs');
}

async function executeClearLogs() {
  closeModal('modal-clear-logs');
  try {
    const res = await fetch('/api/logs/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirm: true })
    });
    const data = await res.json();
    alert(data.message || 'Audit logs cleared.');
    fetchLogsData();
  } catch (e) {
    alert(`Error clearing logs: ${e.message}`);
  }
}

// ----------------- Modal Utility -----------------
function openModal(modalId) {
  const m = document.getElementById(modalId);
  if (m) m.classList.add('active');
}

function closeModal(modalId) {
  const m = document.getElementById(modalId);
  if (m) m.classList.remove('active');
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
