// ------------------------------------------------------------------
// Private AI Inference Server — chat UI logic
// ------------------------------------------------------------------

// State
let conversation = [];
let abortController = null;
let genStartTime = null;

const els = {
  model: document.getElementById('model'),
  modelMeta: document.getElementById('modelMeta'),
  messages: document.getElementById('messages'),
  message: document.getElementById('message'),
  submit: document.getElementById('submit'),
  stop: document.getElementById('stopBtn'),
  stream: document.getElementById('streamToggle'),
  status: document.getElementById('serverStatus'),
  gpuMem: document.getElementById('gpuMem'),
  uptime: document.getElementById('uptime'),
  modelCount: document.getElementById('modelCount'),
  tokens: document.getElementById('tokens'),
  tokPerSec: document.getElementById('tokPerSec'),
  genTime: document.getElementById('genTime'),
};

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\n/g, '<br>');
}

function fmtBytes(n) {
  if (!n) return '—';
  const gb = n / (1024 ** 3);
  if (gb >= 1) return gb.toFixed(1) + ' GB';
  return (n / (1024 ** 2)).toFixed(1) + ' MB';
}

function fmtUptime(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h + 'h ' + m + 'm';
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

// ------------------------------------------------------------------
// Model list + server health
// ------------------------------------------------------------------
async function refreshModels() {
  try {
    const [tagsRes, healthRes] = await Promise.all([
      fetch('/api/tags'),
      fetch('/health'),
    ]);
    const tags = await tagsRes.json();
    const health = await healthRes.json();
    const models = tags.models || [];
    const services = tags.services || {};

    const prev = els.model.value;
    els.model.innerHTML = '<option value="">Select a model</option>';

    // Group models by modality
    const groups = {
      chat: [],
      vision: [],
      imagegen: [],
      voice: [],
      embeddings: [],
      unknown: [],
    };
    models.forEach((m) => {
      const mod = m.modality || 'unknown';
      if (!groups[mod]) groups[mod] = [];
      groups[mod].push(m);
    });

    const groupLabels = {
      chat: 'Chat',
      vision: 'Vision',
      imagegen: 'Image Gen',
      voice: 'Voice',
      embeddings: 'Embeddings',
      unknown: 'Other',
    };
    const groupOrder = ['chat', 'vision', 'imagegen', 'voice', 'embeddings', 'unknown'];

    groupOrder.forEach((mod) => {
      const list = groups[mod];
      if (!list || list.length === 0) return;
      const og = document.createElement('optgroup');
      og.label = groupLabels[mod];
      list.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m.model;
        opt.text = m.name;
        // Only chat/vision models are usable for chat on this server
        const usable = mod === 'chat' || mod === 'vision';
        if (!usable) opt.disabled = true;
        og.appendChild(opt);
      });
      els.model.appendChild(og);
    });

    if (prev) els.model.value = prev;
    els.modelCount.textContent = models.length;

    els.gpuMem.textContent =
      health.gpu_memory_allocated_mb != null
        ? health.gpu_memory_allocated_mb.toFixed(0) + ' MB'
        : 'CPU';
    els.uptime.textContent = fmtUptime(health.uptime_seconds || 0);
    els.status.textContent = models.length + ' models';
    els.status.className = 'badge bg-success';
    updateModelMeta();
  } catch (err) {
    console.error(err);
    els.status.textContent = 'Server unreachable';
    els.status.className = 'badge bg-danger';
  }
}

function updateModelMeta() {
  const name = els.model.value;
  if (!name) {
    els.modelMeta.style.display = 'none';
    return;
  }
  fetch('/models')
    .then((r) => r.json())
    .then((data) => {
      const m = (data.models || []).find((x) => x.model === name);
      if (m) {
        els.modelMeta.style.display = 'block';
        let html =
          'Backend: <b>' + (m.backend || '—') + '</b><br>' +
          'Modality: <b>' + (m.modality || 'unknown') + '</b><br>' +
          'Size: <b>' + fmtBytes(m.size_bytes) + '</b><br>' +
          'Loaded: <b>' + (m.loaded ? 'yes' : 'no') + '</b>';
        // Show a hint for models handled by sibling servers
        if (m.modality === 'imagegen') {
          html += '<br><span style="color:var(--warning)">→ Open Fantasia (:8765)</span>';
        } else if (m.modality === 'voice') {
          html += '<br><span style="color:var(--warning)">→ Olly Voice (:8766)</span>';
        }
        els.modelMeta.innerHTML = html;
      }
    })
    .catch(() => {});
}

// ------------------------------------------------------------------
// Rendering
// ------------------------------------------------------------------
function addMessage(role, content, stats) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');

  // Split out reasoning (think) blocks
  let body = content;
  const thinkMatch = content.match(/^ thinking([\s\S]*?)<\/think>/);
  if (thinkMatch) {
    const thinkDiv = document.createElement('div');
    thinkDiv.className = 'think';
    thinkDiv.innerHTML = escapeHtml(thinkMatch[1].trim());
    wrap.appendChild(thinkDiv);
    body = content.slice(thinkMatch[0].length);
  }

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = body.trim();
  wrap.appendChild(bubble);

  if (stats) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = stats;
    wrap.appendChild(meta);
  }

  // Copy button for assistant messages
  if (role !== 'user') {
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(content);
      copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
      setTimeout(() => {
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
      }, 1500);
    });
    wrap.appendChild(copyBtn);
  }

  els.messages.appendChild(wrap);
  scrollToBottom();
}

function addWritingIndicator() {
  const w = document.createElement('div');
  w.className = 'writing';
  w.id = 'writing';
  w.textContent = 'Writing';
  els.messages.appendChild(w);
  scrollToBottom();
}

function removeWritingIndicator() {
  const w = document.getElementById('writing');
  if (w) w.remove();
}

// ------------------------------------------------------------------
// Generation stats
// ------------------------------------------------------------------
function setGenStats(done, evalCount, evalDurationNs, totalDurationNs) {
  if (done) {
    const sec = evalDurationNs / 1e9;
    const tokPerSec = evalCount && sec > 0 ? (evalCount / sec).toFixed(1) : '—';
    els.tokens.textContent = evalCount != null ? evalCount : '—';
    els.tokPerSec.textContent = tokPerSec;
    els.genTime.textContent = sec.toFixed(2) + 's';
  }
}

// ------------------------------------------------------------------
// Chat request
// ------------------------------------------------------------------
function buildPayload(userMessage) {
  const selectedModel = els.model.value;
  const stream = els.stream.checked;
  return {
    model: selectedModel,
    stream: stream,
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      ...conversation,
      userMessage,
    ],
  };
}

async function sendChat() {
  const text = els.message.value.trim();
  if (!text) return;
  if (!els.model.value) {
    alert('Please select a model first.');
    return;
  }

  const userMessage = { role: 'user', content: text };
  conversation.push(userMessage);
  addMessage('user', text);
  els.message.value = '';
  addWritingIndicator();
  els.stop.disabled = false;
  genStartTime = performance.now();

  const payload = buildPayload(userMessage);
  abortController = new AbortController();

  try {
    let fullText = '';
    let evalCount = null;
    let evalDurationNs = 0;
    let totalDurationNs = 0;

    if (payload.stream) {
      // Ollama NDJSON streaming (one JSON object per line)
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let assistantBubble = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const chunk = JSON.parse(trimmed);
            // Ollama streams message.content deltas
            if (chunk.message && chunk.message.content) {
              fullText += chunk.message.content;
              if (!assistantBubble) {
                assistantBubble = document.createElement('div');
                assistantBubble.className = 'msg assistant';
                const b = document.createElement('div');
                b.className = 'bubble';
                assistantBubble.appendChild(b);
                els.messages.appendChild(assistantBubble);
              }
              assistantBubble.querySelector('.bubble').textContent = fullText;
              scrollToBottom();
            }
            if (chunk.done) {
              evalCount = chunk.eval_count || chunk.prompt_eval_count || null;
              evalDurationNs = chunk.eval_duration || 0;
              totalDurationNs = chunk.total_duration || 0;
            }
          } catch (e) { /* ignore partial */ }
        }
      }

      // finalize the assistant message
      if (assistantBubble) {
        const stats = '⚡ ' + (evalCount || '—') + ' tokens';
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = stats;
        assistantBubble.appendChild(meta);
        addCopyButton(assistantBubble, fullText);
      }
      conversation.push({ role: 'assistant', content: fullText });
      setGenStats(true, evalCount, evalDurationNs, totalDurationNs);
    } else {
      // Non-streaming
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      fullText = (data.message && data.message.content) || '';
      evalCount = data.eval_count || null;
      evalDurationNs = data.eval_duration || 0;
      totalDurationNs = data.total_duration || 0;
      addMessage('assistant', fullText, '⚡ ' + (evalCount || '—') + ' tokens');
      conversation.push({ role: 'assistant', content: fullText });
      setGenStats(true, evalCount, evalDurationNs, totalDurationNs);
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      addMessage('assistant', '⏹️ Generation stopped.');
    } else {
      console.error(err);
      addMessage('assistant', 'Error: ' + err.message);
    }
  } finally {
    removeWritingIndicator();
    els.stop.disabled = true;
  }
}

function addCopyButton(wrap, text) {
  const copyBtn = document.createElement('button');
  copyBtn.className = 'copy-btn';
  copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(text);
    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
    setTimeout(() => {
      copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
    }, 1500);
  });
  wrap.appendChild(copyBtn);
}

// ------------------------------------------------------------------
// Pull model
// ------------------------------------------------------------------
async function pullModel() {
  const input = document.getElementById('pullInput');
  const status = document.getElementById('pullStatus');
  const model = input.value.trim();
  if (!model) {
    alert('Enter a model id to pull (e.g. hf.co/org/repo or path.gguf)');
    return;
  }
  status.style.display = 'block';
  status.textContent = 'Starting pull of ' + model + '…';
  document.getElementById('pullBtn').disabled = true;
  try {
    const res = await fetch('/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: model, quantize: 'auto', init: false }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const jobId = data.job_id;
    status.textContent = 'Pulling… (job ' + jobId.slice(0, 8) + ')';

    // Poll job progress
    const poll = setInterval(async () => {
      try {
        const jres = await fetch('/jobs/' + jobId);
        const job = await jres.json();
        if (job.status === 'succeeded') {
          clearInterval(poll);
          status.textContent = '✅ Pulled ' + model;
          document.getElementById('pullBtn').disabled = false;
          input.value = '';
          refreshModels();
        } else if (job.status === 'failed') {
          clearInterval(poll);
          status.textContent = '❌ Failed: ' + (job.error || 'unknown error');
          status.style.color = 'var(--danger)';
          document.getElementById('pullBtn').disabled = false;
        } else {
          const dl = job.downloaded_bytes || 0;
          const gb = (dl / (1024 ** 3)).toFixed(2);
          status.textContent = 'Downloading… ' + gb + ' GB';
        }
      } catch (e) {
        clearInterval(poll);
        status.textContent = 'Error polling job';
        document.getElementById('pullBtn').disabled = false;
      }
    }, 2000);
  } catch (err) {
    status.textContent = '❌ ' + err.message;
    status.style.color = 'var(--danger)';
    document.getElementById('pullBtn').disabled = false;
  }
}

// ------------------------------------------------------------------
// Event wiring
// ------------------------------------------------------------------
function init() {
  els.submit.addEventListener('click', sendChat);

  els.message.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  document.getElementById('refreshModels').addEventListener('click', refreshModels);
  document.getElementById('clearChat').addEventListener('click', () => {
    els.messages.innerHTML = '';
    conversation = [];
    els.tokens.textContent = '—';
    els.tokPerSec.textContent = '—';
    els.genTime.textContent = '—';
  });
  document.getElementById('pullBtn').addEventListener('click', pullModel);
  document.getElementById('pullInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); pullModel(); }
  });
  els.stop.addEventListener('click', () => {
    if (abortController) abortController.abort();
  });
  els.model.addEventListener('change', updateModelMeta);

  refreshModels();
}

// Start
init();
