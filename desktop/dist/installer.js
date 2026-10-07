// Private AI — Native guided installer frontend.
//
// Runs inside the Tauri webview (or a plain browser for development). It drives
// the 8 setup screens (A–H) and calls native Rust commands via `window.__TAURI__`
// when available. When the AI server is reachable it offers to launch the web UI.

const isTauri = () => typeof window !== 'undefined' && !!window.__TAURI__;
const invoke = (cmd, args) => {
  if (!isTauri()) {
    return Promise.reject(new Error('Native command "' + cmd + '" unavailable in browser mode.'));
  }
  return window.__TAURI__.core.invoke(cmd, args || {});
};

// Force the installer to show even when the server is running (for testing).
const forceSetup = new URLSearchParams(window.location.search).has('setup');

const SERVICES = [
  { key: 'ai', name: 'Private AI Server', url: 'http://127.0.0.1:8005/health' },
  { key: 'imagegen', name: 'Open Fantasia', url: 'http://127.0.0.1:8765/health' },
  { key: 'voice', name: 'Olly Voice', url: 'http://127.0.0.1:8766/health' },
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  statuses: {},      // key -> { reachable, detail }
  selected: { ai: true, imagegen: false, voice: false },
  mode: 'local',     // 'local' | 'docker'
  installDir: '',
  modelsDir: '',
  launch: 'desktop', // 'desktop' | 'web'
  startOnBoot: true,
};

const STEP_LABELS = [
  'Detect', 'Services', 'Run mode', 'Models', 'Install', 'Test', 'Launch', 'Done',
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const show = (id) => {
  document.querySelectorAll('.screen').forEach((s) => (s.hidden = true));
  const el = $(id);
  el.hidden = false;
  renderSteps(el.dataset.step);
};

function renderSteps(active) {
  const nav = $('#stepNav');
  nav.innerHTML = STEP_LABELS
    .map((label, i) => {
      const n = i + 1;
      const cls = n < active ? 'done' : n === active ? 'active' : '';
      return (
        '<div class="step ' + cls + '">' +
        '<div class="num">' + (n < active ? '✓' : n) + '</div>' +
        '<div>' + label + '</div>' +
        '</div>'
      );
    })
    .join('');
}

function toast(msg, isError) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  t.className = 'toast' + (isError ? ' error' : '');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => (t.hidden = true), 4000);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ---------------------------------------------------------------------------
// Service probing
// ---------------------------------------------------------------------------
async function probeAll() {
  // In Tauri, use the native probe command (fast, parallel). Fall back to
  // fetch for browser dev.
  if (isTauri()) {
    try {
      const list = await invoke('probe_services');
      list.forEach((s) => {
        state.statuses[s.key] = { reachable: s.reachable, detail: s.detail };
      });
      return;
    } catch (e) {
      /* fall through to fetch */
    }
  }
  await Promise.all(
    SERVICES.map(async (svc) => {
      try {
        const r = await fetch(svc.url, { method: 'GET' });
        state.statuses[svc.key] = { reachable: r.ok, detail: r.ok ? 'OK' : 'HTTP ' + r.status };
      } catch (e) {
        state.statuses[svc.key] = { reachable: false, detail: 'unreachable' };
      }
    })
  );
}

// ---------------------------------------------------------------------------
// Screen A — Detect running servers
// ---------------------------------------------------------------------------
async function renderDetect() {
  const list = $('#detectList');
  list.innerHTML = '<div class="service pending">Detecting services…</div>';
  await probeAll();
  list.innerHTML = SERVICES.map((svc) => {
    const st = state.statuses[svc.key] || { reachable: false, detail: 'unknown' };
    const badge = st.reachable
      ? '<span class="badge ok">running</span>'
      : '<span class="badge down">not running</span>';
    return (
      '<div class="service">' +
      '<div class="svc-info"><div class="svc-name">' + esc(svc.name) + '</div>' +
      '<div class="svc-url">' + esc(svc.url) + '</div></div>' + badge +
      '</div>'
    );
  }).join('');
}

// ---------------------------------------------------------------------------
// Screen B — Choose services
// ---------------------------------------------------------------------------
function renderServices() {
  const list = $('#servicesList');
  list.innerHTML = SERVICES.map((svc) => {
    const st = state.statuses[svc.key] || { reachable: false };
    // AI server is always required; siblings default to checked if running.
    if (svc.key !== 'ai' && st.reachable) state.selected[svc.key] = true;
    const checked = svc.key === 'ai' || state.selected[svc.key] ? 'checked' : '';
    const badge = st.reachable
      ? '<span class="badge ok">running</span>'
      : '<span class="badge idle">not installed</span>';
    const disabled = svc.key === 'ai' ? ' style="opacity:0.6"' : '';
    return (
      '<div class="service ' + checked + '" data-key="' + svc.key + '"' + disabled + '>' +
      '<div class="checkbox">' + (checked ? '✓' : '') + '</div>' +
      '<div class="svc-info"><div class="svc-name">' + esc(svc.name) + '</div>' +
      '<div class="svc-url">' + esc(svc.url) + '</div></div>' + badge +
      '</div>'
    );
  }).join('');

  list.querySelectorAll('.service[data-key]').forEach((row) => {
    row.addEventListener('click', () => {
      const key = row.dataset.key;
      if (key === 'ai') return; // always required
      state.selected[key] = !state.selected[key];
      row.classList.toggle('checked', state.selected[key]);
      row.querySelector('.checkbox').textContent = state.selected[key] ? '✓' : '';
    });
  });
}

// ---------------------------------------------------------------------------
// Screen C — Deployment mode + install dir
// ---------------------------------------------------------------------------
function setMode(mode) {
  state.mode = mode;
  $('#modeLocal').classList.toggle('selected', mode === 'local');
  $('#modeDocker').classList.toggle('selected', mode === 'docker');
}
function renderMode() {
  setMode(state.mode || 'local');
}

// ---------------------------------------------------------------------------
// Screen E — Requirements checklist
// ---------------------------------------------------------------------------
async function runRequirements() {
  const list = $('#reqChecklist');
  const items = [];

  // Build the checklist based on mode + selected services.
  if (state.mode === 'docker') {
    items.push({ name: 'Docker installed', check: () => invoke('check_docker') });
    items.push({ name: 'Start Docker stack', check: () => invoke('docker_up') });
  } else {
    items.push({ name: 'Python detected', check: () => invoke('check_python') });
    items.push({ name: 'CUDA / GPU detected', check: () => invoke('check_cuda') });
  }
  if (state.selected.imagegen) {
    items.push({
      name: 'Open Fantasia installed',
      check: () => invoke('clone_sibling', { repo: 'open-fantasia-imagegen', installDir: state.installDir }),
    });
  }
  if (state.selected.voice) {
    items.push({
      name: 'Olly Voice installed',
      check: () => invoke('clone_sibling', { repo: 'olly-voice-server', installDir: state.installDir }),
    });
  }
  items.push({
    name: 'Write .env configuration',
    check: () => invoke('write_env', { installDir: state.installDir, modelsDir: state.modelsDir, mode: state.mode }),
  });
  items.push({ name: 'Start services', check: () => invoke('start_services', { mode: state.mode }) });

  list.innerHTML = items
    .map((it, i) => (
      '<div class="req pending" data-i="' + i + '">' +
      '<div class="status"></div>' +
      '<div class="req-name">' + esc(it.name) + '</div>' +
      '<div class="req-msg"></div>' +
      '</div>'
    ))
    .join('');

  const rows = [...list.children];
  for (let i = 0; i < items.length; i++) {
    const row = rows[i];
    row.classList.remove('pending');
    row.classList.add('running');
    try {
      await items[i].check();
      row.classList.remove('running');
      row.classList.add('done');
      row.querySelector('.status').textContent = '✓';
    } catch (e) {
      row.classList.remove('running');
      row.classList.add('error');
      row.querySelector('.status').textContent = '✕';
      row.querySelector('.req-msg').textContent = ' ' + esc(e.message || e);
      // Add a retry button.
      row.querySelector('.status').insertAdjacentHTML(
        'afterend',
        '<button class="btn test retry" data-i="' + i + '">Retry</button>'
      );
    }
  }

  // Show Continue once all steps are done (or have failed but user retries).
  const allDone = () => [...rows].every((r) => r.classList.contains('done'));
  list.addEventListener('click', async (ev) => {
    const btn = ev.target.closest('.retry');
    if (!btn) return;
    const i = +btn.dataset.i;
    const row = rows[i];
    row.classList.remove('error');
    row.classList.add('running');
    row.querySelector('.status').textContent = '';
    btn.remove();
    try {
      await items[i].check();
      row.classList.remove('running');
      row.classList.add('done');
      row.querySelector('.status').textContent = '✓';
    } catch (e) {
      row.classList.remove('running');
      row.classList.add('error');
      row.querySelector('.status').textContent = '✕';
      row.querySelector('.req-msg').textContent = ' ' + esc(e.message || e);
      row.querySelector('.status').insertAdjacentHTML(
        'afterend',
        '<button class="btn test retry" data-i="' + i + '">Retry</button>'
      );
    }
    if (allDone()) $('#eContinue').hidden = false;
  });

  // Enable continue if everything succeeded on first pass.
  const timer = setInterval(() => {
    if (allDone()) {
      $('#eContinue').hidden = false;
      clearInterval(timer);
    }
  }, 300);
}

// ---------------------------------------------------------------------------
// Screen F — Service status + test connections
// ---------------------------------------------------------------------------
function renderStatus() {
  const list = $('#statusList');
  list.innerHTML = SERVICES.filter((s) => state.selected[s.key]).map((svc) => {
    const st = state.statuses[svc.key] || { reachable: false, detail: 'unknown' };
    const badge = st.reachable ? 'ok' : 'down';
    return (
      '<div class="service" data-key="' + svc.key + '">' +
      '<div class="checkbox" style="background:var(--success);border-color:var(--success)">✓</div>' +
      '<div class="svc-info"><div class="svc-name">' + esc(svc.name) + '</div>' +
      '<div class="svc-url" data-url="' + esc(svc.url) + '">' + esc(svc.url) + '</div></div>' +
      '<span class="badge ' + badge + '" data-badge>' + (st.reachable ? 'running' : 'down') + '</span>' +
      '<button class="btn test" data-test="' + svc.key + '">Test connection</button>' +
      '</div>'
    );
  }).join('');

  list.querySelectorAll('[data-test]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const key = btn.dataset.test;
      btn.disabled = true;
      btn.textContent = 'Testing…';
      const svc = SERVICES.find((s) => s.key === key);
      let ok = false;
      try {
        if (isTauri()) {
          const st = await invoke('test_connection', { key });
          ok = st.reachable;
        } else {
          const r = await fetch(svc.url);
          ok = r.ok;
        }
      } catch (e) {
        ok = false;
      }
      const row = btn.closest('.service');
      const badge = row.querySelector('[data-badge]');
      badge.className = 'badge ' + (ok ? 'ok' : 'down');
      badge.textContent = ok ? 'running' : 'down';
      btn.disabled = false;
      btn.textContent = 'Test connection';
      toast(ok ? svc.name + ' is running.' : svc.name + ' is not reachable.', !ok);
    });
  });
}

// ---------------------------------------------------------------------------
// Navigation / wiring
// ---------------------------------------------------------------------------
function wire() {
  document.querySelectorAll('[data-back]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const cur = document.querySelector('.screen:not([hidden])');
      const step = +cur.dataset.step;
      show('#screen-' + 'abcdefgh'[step - 2]);
    });
  });

  $('#aContinue').addEventListener('click', () => {
    // If the AI server is already running, offer to skip straight to launch.
    if (state.statuses.ai && state.statuses.ai.reachable) {
      // Pre-select siblings that are running.
      show('#screen-g');
      renderLaunch();
    } else {
      show('#screen-b');
      renderServices();
    }
  });

  $('#bContinue').addEventListener('click', () => {
    show('#screen-c');
    renderMode();
    loadDefaultDirs();
  });

  $('#modeLocal').addEventListener('click', () => setMode('local'));
  $('#modeDocker').addEventListener('click', () => setMode('docker'));

  $('#pickInstallDir').addEventListener('click', async () => {
    try {
      const p = await invoke('pick_folder', { title: 'Choose installation directory' });
      if (p) state.installDir = p;
      $('#installDir').value = state.installDir;
    } catch (e) { toast(e.message, true); }
  });
  $('#pickModelsDir').addEventListener('click', async () => {
    try {
      const p = await invoke('pick_folder', { title: 'Choose model storage directory' });
      if (p) state.modelsDir = p;
      $('#modelsDir').value = state.modelsDir;
    } catch (e) { toast(e.message, true); }
  });

  $('#cContinue').addEventListener('click', () => { show('#screen-d'); });
  $('#dContinue').addEventListener('click', () => {
    show('#screen-e');
    runRequirements();
  });
  $('#eBack').addEventListener('click', () => show('#screen-d'));
  $('#eContinue').addEventListener('click', () => {
    show('#screen-f');
    renderStatus();
  });
  $('#fContinue').addEventListener('click', () => {
    show('#screen-g');
    renderLaunch();
  });

  $('#launchDesktop').addEventListener('click', () => setLaunch('desktop'));
  $('#launchWeb').addEventListener('click', () => setLaunch('web'));

  $('#rerunSetup').addEventListener('click', () => {
    // Start the full installer flow from the beginning.
    show('#screen-b');
    renderServices();
  });

  $('#gContinue').addEventListener('click', () => {
    show('#screen-h');
  });

  $('#hFinish').addEventListener('click', async () => {
    // Apply start-on-startup if requested.
    if ($('#startOnBoot').checked) {
      try {
        // Auto-start the bundled server at logon (no admin).
        const serverCmd = isTauri() ? await bundledServerLaunchCmd() : null;
        if (serverCmd) {
          await invoke('register_autostart', {
            taskName: 'PrivateAI-Services',
            command: serverCmd,
          });
        } else {
          toast('Server auto-start not available in browser mode.', true);
        }
      } catch (e) {
        toast('Could not register autostart: ' + e.message, true);
      }
    }
    await launch();
  });
}

// ---------------------------------------------------------------------------
// Screen G — Launch choice
// ---------------------------------------------------------------------------
function setLaunch(mode) {
  state.launch = mode;
  $('#launchDesktop').classList.toggle('selected', mode === 'desktop');
  $('#launchWeb').classList.toggle('selected', mode === 'web');
}
function renderLaunch() {
  setLaunch(state.launch || 'desktop');
}

// ---------------------------------------------------------------------------
// Launch — start the server (if needed), then open the web UI
// ---------------------------------------------------------------------------
// Resolve a command line that launches the bundled server (for autostart).
async function bundledServerLaunchCmd() {
  try {
    const exe = await invoke('server_binary_path');
    return 'cmd /c start "" /min "' + exe + '" --port 8005';
  } catch (e) {
    return null;
  }
}

async function launch() {
  const webUrl = isTauri() ? (await invoke('web_ui_url').catch(() => 'http://127.0.0.1:8005')) : 'http://127.0.0.1:8005';

  // Ensure the server is running before opening the UI.
  if (isTauri()) {
    try {
      await invoke('start_bundled_server');
      // Give the server a moment to come up.
      await new Promise((r) => setTimeout(r, 1500));
    } catch (e) {
      toast('Could not start the server: ' + e.message, true);
    }
  }

  if (state.launch === 'desktop' && isTauri()) {
    // Navigate this Tauri window to the web UI.
    window.location.href = webUrl;
    return;
  }

  // Web mode (or browser dev): open the web UI in the default browser.
  if (isTauri()) {
    try {
      await window.__TAURI__.shell.open(webUrl);
    } catch (e) {
      toast('Could not open browser: ' + e.message, true);
    }
  } else {
    window.open(webUrl, '_blank');
  }
  toast('Private AI is running at ' + webUrl);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function loadDefaultDirs() {
  (async () => {
    if (isTauri()) {
      try {
        state.installDir = await invoke('default_install_dir');
        state.modelsDir = await invoke('default_models_dir');
      } catch (e) { /* keep empty */ }
    } else {
      state.installDir = 'C:\\PrivateAI';
      state.modelsDir = 'C:\\Users\\' + (window.__TAURI__ ? '' : 'you') + '\\ai-server-models';
    }
    $('#installDir').value = state.installDir;
    $('#modelsDir').value = state.modelsDir;
  })();
}

async function boot() {
  renderSteps(1);

  // If the AI server is already running and we're not forcing setup, offer to
  // go straight to the app.
  if (!forceSetup) {
    await probeAll();
    if (state.statuses.ai && state.statuses.ai.reachable) {
      // Server already up — show a lightweight "already running" landing that
      // lets the user launch immediately or re-run setup.
      show('#screen-g');
      renderLaunch();
      return;
    }
  }

  show('#screen-a');
  await renderDetect();
  wire();
}

boot();
