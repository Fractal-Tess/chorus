import { Slot, formatTime, peaksFrom } from './player.js';

const TAGS = ['A', 'B'];
const FIXED_SPEED_ENGINES = new Set(['pocket', 'breeze', 'fish']);
const ENGLISH_ALIGNMENT = new Set(['a', 'b', 'en', 'en-US', 'en-us', 'en-gb', 'english']);
const VOICE_INPUT_LABELS = { description: 'Voice design', style: 'Style control' };
const LANGUAGE_NAMES = {
  a: 'American English', b: 'British English', e: 'Spanish', f: 'French', h: 'Hindi', i: 'Italian',
  j: 'Japanese', p: 'Portuguese', z: 'Mandarin', en: 'English', 'en-US': 'English (US)',
  english: 'English', french_24l: 'French', spanish_24l: 'Spanish', german_24l: 'German',
  italian_24l: 'Italian', portuguese_24l: 'Portuguese', ko: 'Korean', ja: 'Japanese', ar: 'Arabic',
  bg: 'Bulgarian', cs: 'Czech', da: 'Danish', de: 'German', el: 'Greek', es: 'Spanish',
  et: 'Estonian', fi: 'Finnish', fr: 'French', hi: 'Hindi', hr: 'Croatian', hu: 'Hungarian',
  id: 'Indonesian', it: 'Italian', lt: 'Lithuanian', lv: 'Latvian', nl: 'Dutch', pl: 'Polish',
  pt: 'Portuguese', ro: 'Romanian', ru: 'Russian', sk: 'Slovak', sl: 'Slovenian', sv: 'Swedish',
  tr: 'Turkish', uk: 'Ukrainian', vi: 'Vietnamese', zh: 'Chinese', na: 'Language agnostic',
};
const VOICE_LANGUAGE_HINTS = {
  giovanni: 'italian_24l', lola: 'spanish_24l', juergen: 'german_24l',
  rafael: 'portuguese_24l', estelle: 'french_24l',
};

const $ = (selector) => document.querySelector(selector);
const el = Object.fromEntries([
  'engine-chips', 'composer', 'speech-text', 'char-count', 'voice-field', 'voice-title',
  'voice-select', 'voice-description', 'language-select', 'model-select', 'speed-input',
  'speed-value', 'channel-select', 'channel-default', 'channel-status', 'lava-sr-input',
  'force-align-input', 'form-note', 'generate-button', 'generate-label', 'engine-name',
  'engine-summary', 'voice-count', 'sample-rate', 'loaded-badge', 'error-panel', 'health-dot',
  'health-label', 'diagnostic-devices', 'diagnostic-models', 'diagnostic-vram',
  'diagnostic-requests', 'slots', 'audition-switch', 'stop-button', 'swap-button',
  'link-playhead', 'compare-hint', 'alignment-panel', 'alignment-owner', 'alignment-words',
  'target-switch', 'theme-switch', 'server-address',
].map((id) => [id.replace(/-(.)/g, (_, char) => char.toUpperCase()), $(`#${id}`)]));

const state = { engines: [], models: [], engine: null, model: null, channel: null, target: 'auto', focus: null, generating: false };

const slots = TAGS.map((tag) => {
  const node = $('#slot-template').content.firstElementChild.cloneNode(true);
  el.slots.append(node);
  const slot = new Slot(node, onSlotEvent);
  slot.setTag(tag);
  return slot;
});

/* ---------- comparison ---------- */

function onSlotEvent(slot, event) {
  if (event === 'play') {
    slots.forEach((other) => other !== slot && other.pause());
    setFocus(slot);
  } else if (event === 'load') {
    setFocus(slot);
  } else if (event === 'clear' && state.focus === slot) {
    setFocus(slots.find((other) => !other.isEmpty) || null);
  }
  refreshCompare();
}

function setFocus(slot) {
  state.focus = slot;
  const words = slot?.take?.alignment?.words || [];
  el.alignmentPanel.hidden = words.length === 0;
  el.alignmentOwner.textContent = slot ? slot.tag : '—';
  el.alignmentWords.replaceChildren(...words.map((word) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'word';
    button.title = `Confidence ${(word.score * 100).toFixed(1)}%`;
    button.append(
      Object.assign(document.createElement('b'), { textContent: word.word }),
      Object.assign(document.createElement('span'), { textContent: `${word.start_ms}–${word.end_ms} ms` }),
    );
    button.addEventListener('click', () => slot.seek(word.start_ms / 1000));
    return button;
  }));
}

function refreshCompare() {
  const filled = slots.filter((slot) => !slot.isEmpty);
  el.auditionSwitch.querySelectorAll('button').forEach((button, index) => {
    button.disabled = slots[index].isEmpty;
    button.textContent = `Play ${TAGS[index]}`;
    button.setAttribute('aria-pressed', String(slots[index].playing));
  });
  el.stopButton.disabled = !slots.some((slot) => slot.playing);
  el.swapButton.disabled = filled.length === 0;
  el.compareHint.textContent = filled.length < 2
    ? 'Render twice and switch between the takes without losing your place.'
    : slots[0].take.script === slots[1].take.script
      ? 'Both takes use the same script, so the playhead lines up.'
      : 'Heads up: these takes were rendered from different scripts.';
}

function audition(index) {
  const slot = slots[index];
  if (slot.isEmpty) return;
  const linked = el.linkPlayhead.checked && state.focus && state.focus !== slot;
  slot.play(linked ? state.focus.time : null);
}

function swapSlots() {
  slots.reverse();
  slots.forEach((slot, index) => slot.setTag(TAGS[index]));
  el.slots.append(...slots.map((slot) => slot.node));
  setFocus(state.focus);
  refreshCompare();
}

function targetSlot() {
  if (state.target !== 'auto') return slots[state.target];
  return slots.find((slot) => slot.isEmpty) || slots.find((slot) => slot !== state.focus) || slots[0];
}

/* ---------- catalog and form ---------- */

function fillSelect(select, values, selected, label = (value) => value) {
  select.replaceChildren(...values.map((value) => new Option(label(value), value, false, value === selected)));
}

function renderEngineChips() {
  el.engineChips.replaceChildren(...state.engines.map((engine) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.setAttribute('aria-pressed', String(engine.id === state.engine?.id));
    chip.title = engine.loaded ? 'Model is warm' : 'First render loads the model';
    const dot = document.createElement('span');
    dot.className = `chip__dot${engine.loaded ? ' is-warm' : ''}`;
    chip.append(dot, engine.label);
    chip.addEventListener('click', () => selectEngine(engine.id));
    return chip;
  }));
}

function channelStatus(channel) {
  return state.model?.channels?.find((item) => item.id === channel)
    || { id: channel, supported: false, enabled: false, available: false };
}

function channelLabel(status) {
  if (!status.supported) return `${status.id.toUpperCase()} — unsupported`;
  if (!status.enabled) return `${status.id.toUpperCase()} — disabled by policy`;
  if (!status.available) return `${status.id.toUpperCase()} — unavailable`;
  return `${status.id.toUpperCase()} — ready`;
}

function refreshChannel() {
  const configured = state.model?.default_channel || 'cpu';
  el.channelDefault.textContent = `Default ${configured.toUpperCase()}`;
  el.channelSelect.replaceChildren(...['cpu', 'gpu'].map((channel) => {
    const status = channelStatus(channel);
    const option = new Option(channelLabel(status), channel, false, channel === state.channel);
    option.disabled = !status.available;
    return option;
  }));
  el.channelSelect.value = state.channel;

  const status = channelStatus(state.channel);
  const anyAvailable = Boolean(state.model?.channels?.some((item) => item.available));
  el.channelStatus.textContent = status.available
    ? `${channelLabel(status)} · configured default is ${configured.toUpperCase()}.`
    : `${channelLabel(status)}. ${anyAvailable ? 'Choose an available channel.' : 'No channel is available; check server policy and hardware.'}`;
  el.channelStatus.classList.toggle('is-bad', !status.available);
  el.generateButton.disabled = !status.available || state.generating;
  el.modelSelect.disabled = state.generating;
  el.channelSelect.disabled = state.generating;
}

function refreshAlignmentAvailability() {
  const supported = ENGLISH_ALIGNMENT.has(el.languageSelect.value);
  el.forceAlignInput.disabled = !supported || state.generating;
  if (!supported) el.forceAlignInput.checked = false;
  el.forceAlignInput.parentElement.title = supported
    ? 'Generate English word timestamps with Wav2Vec2'
    : 'Word alignment currently supports English only';
}

function selectEngine(id) {
  const previous = state.model;
  state.engine = state.engines.find((engine) => engine.id === id);
  if (!state.engine) return;
  const models = state.models.filter((model) => model.engine === id);
  state.model = models.find((model) => model.model === previous?.model)
    || models.find((model) => model.default) || models[0] || null;
  if (previous?.engine !== id || state.model?.model !== previous?.model) {
    state.channel = state.model?.default_channel || 'cpu';
  }
  fillSelect(el.modelSelect, models.map((model) => model.model), state.model?.model);

  const freeform = Boolean(VOICE_INPUT_LABELS[state.engine.voice_input]);
  const style = state.engine.voice_input === 'style';
  el.voiceSelect.hidden = freeform;
  el.voiceDescription.hidden = !freeform;
  el.voiceField.htmlFor = freeform ? 'voice-description' : 'voice-select';
  el.voiceTitle.textContent = style ? 'Style' : 'Voice';
  el.voiceDescription.placeholder = style ? 'e.g. calm narration (optional)' : 'Describe a voice (optional)';
  fillSelect(el.voiceSelect, state.engine.voices, state.engine.default_voice);
  fillSelect(el.languageSelect, state.engine.languages, state.engine.default_language,
    (value) => LANGUAGE_NAMES[value] || value);

  el.engineName.textContent = state.engine.label;
  el.engineSummary.textContent = state.engine.summary;
  el.voiceCount.textContent = VOICE_INPUT_LABELS[state.engine.voice_input] || state.engine.voices.length;
  el.sampleRate.textContent = `${(state.engine.sample_rate / 1000).toFixed(state.engine.sample_rate % 1000 ? 2 : 0)} kHz`;
  el.loadedBadge.textContent = state.engine.loaded ? 'Warm' : 'Cold';
  el.loadedBadge.className = state.engine.loaded ? 'badge badge--ok' : 'badge';

  const adjustable = !FIXED_SPEED_ENGINES.has(id);
  el.speedInput.disabled = !adjustable;
  if (!adjustable) {
    el.speedInput.value = '1';
    el.speedValue.textContent = '1.00×';
  }
  el.formNote.textContent = !adjustable && style
    ? 'Use [whisper] or [excited] inline for direction. Speaking speed is fixed.'
    : !adjustable
      ? `${state.engine.label} uses a fixed speaking speed.`
      : 'Models load on first use, then stay warm.';

  hideError();
  refreshAlignmentAvailability();
  refreshChannel();
  renderEngineChips();
}

async function refreshCatalog(keepSelection = true) {
  const [engines, models] = await Promise.all([
    fetch('/v1/engines').then(readJson),
    fetch('/v1/models').then(readJson),
  ]);
  state.engines = engines.engines;
  state.models = models.models;
  selectEngine((keepSelection && state.engine?.id) || state.engines[0].id);
  refreshDiagnostics().catch(() => {});
}

function readJson(response) {
  if (!response.ok) throw new Error(`${response.url} failed with status ${response.status}.`);
  return response.json();
}

/* ---------- generation ---------- */

function readTiming(response, header) {
  const value = Number(response.headers.get(header));
  return Number.isFinite(value) && response.headers.has(header) ? value : null;
}

const milliseconds = (value) => `${Math.round(value).toLocaleString()} ms`;

async function generate(event) {
  event.preventDefault();
  if (state.generating) return;
  hideError();
  const script = el.speechText.value.trim();
  if (!script) return showError('Enter some text before generating speech.');
  const status = channelStatus(state.channel);
  if (!status.available) return showError(`${channelLabel(status)}. Choose an available channel to generate.`);

  const freeform = Boolean(VOICE_INPUT_LABELS[state.engine.voice_input]);
  const voice = freeform ? el.voiceDescription.value.trim() || null : el.voiceSelect.value;
  const slot = targetSlot();

  state.generating = true;
  el.lavaSrInput.disabled = true;
  el.forceAlignInput.disabled = true;
  refreshChannel();
  const started = performance.now();
  const ticker = setInterval(() => {
    el.generateLabel.textContent = `Generating into ${slot.tag} · ${((performance.now() - started) / 1000).toFixed(1)}s`;
  }, 100);

  try {
    const response = await fetch('/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engine: state.engine.id,
        model: state.model?.model,
        channel: state.channel,
        input: script,
        voice,
        language: el.languageSelect.value,
        speed: Number(el.speedInput.value),
        lava_sr: el.lavaSrInput.checked,
        force_align: el.forceAlignInput.checked,
      }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Synthesis failed with status ${response.status}.`);
    }
    slot.load(await readTake(response, script, voice));
    await refreshCatalog(true);
  } catch (error) {
    showError(error.message || 'Synthesis failed.');
  } finally {
    clearInterval(ticker);
    state.generating = false;
    el.lavaSrInput.disabled = false;
    el.generateLabel.textContent = 'Generate speech';
    refreshAlignmentAvailability();
    refreshChannel();
  }
}

async function readTake(response, script, voice) {
  const enhanced = response.headers.get('X-LavaSR-Applied') === 'true';
  const alignmentUrl = response.headers.get('X-Alignment-Url');
  const alignment = alignmentUrl ? await fetch(alignmentUrl).then(readJson) : null;
  const bytes = await response.arrayBuffer();

  const context = new AudioContext();
  const buffer = await context.decodeAudioData(bytes.slice(0));
  await context.close();

  const inference = readTiming(response, 'X-Inference-Time-Ms');
  const total = readTiming(response, 'X-Backend-Time-Ms');
  const model = response.headers.get('X-TTS-Model') || state.model?.model || '';
  const language = LANGUAGE_NAMES[el.languageSelect.value] || el.languageSelect.value;
  const speed = Number(el.speedInput.value);

  return {
    url: URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' })),
    peaks: peaksFrom(buffer.getChannelData(0)),
    duration: buffer.duration,
    script,
    alignment,
    title: `${state.engine.label} · ${model}`,
    subtitle: [voice, language, `${speed.toFixed(2)}×`].filter(Boolean).join(' · '),
    filename: `${state.engine.id}-${model}${enhanced ? '-lavasr' : ''}.wav`,
    badges: [enhanced && 'LavaSR 48 kHz', alignment && 'Word aligned'].filter(Boolean),
    meta: [
      ['Device', response.headers.get('X-TTS-Device') || '—'],
      ['Sample rate', `${(Number(response.headers.get('X-Sample-Rate')) / 1000).toFixed(1)} kHz`],
      ['Length', formatTime(buffer.duration)],
      ['Inference', inference === null ? '—' : milliseconds(inference)],
      ['Backend', total === null ? '—' : milliseconds(total)],
    ],
  };
}

/* ---------- status ---------- */

function showError(message) {
  el.errorPanel.textContent = message;
  el.errorPanel.hidden = false;
}

function hideError() {
  el.errorPanel.hidden = true;
  el.errorPanel.textContent = '';
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return '—';
  if (value < 1024 * 1024) return `${Math.round(value / 1024).toLocaleString()} KiB`;
  return `${(value / 1024 ** 3).toFixed(2)} GiB`;
}

async function refreshDiagnostics() {
  const [devices, resources] = await Promise.all([
    fetch('/v1/devices').then(readJson),
    fetch('/v1/resources').then(readJson),
  ]);
  el.diagnosticDevices.textContent = (devices.devices || [])
    .map((device) => `${device.id}${device.enabled ? '' : ' (disabled)'}`).join(', ') || 'None detected';
  const loaded = resources.models || [];
  el.diagnosticModels.textContent = loaded.length
    ? loaded.map((item) => `${item.engine}/${item.model} · ${item.device}`).join('; ')
    : 'None loaded';
  const vram = Object.entries(resources.vram_used_bytes || {});
  el.diagnosticVram.textContent = vram.length
    ? vram.map(([device, bytes]) => `${device}: ${formatBytes(bytes)}`).join(' · ')
    : 'No VRAM usage reported';
  const queue = resources.gpu_queue;
  el.diagnosticRequests.textContent = queue
    ? `GPU ${queue.running ?? 0} running · ${queue.waiting ?? 0} waiting${queue.capacity == null ? '' : ` / ${queue.capacity}`}`
    : loaded.map((item) => `${item.engine}/${item.model}: ${item.active_requests ?? 0}`).join(' · ') || 'No active requests';
}

async function checkHealth() {
  try {
    const health = await fetch('/health').then(readJson);
    el.healthDot.className = 'dot is-up';
    const channels = Array.isArray(health.channels) ? health.channels : [];
    el.healthLabel.textContent = channels.length
      ? channels.map((item) => `${item.id.toUpperCase()} ${item.available ? 'ready' : item.supported ? 'unavailable' : 'unsupported'}`).join(' · ')
      : 'API online';
  } catch {
    el.healthDot.className = 'dot is-down';
    el.healthLabel.textContent = 'API offline';
  }
}

/* ---------- theme ---------- */

function setTheme(mode) {
  localStorage.setItem('chorus-theme', mode);
  document.documentElement.dataset.theme = mode;
  document.documentElement.classList.toggle('dark',
    mode === 'dark' || (mode === 'system' && matchMedia('(prefers-color-scheme: dark)').matches));
  el.themeSwitch.querySelectorAll('button').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.themeChoice === mode));
  });
  slots.forEach((slot) => slot.draw());
}

function setTarget(target) {
  state.target = target;
  el.targetSwitch.querySelectorAll('button').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.target === String(target)));
  });
}

/* ---------- wiring ---------- */

el.composer.addEventListener('submit', generate);
el.speechText.addEventListener('input', () => {
  el.charCount.textContent = `${el.speechText.value.length.toLocaleString()} / 10,000`;
});
el.speedInput.addEventListener('input', () => {
  el.speedValue.textContent = `${Number(el.speedInput.value).toFixed(2)}×`;
});
el.voiceSelect.addEventListener('change', () => {
  const language = state.engine?.id === 'kokoro'
    ? el.voiceSelect.value.charAt(0)
    : state.engine?.id === 'pocket' ? VOICE_LANGUAGE_HINTS[el.voiceSelect.value] || 'english' : null;
  if (language && [...el.languageSelect.options].some((option) => option.value === language)) {
    el.languageSelect.value = language;
  }
  refreshAlignmentAvailability();
});
el.languageSelect.addEventListener('change', refreshAlignmentAvailability);
el.modelSelect.addEventListener('change', () => {
  state.model = state.models.find((model) => model.engine === state.engine.id && model.model === el.modelSelect.value) || state.model;
  state.channel = state.model?.default_channel || 'cpu';
  refreshChannel();
});
el.channelSelect.addEventListener('change', () => {
  state.channel = el.channelSelect.value;
  refreshChannel();
});

el.auditionSwitch.addEventListener('click', (event) => {
  const button = event.target.closest('[data-audition]');
  if (button) audition(Number(button.dataset.audition));
});
el.stopButton.addEventListener('click', () => slots.forEach((slot) => slot.pause()));
el.swapButton.addEventListener('click', swapSlots);
el.targetSwitch.addEventListener('click', (event) => {
  const button = event.target.closest('[data-target]');
  if (button) setTarget(button.dataset.target === 'auto' ? 'auto' : Number(button.dataset.target));
});
el.themeSwitch.addEventListener('click', (event) => {
  const button = event.target.closest('[data-theme-choice]');
  if (button) setTheme(button.dataset.themeChoice);
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if ((document.documentElement.dataset.theme || 'system') === 'system') setTheme('system');
});

document.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.target.closest('input, textarea, select, button, a, [contenteditable]')) return;
  const key = event.key.toLowerCase();
  if (key === 'a' || key === 'b') {
    event.preventDefault();
    audition(key === 'a' ? 0 : 1);
  } else if (event.key === ' ' && state.focus) {
    event.preventDefault();
    state.focus.toggle();
  }
});
window.addEventListener('resize', () => slots.forEach((slot) => slot.draw()));

setTheme(document.documentElement.dataset.theme || 'system');
setTarget('auto');
refreshCompare();
el.speechText.dispatchEvent(new Event('input'));
el.serverAddress.textContent = location.host;
checkHealth();
refreshCatalog(false).catch((error) => showError(error.message));
