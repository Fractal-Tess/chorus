const state = {
  engines: [],
  engine: null,
  audioBuffer: null,
  waveform: [],
  audioUrl: null,
};

const elements = {
  engineGrid: document.querySelector('#engine-grid'),
  form: document.querySelector('#speech-form'),
  text: document.querySelector('#speech-text'),
  charCount: document.querySelector('#char-count'),
  voice: document.querySelector('#voice-select'),
  voiceDescription: document.querySelector('#voice-description'),
  voiceLabel: document.querySelector('#voice-label'),
  voiceChevron: document.querySelector('#voice-chevron'),
  narrationBadge: document.querySelector('#narration-badge'),
  language: document.querySelector('#language-select'),
  speed: document.querySelector('#speed-input'),
  speedValue: document.querySelector('#speed-value'),
  lavaSr: document.querySelector('#lava-sr-input'),
  forceAlign: document.querySelector('#force-align-input'),
  forceAlignLabel: document.querySelector('#force-align-label'),
  formNote: document.querySelector('#form-note'),
  generateButton: document.querySelector('#generate-button'),
  generateIcon: document.querySelector('#generate-icon'),
  generateLabel: document.querySelector('#generate-label'),
  resultPanel: document.querySelector('#result-panel'),
  resultTitle: document.querySelector('#result-title'),
  resultEnhancement: document.querySelector('#result-enhancement'),
  resultAlignment: document.querySelector('#result-alignment'),
  resultTimings: document.querySelector('#result-timings'),
  alignmentPanel: document.querySelector('#alignment-panel'),
  alignmentWords: document.querySelector('#alignment-words'),
  downloadLink: document.querySelector('#download-link'),
  audio: document.querySelector('#audio-player'),
  playButton: document.querySelector('#play-button'),
  playIcon: document.querySelector('#play-icon'),
  waveform: document.querySelector('#waveform'),
  currentTime: document.querySelector('#current-time'),
  totalTime: document.querySelector('#total-time'),
  engineName: document.querySelector('#engine-name'),
  engineSummary: document.querySelector('#engine-summary'),
  voiceCount: document.querySelector('#voice-count'),
  sampleRate: document.querySelector('#sample-rate'),
  loadedBadge: document.querySelector('#loaded-badge'),
  errorPanel: document.querySelector('#error-panel'),
  healthDot: document.querySelector('#health-dot'),
  healthLabel: document.querySelector('#health-label'),
};

const languageNames = {
  a: 'American English', b: 'British English', e: 'Spanish', f: 'French', h: 'Hindi', i: 'Italian', j: 'Japanese', p: 'Portuguese', z: 'Mandarin',
  en: 'English', 'en-US': 'English (US)', english: 'English', french_24l: 'French', spanish_24l: 'Spanish', german_24l: 'German', italian_24l: 'Italian', portuguese_24l: 'Portuguese',
  ko: 'Korean', ja: 'Japanese', ar: 'Arabic', bg: 'Bulgarian', cs: 'Czech', da: 'Danish', de: 'German', el: 'Greek', es: 'Spanish', et: 'Estonian', fi: 'Finnish', fr: 'French', hi: 'Hindi', hr: 'Croatian', hu: 'Hungarian', id: 'Indonesian', it: 'Italian', lt: 'Lithuanian', lv: 'Latvian', nl: 'Dutch', pl: 'Polish', pt: 'Portuguese', ro: 'Romanian', ru: 'Russian', sk: 'Slovak', sl: 'Slovenian', sv: 'Swedish', tr: 'Turkish', uk: 'Ukrainian', vi: 'Vietnamese', na: 'Language agnostic',
};

const englishAlignmentLanguages = new Set(['a', 'b', 'en', 'en-US', 'en-us', 'en-gb', 'english']);

const narrationFavorites = {
  pocket: ['peter_yearsley'],
  kokoro: ['af_heart', 'af_bella', 'bf_emma'],
  supertonic: ['M5', 'F5', 'M2', 'M3', 'F3'],
};

function escapeHtml(value) {
  const node = document.createElement('div');
  node.textContent = value;
  return node.innerHTML;
}

function setTheme(mode) {
  localStorage.setItem('mini-tts-theme', mode);
  document.documentElement.dataset.theme = mode;
  const dark = mode === 'dark' || (mode === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', dark);
  updateThemeButtons();
  requestAnimationFrame(drawWaveform);
}

function updateThemeButtons() {
  const active = document.documentElement.dataset.theme || 'system';
  document.querySelectorAll('.theme-button').forEach((button) => {
    const selected = button.dataset.themeChoice === active;
    button.classList.toggle('bg-slate-100', selected);
    button.classList.toggle('text-slate-950', selected);
    button.classList.toggle('dark:bg-slate-700', selected);
    button.classList.toggle('dark:text-white', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
}

function renderEngineCards() {
  elements.engineGrid.innerHTML = state.engines.map((engine) => {
    const active = engine.id === state.engine?.id;
    return `
      <button type="button" data-engine="${engine.id}" aria-pressed="${active}" class="engine-card group min-h-28 rounded-2xl border p-4 text-left transition ${active ? 'border-slate-950 bg-slate-950 text-white shadow-lg shadow-slate-900/15 dark:border-white dark:bg-white dark:text-slate-950' : 'border-slate-200/80 bg-white/70 hover:-translate-y-0.5 hover:border-slate-300 hover:bg-white dark:border-slate-800 dark:bg-slate-900/65 dark:hover:border-slate-700 dark:hover:bg-slate-900'}">
        <span class="mb-6 flex items-center justify-between gap-2">
          <span class="font-mono text-[9px] font-bold uppercase tracking-[0.15em] ${active ? 'text-slate-400' : 'text-slate-400'}">${engine.loaded ? 'Warm' : 'Cold'}</span>
          <span class="size-1.5 rounded-full ${engine.loaded ? 'bg-emerald-500' : 'bg-slate-300 dark:bg-slate-700'}"></span>
        </span>
        <span class="block text-sm font-black tracking-[-0.02em]">${escapeHtml(engine.label)}</span>
        <span class="mt-1 block text-[10px] ${active ? 'text-slate-400' : 'text-slate-500 dark:text-slate-400'}">${engine.voice_input === 'description' ? 'Voice design' : `${engine.voices.length} ${engine.voices.length === 1 ? 'voice' : 'voices'}`}</span>
      </button>`;
  }).join('');

  document.querySelectorAll('.engine-card').forEach((button) => {
    button.addEventListener('click', () => selectEngine(button.dataset.engine));
  });
}

function fillSelect(select, values, selected, labeler = (value) => value) {
  select.innerHTML = values.map((value) => `<option value="${escapeHtml(value)}" ${value === selected ? 'selected' : ''}>${escapeHtml(labeler(value))}</option>`).join('');
}

function fillVoiceSelect(engine) {
  const favorites = narrationFavorites[engine.id] || [];
  if (!favorites.length) {
    fillSelect(elements.voice, engine.voices, engine.default_voice);
    return;
  }
  const favoriteSet = new Set(favorites);
  const option = (voice) => `<option value="${escapeHtml(voice)}" ${voice === engine.default_voice ? 'selected' : ''}>${escapeHtml(voice)}</option>`;
  elements.voice.innerHTML = `
    <optgroup label="Narration favorites">${favorites.filter((voice) => engine.voices.includes(voice)).map(option).join('')}</optgroup>
    <optgroup label="All voices">${engine.voices.filter((voice) => !favoriteSet.has(voice)).map(option).join('')}</optgroup>`;
}

function updateNarrationBadge() {
  const favorites = narrationFavorites[state.engine?.id] || [];
  elements.narrationBadge.classList.toggle('hidden', !favorites.includes(elements.voice.value));
}

function updateAlignmentAvailability() {
  const supported = englishAlignmentLanguages.has(elements.language.value);
  elements.forceAlign.disabled = !supported;
  if (!supported) elements.forceAlign.checked = false;
  elements.forceAlignLabel.classList.toggle('cursor-pointer', supported);
  elements.forceAlignLabel.classList.toggle('cursor-not-allowed', !supported);
  elements.forceAlignLabel.classList.toggle('opacity-50', !supported);
  elements.forceAlignLabel.title = supported
    ? 'Generate English word timestamps with Wav2Vec2'
    : 'Word alignment currently supports English only';
}

function selectEngine(id) {
  state.engine = state.engines.find((engine) => engine.id === id);
  if (!state.engine) return;

  const voiceDesign = state.engine.voice_input === 'description';
  elements.voice.hidden = voiceDesign;
  elements.voiceChevron.classList.toggle('hidden', voiceDesign);
  elements.voiceDescription.hidden = !voiceDesign;
  elements.voiceLabel.htmlFor = voiceDesign ? 'voice-description' : 'voice-select';
  fillVoiceSelect(state.engine);
  fillSelect(elements.language, state.engine.languages, state.engine.default_language, (value) => languageNames[value] || value);
  updateNarrationBadge();
  updateAlignmentAvailability();
  elements.engineName.textContent = state.engine.label;
  elements.engineSummary.textContent = state.engine.summary;
  elements.voiceCount.textContent = voiceDesign ? 'Voice design' : state.engine.voices.length;
  elements.sampleRate.textContent = `${(state.engine.sample_rate / 1000).toFixed(state.engine.sample_rate % 1000 ? 2 : 0)} kHz`;
  elements.loadedBadge.textContent = state.engine.loaded ? 'Warm' : 'Cold';
  elements.loadedBadge.className = state.engine.loaded
    ? 'rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400'
    : 'rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:bg-slate-800 dark:text-slate-400';

  const supportsSpeed = !['pocket', 'breeze'].includes(id);
  elements.speed.disabled = !supportsSpeed;
  if (!supportsSpeed) {
    elements.speed.value = '1';
    elements.speedValue.textContent = '1.00×';
    elements.formNote.textContent = `${state.engine.label} uses a fixed speaking speed.`;
  } else {
    elements.formNote.textContent = state.engine.loaded ? 'Model is warm and ready.' : 'First render loads the model; later renders are faster.';
  }
  hideError();
  renderEngineCards();
}

function inferLanguage(engine, voice) {
  if (engine.id === 'kokoro') return voice.charAt(0);
  if (engine.id === 'pocket') {
    const map = { giovanni: 'italian_24l', lola: 'spanish_24l', juergen: 'german_24l', rafael: 'portuguese_24l', estelle: 'french_24l' };
    return map[voice] || 'english';
  }
  return null;
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '0:00';
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
}

function readTiming(response, header) {
  const value = response.headers.get(header);
  if (value === null) return null;
  const milliseconds = Number(value);
  return Number.isFinite(milliseconds) ? milliseconds : null;
}

function formatMilliseconds(milliseconds) {
  return `${Math.round(milliseconds).toLocaleString()} ms`;
}

function renderAlignment(alignment) {
  elements.alignmentWords.replaceChildren();
  const words = alignment?.words || [];
  elements.alignmentPanel.classList.toggle('hidden', words.length === 0);
  elements.resultAlignment.classList.toggle('hidden', words.length === 0);
  for (const word of words) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-left transition hover:border-orange-300 hover:bg-orange-50 dark:border-slate-700 dark:bg-slate-800 dark:hover:border-orange-700 dark:hover:bg-slate-800';
    button.title = `Confidence ${(word.score * 100).toFixed(1)}%`;

    const label = document.createElement('span');
    label.className = 'block text-xs font-semibold text-slate-700 dark:text-slate-200';
    label.textContent = word.word;
    const timing = document.createElement('span');
    timing.className = 'block font-mono text-[9px] text-slate-400';
    timing.textContent = `${word.start_ms}–${word.end_ms} ms`;
    button.append(label, timing);
    button.addEventListener('click', () => {
      elements.audio.currentTime = word.start_ms / 1_000;
      drawWaveform();
    });
    elements.alignmentWords.append(button);
  }
}

function setPlaying(playing) {
  elements.playButton.setAttribute('aria-label', playing ? 'Pause audio' : 'Play audio');
  elements.playIcon.classList.toggle('translate-x-px', !playing);
  elements.playIcon.innerHTML = playing
    ? '<path d="M7 5h4v14H7V5Zm6 0h4v14h-4V5Z"/>'
    : '<path d="m8 5 11 7-11 7V5Z"/>';
}

function buildWaveform(channelData, count = 180) {
  const size = Math.max(1, Math.floor(channelData.length / count));
  const values = [];
  let peak = 0;
  for (let index = 0; index < count; index += 1) {
    let maximum = 0;
    const start = index * size;
    const end = Math.min(start + size, channelData.length);
    for (let sample = start; sample < end; sample += 1) maximum = Math.max(maximum, Math.abs(channelData[sample]));
    values.push(maximum);
    peak = Math.max(peak, maximum);
  }
  return values.map((value) => Math.max(0.06, value / (peak || 1)));
}

function drawWaveform() {
  if (!state.waveform.length) return;
  const canvas = elements.waveform;
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(bounds.width * ratio));
  canvas.height = Math.max(1, Math.round(bounds.height * ratio));
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.clearRect(0, 0, bounds.width, bounds.height);

  const styles = getComputedStyle(document.documentElement);
  const idle = styles.getPropertyValue('--wave-idle').trim();
  const active = styles.getPropertyValue('--wave-active').trim();
  const progress = elements.audio.duration ? elements.audio.currentTime / elements.audio.duration : 0;
  const gap = 2;
  const barWidth = Math.max(1.5, (bounds.width - gap * (state.waveform.length - 1)) / state.waveform.length);
  const center = bounds.height / 2;

  state.waveform.forEach((level, index) => {
    const height = Math.max(4, level * (bounds.height - 8));
    const x = index * (barWidth + gap);
    context.fillStyle = index / state.waveform.length <= progress ? active : idle;
    context.beginPath();
    context.roundRect(x, center - height / 2, barWidth, height, barWidth / 2);
    context.fill();
  });
}

function showError(message) {
  elements.errorPanel.textContent = message;
  elements.errorPanel.classList.remove('hidden');
}

function hideError() {
  elements.errorPanel.classList.add('hidden');
  elements.errorPanel.textContent = '';
}

async function refreshEngines(keepSelection = true) {
  const response = await fetch('/v1/engines');
  if (!response.ok) throw new Error('Could not load engine catalog.');
  const data = await response.json();
  const selectedId = keepSelection ? state.engine?.id : null;
  state.engines = data.engines;
  selectEngine(selectedId || state.engines[0].id);
}

async function generateSpeech(event) {
  event.preventDefault();
  hideError();
  const text = elements.text.value.trim();
  if (!text) {
    showError('Enter some text before generating speech.');
    return;
  }
  const useLavaSr = elements.lavaSr.checked;
  const useForceAlign = elements.forceAlign.checked;
  const voice = state.engine.voice_input === 'description'
    ? elements.voiceDescription.value.trim() || null
    : elements.voice.value;

  elements.generateButton.disabled = true;
  elements.lavaSr.disabled = true;
  elements.forceAlign.disabled = true;
  elements.generateIcon.classList.add('animate-spin');
  elements.generateIcon.innerHTML = '<path d="M12 3a9 9 0 1 0 9 9" stroke-linecap="round"/>';
  const started = performance.now();
  const timer = setInterval(() => {
    elements.generateLabel.textContent = `Generating ${((performance.now() - started) / 1000).toFixed(1)}s`;
  }, 100);

  try {
    const response = await fetch('/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engine: state.engine.id,
        input: text,
        voice,
        language: elements.language.value,
        speed: Number(elements.speed.value),
        lava_sr: useLavaSr,
        force_align: useForceAlign,
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Synthesis failed with status ${response.status}.`);
    }
    const enhanced = response.headers.get('X-LavaSR-Applied') === 'true';
    const alignmentUrl = response.headers.get('X-Alignment-Url');
    const timings = {
      queue: readTiming(response, 'X-Queue-Time-Ms'),
      inference: readTiming(response, 'X-Inference-Time-Ms'),
      lavaSr: readTiming(response, 'X-LavaSR-Time-Ms'),
      alignment: readTiming(response, 'X-Alignment-Time-Ms'),
      total: readTiming(response, 'X-Backend-Time-Ms'),
    };
    let alignment = null;
    if (alignmentUrl) {
      const alignmentResponse = await fetch(alignmentUrl);
      if (!alignmentResponse.ok) throw new Error('Generated word alignment could not be retrieved.');
      alignment = await alignmentResponse.json();
    }

    const bytes = await response.arrayBuffer();
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
    elements.audio.src = state.audioUrl;
    elements.downloadLink.href = state.audioUrl;
    elements.downloadLink.download = `${state.engine.id}${enhanced ? '-lavasr' : ''}${alignment ? '-aligned' : ''}.wav`;

    const audioContext = new AudioContext();
    state.audioBuffer = await audioContext.decodeAudioData(bytes.slice(0));
    state.waveform = buildWaveform(state.audioBuffer.getChannelData(0));
    await audioContext.close();

    elements.resultTitle.textContent = `${state.engine.label}${voice ? ` · ${voice}` : ''}`;
    elements.resultEnhancement.classList.toggle('hidden', !enhanced);
    renderAlignment(alignment);
    const timingParts = [];
    if (timings.inference !== null) timingParts.push(`Inference: ${formatMilliseconds(timings.inference)}`);
    if (enhanced && timings.lavaSr !== null) timingParts.push(`LavaSR: ${formatMilliseconds(timings.lavaSr)}`);
    if (alignment && timings.alignment !== null) timingParts.push(`Alignment: ${formatMilliseconds(timings.alignment)}`);
    if (timings.queue !== null && timings.queue >= 1) timingParts.push(`Queue: ${formatMilliseconds(timings.queue)}`);
    if (timings.total !== null) timingParts.push(`Backend total: ${formatMilliseconds(timings.total)}`);
    elements.resultTimings.textContent = timingParts.join(' · ');
    elements.totalTime.textContent = formatTime(state.audioBuffer.duration);
    elements.currentTime.textContent = '0:00';
    elements.resultPanel.classList.remove('hidden');
    setPlaying(false);
    requestAnimationFrame(drawWaveform);
    await refreshEngines(true);
  } catch (error) {
    showError(error.message || 'Synthesis failed.');
  } finally {
    clearInterval(timer);
    elements.generateButton.disabled = false;
    elements.lavaSr.disabled = false;
    updateAlignmentAvailability();
    elements.generateIcon.classList.remove('animate-spin');
    elements.generateIcon.innerHTML = '<path d="M5 12h14M13 6l6 6-6 6" stroke-linecap="round" stroke-linejoin="round"/>';
    elements.generateLabel.textContent = 'Generate speech';
  }
}

async function checkHealth() {
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error();
    const health = await response.json();
    elements.healthDot.className = 'size-2 rounded-full bg-emerald-500 shadow-[0_0_0_3px_rgb(16_185_129_/_0.12)]';
    elements.healthLabel.textContent = `${(health.allowed_devices || [health.device]).join(', ')} online`;
  } catch {
    elements.healthDot.className = 'size-2 rounded-full bg-red-500';
    elements.healthLabel.textContent = 'API offline';
  }
}

function seekFromEvent(event) {
  if (!elements.audio.duration) return;
  const bounds = elements.waveform.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
  elements.audio.currentTime = ratio * elements.audio.duration;
  drawWaveform();
}

document.querySelectorAll('.theme-button').forEach((button) => button.addEventListener('click', () => setTheme(button.dataset.themeChoice)));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if ((document.documentElement.dataset.theme || 'system') === 'system') setTheme('system');
});

elements.text.addEventListener('input', () => { elements.charCount.textContent = `${elements.text.value.length.toLocaleString()} / 10,000`; });
elements.speed.addEventListener('input', () => { elements.speedValue.textContent = `${Number(elements.speed.value).toFixed(2)}×`; });
elements.voice.addEventListener('change', () => {
  const language = inferLanguage(state.engine, elements.voice.value);
  if (language && [...elements.language.options].some((option) => option.value === language)) elements.language.value = language;
  updateNarrationBadge();
  updateAlignmentAvailability();
});
elements.language.addEventListener('change', updateAlignmentAvailability);
elements.form.addEventListener('submit', generateSpeech);
elements.playButton.addEventListener('click', () => elements.audio.paused ? elements.audio.play() : elements.audio.pause());
elements.waveform.addEventListener('click', seekFromEvent);
elements.waveform.addEventListener('keydown', (event) => {
  if (event.key === ' ' || event.key === 'Enter') {
    event.preventDefault();
    elements.audio.paused ? elements.audio.play() : elements.audio.pause();
  } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    elements.audio.currentTime = Math.min(elements.audio.duration || 0, Math.max(0, elements.audio.currentTime + (event.key === 'ArrowRight' ? 5 : -5)));
  }
});
elements.audio.addEventListener('play', () => setPlaying(true));
elements.audio.addEventListener('pause', () => setPlaying(false));
elements.audio.addEventListener('ended', () => setPlaying(false));
elements.audio.addEventListener('timeupdate', () => {
  elements.currentTime.textContent = formatTime(elements.audio.currentTime);
  elements.waveform.setAttribute('aria-valuenow', String(Math.round(elements.audio.currentTime)));
  drawWaveform();
});
window.addEventListener('resize', drawWaveform);

updateThemeButtons();
document.querySelector('#server-address').textContent = location.host;
elements.text.dispatchEvent(new Event('input'));
checkHealth();
refreshEngines(false).catch((error) => showError(error.message));
