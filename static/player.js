/* One comparison slot: an audio clip with a waveform, transport, and metadata.
 *
 * A slot knows nothing about engines. The composer hands it a plain take object
 * so both slots stay interchangeable and the A/B controls stay trivial.
 */

const BAR_COUNT = 110;
const BAR_GAP = 2;

export function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '0:00';
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
}

/** Reduce a decoded channel to per-bar peaks, normalized against the loudest bar. */
export function peaksFrom(samples, count = BAR_COUNT) {
  const size = Math.max(1, Math.floor(samples.length / count));
  const bars = [];
  let loudest = 0;
  for (let index = 0; index < count; index += 1) {
    const end = Math.min((index + 1) * size, samples.length);
    let peak = 0;
    for (let at = index * size; at < end; at += 1) peak = Math.max(peak, Math.abs(samples[at]));
    bars.push(peak);
    loudest = Math.max(loudest, peak);
  }
  return bars.map((peak) => Math.max(0.06, peak / (loudest || 1)));
}

export class Slot {
  /** @param {(slot: Slot, event: string) => void} notify */
  constructor(node, notify) {
    this.node = node;
    this.take = null;
    this.notify = notify;
    this.audio = new Audio();
    this.audio.preload = 'metadata';
    this.el = Object.fromEntries(
      ['tag', 'title', 'subtitle', 'badges', 'meta', 'wave', 'play', 'current', 'total', 'download', 'clear']
        .map((name) => [name, node.querySelector(`[data-${name}]`)]),
    );
    this.context = this.el.wave.getContext('2d');

    this.el.play.addEventListener('click', () => this.toggle());
    this.el.clear.addEventListener('click', () => this.clear());
    this.el.wave.addEventListener('click', (event) => {
      const bounds = this.el.wave.getBoundingClientRect();
      this.seek(((event.clientX - bounds.left) / bounds.width) * this.duration);
    });
    this.el.wave.addEventListener('keydown', (event) => this.onWaveKey(event));
    this.audio.addEventListener('play', () => this.onPlaybackChange('play'));
    this.audio.addEventListener('pause', () => this.onPlaybackChange('pause'));
    this.audio.addEventListener('ended', () => this.onPlaybackChange('pause'));
    this.audio.addEventListener('timeupdate', () => this.renderTime());
  }

  get tag() { return this.el.tag.textContent; }
  get isEmpty() { return this.take === null; }
  get duration() { return this.take?.duration || 0; }
  get time() { return this.audio.currentTime; }
  get playing() { return !this.audio.paused && !this.audio.ended; }

  setTag(tag) {
    this.el.tag.textContent = tag;
    this.el.play.setAttribute('aria-label', `Play take ${tag}`);
  }

  load(take) {
    if (this.take) URL.revokeObjectURL(this.take.url);
    this.take = take;
    this.audio.src = take.url;
    this.node.classList.remove('is-empty');
    this.el.title.textContent = take.title;
    this.el.subtitle.textContent = take.subtitle;
    this.el.download.href = take.url;
    this.el.download.download = take.filename;
    this.el.download.removeAttribute('aria-disabled');
    this.el.clear.disabled = false;
    this.el.play.disabled = false;
    this.el.total.textContent = formatTime(take.duration);
    this.el.wave.setAttribute('aria-valuemax', take.duration.toFixed(1));

    this.el.badges.replaceChildren(...take.badges.map((text) => {
      const badge = document.createElement('span');
      badge.className = 'badge badge--on';
      badge.textContent = text;
      return badge;
    }));
    this.el.meta.replaceChildren(...take.meta.map(([label, value]) => {
      const row = document.createElement('div');
      row.append(
        Object.assign(document.createElement('dt'), { textContent: label }),
        Object.assign(document.createElement('dd'), { textContent: value }),
      );
      return row;
    }));
    this.renderTime();
    this.notify(this, 'load');
  }

  clear() {
    if (this.isEmpty) return;
    this.audio.pause();
    this.audio.removeAttribute('src');
    URL.revokeObjectURL(this.take.url);
    this.take = null;
    this.node.classList.add('is-empty');
    this.el.title.textContent = 'Empty';
    this.el.subtitle.textContent = 'Generate speech to fill this slot.';
    this.el.download.removeAttribute('href');
    this.el.download.setAttribute('aria-disabled', 'true');
    this.el.clear.disabled = true;
    this.el.play.disabled = true;
    this.el.badges.replaceChildren();
    this.el.meta.replaceChildren();
    this.el.current.textContent = '0:00';
    this.el.total.textContent = '0:00';
    this.draw();
    this.notify(this, 'clear');
  }

  play(at = null) {
    if (this.isEmpty) return;
    if (at !== null) this.audio.currentTime = Math.min(at, Math.max(0, this.duration - 0.01));
    this.audio.play().catch(() => {});
  }

  pause() { this.audio.pause(); }

  toggle() { this.playing ? this.pause() : this.play(); }

  seek(seconds) {
    if (this.isEmpty) return;
    this.audio.currentTime = Math.min(Math.max(0, seconds), this.duration);
    this.renderTime();
  }

  onWaveKey(event) {
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault();
      this.toggle();
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      this.seek(this.time + (event.key === 'ArrowRight' ? 5 : -5));
    }
  }

  onPlaybackChange(event) {
    this.node.classList.toggle('is-playing', event === 'play');
    this.notify(this, event);
  }

  renderTime() {
    this.el.current.textContent = formatTime(this.time);
    this.el.wave.setAttribute('aria-valuenow', this.time.toFixed(1));
    this.draw();
  }

  draw() {
    const canvas = this.el.wave;
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * ratio);
    canvas.height = Math.round(bounds.height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.context.clearRect(0, 0, bounds.width, bounds.height);
    const bars = this.take?.peaks;
    if (!bars?.length) return;

    const styles = getComputedStyle(document.documentElement);
    const idle = styles.getPropertyValue('--wave').trim();
    const active = styles.getPropertyValue('--wave-on').trim();
    const progress = this.duration ? this.time / this.duration : 0;
    const width = Math.max(1.5, (bounds.width - BAR_GAP * (bars.length - 1)) / bars.length);
    const middle = bounds.height / 2;

    bars.forEach((level, index) => {
      const height = Math.max(3, level * (bounds.height - 6));
      this.context.fillStyle = index / bars.length <= progress ? active : idle;
      this.context.beginPath();
      this.context.roundRect(index * (width + BAR_GAP), middle - height / 2, width, height, width / 2);
      this.context.fill();
    });
  }
}
