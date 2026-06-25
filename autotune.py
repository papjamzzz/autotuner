"""
Core audio processing engine.
Pitch detection: YIN algorithm in pure numpy (no aubio required).
Pitch shifting: pyrubberband (Rubber Band Library), scipy fallback.
"""
import threading
import numpy as np

try:
    import pyrubberband as pyrb
    HAS_RUBBERBAND = True
except Exception:
    HAS_RUBBERBAND = False
    from scipy.signal import resample as scipy_resample

SAMPLE_RATE = 44100
CHUNK       = 4096   # ~93ms per block — enough for reliable low-pitch YIN
HOP         = 512

NOTE_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALES = {
    'chromatic':  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'major':      [0, 2, 4, 5, 7, 9, 11],
    'minor':      [0, 2, 3, 5, 7, 8, 10],
    'pentatonic': [0, 2, 4, 7, 9],
    'blues':      [0, 3, 5, 6, 7, 10],
}

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ── YIN pitch detection ───────────────────────────────────────────────────────

YIN_THRESHOLD = 0.15     # lower = stricter; 0.10–0.20 works well
FREQ_MIN      = 80.0     # Hz — lowest detectable (below human chest voice)
FREQ_MAX      = 1100.0   # Hz — highest detectable (above soprano high C)

def _yin_pitch(samples: np.ndarray, sr: int = SAMPLE_RATE):
    """
    Returns (freq_hz, confidence 0-1).
    confidence = 1 - d'[tau_best], so >0.85 is reliable.
    """
    N    = len(samples)
    half = N // 2

    tau_min = max(2, int(sr / FREQ_MAX))
    tau_max = min(half - 1, int(sr / FREQ_MIN))

    if tau_max <= tau_min:
        return 0.0, 0.0

    # Difference function via autocorrelation (FFT-based, O(N log N))
    x     = samples - samples.mean()
    pad   = np.zeros(N, dtype=np.float32)
    xpad  = np.concatenate([x, pad])
    X     = np.fft.rfft(xpad)
    acf   = np.fft.irfft(X * np.conj(X))[:half].real

    # d[tau] = 2*(r[0] - r[tau])  where r = autocorrelation
    r0  = acf[0]
    d   = np.zeros(half, dtype=np.float32)
    d[1:] = 2 * (r0 - acf[1:half])

    # Cumulative mean normalised difference (CMNDF)
    cmndf    = np.ones(half, dtype=np.float32)
    cum_sum  = 0.0
    for tau in range(1, half):
        cum_sum      += d[tau]
        cmndf[tau]    = d[tau] * tau / cum_sum if cum_sum > 0 else 1.0

    # Find first local minimum below threshold in vocal range
    best_tau   = -1
    best_val   = 1.0
    search     = cmndf[tau_min:tau_max + 1]

    # Prefer the absolute minimum below threshold
    for i, v in enumerate(search):
        tau = tau_min + i
        if v < YIN_THRESHOLD:
            # take first minimum (check it's a local min)
            if i > 0 and i < len(search) - 1 and v <= search[i - 1] and v <= search[i + 1]:
                best_tau = tau
                best_val = v
                break

    if best_tau == -1:
        # No reliable minimum found — return global minimum in range
        idx       = int(np.argmin(search))
        best_tau  = tau_min + idx
        best_val  = search[idx]
        if best_val > 0.4:            # not confident enough
            return 0.0, max(0.0, 1.0 - best_val)

    # Parabolic interpolation for sub-sample accuracy
    if 0 < best_tau < half - 1:
        v0, v1, v2 = cmndf[best_tau - 1], cmndf[best_tau], cmndf[best_tau + 1]
        denom = v0 - 2 * v1 + v2
        if denom != 0:
            delta     = 0.5 * (v0 - v2) / denom
            best_tau  = best_tau + delta

    freq       = float(sr / best_tau) if best_tau > 0 else 0.0
    confidence = max(0.0, 1.0 - best_val)
    return freq, confidence


# ── Tuner engine ─────────────────────────────────────────────────────────────

class AutoTuner:
    def __init__(self):
        self._lock = threading.Lock()

        # Settings
        self.active = False
        self.key    = 0       # 0-11 (C=0)
        self.scale  = 'major'
        self.amount = 1.0     # 0.0-1.0 correction strength
        self.speed  = 0.4     # 0.0-1.0 ramp speed (1 = instant snap)

        # UI-visible state
        self.freq_detected  = 0.0
        self.note_detected  = ''
        self.note_target    = ''
        self.cents_raw      = 0.0
        self.cents_applied  = 0.0
        self.confidence     = 0.0
        self.has_signal     = False

        # Internal
        self._smooth_shift  = 0.0
        self._buf           = np.zeros(CHUNK, dtype=np.float32)
        self._buf_pos       = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def get_status(self):
        with self._lock:
            return {
                'active':        self.active,
                'freq':          round(self.freq_detected, 1),
                'note_detected': self.note_detected,
                'note_target':   self.note_target,
                'cents_raw':     round(self.cents_raw, 1),
                'cents_applied': round(self.cents_applied, 1),
                'confidence':    round(self.confidence, 2),
                'has_signal':    self.has_signal,
                'key':           self.key,
                'scale':         self.scale,
                'amount':        self.amount,
                'speed':         self.speed,
            }

    # ── Audio processing ──────────────────────────────────────────────────────

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Called from the sounddevice callback every CHUNK samples.
        Returns pitch-corrected audio of the same length.
        """
        with self._lock:
            active = self.active

        if not active:
            return samples

        freq, conf = _yin_pitch(samples)

        vocal   = FREQ_MIN < freq < FREQ_MAX
        good    = conf > 0.78

        with self._lock:
            self.confidence    = conf
            self.freq_detected = freq if vocal else 0.0
            self.has_signal    = vocal and good

        if not (vocal and good):
            self._smooth_shift *= 0.85   # decay toward zero
            with self._lock:
                self.note_detected = ''
                self.note_target   = ''
                self.cents_raw     = 0.0
                self.cents_applied = self._smooth_shift
            if abs(self._smooth_shift) < 1.5:
                return samples
            return self._shift(samples, self._smooth_shift / 100.0)

        midi         = 69 + 12 * np.log2(freq / 440.0)
        note_in_oct  = midi % 12
        det_name     = NOTE_NAMES[int(round(midi)) % 12]

        with self._lock:
            key   = self.key
            scale = self.scale
            amt   = self.amount
            spd   = self.speed

        intervals  = SCALES.get(scale, SCALES['major'])
        best_dist  = float('inf')

        for interval in intervals:
            scale_note = (key + interval) % 12
            dist = (note_in_oct - scale_note) % 12
            if dist > 6:
                dist -= 12
            if abs(dist) < abs(best_dist):
                best_dist = dist

        cents_raw    = best_dist * 100          # how far off the nearest note
        cents_target = -cents_raw * amt         # shift needed (flip sign)
        tgt_midi     = int(round(midi - best_dist))
        tgt_name     = NOTE_NAMES[tgt_midi % 12]

        # Smooth shift (retune speed)
        self._smooth_shift = self._smooth_shift * (1 - spd) + cents_target * spd

        with self._lock:
            self.note_detected = det_name
            self.note_target   = tgt_name
            self.cents_raw     = float(cents_raw)
            self.cents_applied = float(self._smooth_shift)

        if abs(self._smooth_shift) < 2.0:
            return samples

        return self._shift(samples, self._smooth_shift / 100.0)

    def _shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        n = len(audio)
        if HAS_RUBBERBAND:
            try:
                out = pyrb.pitch_shift(audio, SAMPLE_RATE, semitones)
                if len(out) >= n:
                    return out[:n].astype(np.float32)
                return np.pad(out, (0, n - len(out))).astype(np.float32)
            except Exception:
                pass
        # Fallback: resampling (good for small shifts ≤ ±2 semitones)
        from scipy.signal import resample as scipy_resample
        ratio     = 2 ** (semitones / 12)
        n_out     = max(1, int(n / ratio))
        resampled = scipy_resample(audio, n_out)
        if len(resampled) >= n:
            return resampled[:n].astype(np.float32)
        return np.pad(resampled, (0, n - len(resampled))).astype(np.float32)
