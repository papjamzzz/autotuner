import os, json, time, threading
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from autotune import AutoTuner, SAMPLE_RATE, CHUNK, KEY_NAMES, SCALES

load_dotenv()

app    = Flask(__name__)
tuner  = AutoTuner()
stream = None
stream_lock = threading.Lock()


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'no-referrer')
    return resp


# ── Audio stream ──────────────────────────────────────────────────────────────

def _audio_callback(indata, outdata, frames, time_info, status):
    samples = indata[:, 0].copy()
    try:
        out = tuner.process(samples)
        if out.shape[0] != frames:
            # Defensive: a processing bug returning the wrong length would
            # otherwise raise inside PortAudio's callback and silently kill
            # the stream (UI would keep showing "ON" with no audio).
            out = samples
    except Exception:
        # Never let a pitch-detection/shift error abort the audio stream —
        # pass the dry signal through instead of dropping audio or crashing.
        out = samples
    outdata[:] = out.reshape(-1, 1)


def start_stream(in_dev=None, out_dev=None):
    global stream
    with stream_lock:
        if stream and stream.active:
            stream.stop()
            stream.close()
        kw = dict(
            callback    = _audio_callback,
            samplerate  = SAMPLE_RATE,
            blocksize   = CHUNK,
            dtype       = 'float32',
            channels    = 1,
        )
        if in_dev  is not None: kw['device'] = (int(in_dev), int(out_dev) if out_dev is not None else None)
        stream = sd.Stream(**kw)
        stream.start()


def stop_stream():
    global stream
    with stream_lock:
        if stream:
            stream.stop()
            stream.close()
            stream = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    devices = _device_list()
    keys    = [{'index': i, 'name': n} for i, n in enumerate(KEY_NAMES)]
    scales  = list(SCALES.keys())
    return render_template('index.html', devices=devices, keys=keys, scales=scales)


@app.route('/api/devices')
def api_devices():
    return jsonify(_device_list())


@app.route('/api/start', methods=['POST'])
def api_start():
    d    = request.json or {}
    in_d = d.get('input_device')
    ou_d = d.get('output_device')
    try:
        start_stream(in_d, ou_d)
        tuner.update(active=True)
        return jsonify({'ok': True})
    except Exception as ex:
        return jsonify({'error': str(ex)}), 500


@app.route('/api/stop', methods=['POST'])
def api_stop():
    tuner.update(active=False)
    stop_stream()
    return jsonify({'ok': True})


@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    return jsonify(tuner.get_status())


@app.route('/api/settings', methods=['POST'])
def api_settings_post():
    d = request.json or {}
    updates = {}
    try:
        if 'key'    in d: updates['key']    = int(d['key']) % 12
        if 'scale'  in d: updates['scale']  = str(d['scale'])
        if 'amount' in d: updates['amount'] = max(0.0, min(1.0, float(d['amount'])))
        if 'speed'  in d: updates['speed']  = max(0.0, min(1.0, float(d['speed'])))
    except (TypeError, ValueError) as ex:
        return jsonify({'error': f'invalid settings payload: {ex}'}), 400
    tuner.update(**updates)
    return jsonify({'ok': True})


@app.route('/api/status/stream')
def api_status_stream():
    """Server-Sent Events — pushes status to UI every 80ms."""
    def generate():
        while True:
            status = tuner.get_status()
            yield f"data: {json.dumps(status)}\n\n"
            time.sleep(0.08)
    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _device_list():
    try:
        devs = sd.query_devices()
    except Exception:
        # No audio backend / no devices available (e.g. headless machine) —
        # let the UI render with an empty device list instead of a 500.
        return []
    out = []
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0 or d['max_output_channels'] > 0:
            out.append({
                'index':   i,
                'name':    d['name'],
                'inputs':  d['max_input_channels'],
                'outputs': d['max_output_channels'],
                'default_sr': int(d['default_samplerate']),
            })
    return out


if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5576))
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
