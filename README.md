# AutoTuner

Real-time vocal pitch correction. Sing into a microphone, pick a key and scale, and hear
the corrected signal back with latency low enough to perform against.

## How it works

**Pitch detection is YIN**, not a naive FFT peak-pick. FFT resolution at the low end of a
vocal range is too coarse to tell a flat note from an in-tune one, and it octave-errors on
signals with a weak fundamental, which is most singing. YIN works on the autocorrelation
difference function instead and stays stable through vibrato.

**Shifting is pyrubberband**, chosen because it preserves formants. Naive resampling moves
the vocal tract resonances along with the pitch, which is the chipmunk artifact. Rubber Band
keeps them where they were, so a corrected note still sounds like the same person.

## Controls

| Control | What it does |
|---|---|
| Key + scale | The set of notes the detected pitch is allowed to snap to |
| Amount | How far toward the target note to move, 0 to 100 percent |
| Speed | How fast the correction arrives. Instant is the hard-tuned effect, slower is transparent |

Amount and speed are separate on purpose. They are the two independent axes between
"nobody can hear this" and the deliberate artifact.

## Running it

```bash
make run
```

Or double-click `launch.command` on a Mac. Then open the local URL and allow microphone
access.

## Status

Working prototype. It exists to prove the DSP chain before that chain is rebuilt in C++ as
a real VST3 plugin, so the priority here was getting correction quality right rather than
packaging.

Python, Flask, YIN, pyrubberband.
