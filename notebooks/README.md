# Notebooks

All four notebooks work on the same problem. A camera watches a room. The user writes what to watch
for in one line: "the cat got onto the table", "the washing machine finished". The system has to
notice the moment it happens.

We use still frames for now. Live video comes later.

Frames are in `data/`: `1.png`, `2.png`, `3.png`. Night footage, infrared, about 2000 pixels wide.
The cat is on the table only in `3.png`.

| Notebook | Question | Answer |
|---|---|---|
| [`frame_gate.ipynb`](frame_gate.ipynb) | Which frames are worth sending to the model? | The ones where some small part of the picture changed a lot. 3 calls for 7 frames, 4.9 ms per check |
| [`groq.ipynb`](groq.ipynb) | Can Groq spot the event? | Only one frame at a time. Comparing frames is left to Python. 3/3 frames, 5/5 cases |
| [`gate.ipynb`](gate.ipynb) | Why check frames that way and not the simple way? | The simple way mistakes a lamp for a cat |
| [`gemini.ipynb`](gemini.ipynb) | Gemini: prompts, image size, which model | `gemini-3.1-flash-lite` at low resolution, 2.4s per call |

Start with `frame_gate` and `groq` to see how the thing works. The other two are notes from the
experiments. Open them if you want to know why a decision was made.

## frame_gate.ipynb — deciding what to send

A model call costs about half a second and 900 tokens. Most frames are not worth it, because
nothing happened in the room. Grain, a light switching on and image compression change the pixels
anyway.

So before each call we ask a cheap local question: did anything change since the last frame we paid
for? The check takes 4.9 ms:

1. shrink the frame to 128×72 and turn it grey;
2. blur it, which removes the grain;
3. make both frames the same brightness, which removes lamps;
4. split the picture into an 8×8 grid and find the square that changed the most;
5. if that number is above the threshold, send the frame.

Two details matter. We compare against the **last frame we sent**, not the previous frame. And we
look at the worst square, not the whole picture. A cat covers a few percent of a wide shot, so an
average over the whole frame barely moves.

The demo runs seven frames in a row. Grain, a brightness jump and a re-compressed copy are all
skipped. Both real changes are sent.

One more measurement: decoding the original 2000px PNG takes 61 ms, twelve times longer than the
check itself. Take frames from the camera already small.

Still to do, listed in the last cell: set the threshold using your own camera, send one frame every
few minutes anyway (the check can say "nothing changed" but never "the event did not happen"), stay
quiet for a few seconds after a call, and handle a bumped camera.

## groq.ipynb — the provider, and a change of plan

Groq has two models that can look at images. One cannot read this night footage at all. The other
sees objects fine but cannot compare two frames and say what changed.

Instead of fighting it with prompts, we split the job. **The model looks at one frame and answers
one question: is the event already true here?** Python does the rest: the event happened if the
answer was "no" before and "yes" after.

This works: 3 frames out of 3 correct, 5 cases out of 5. It also costs less, because there is one
request per frame instead of one per pair.

Smaller things that were needed: turn reasoning off, or the answer gets buried in the model's
thinking; ask for JSON instead of parsing text; report request time and rate-limit waiting
separately, so a slow plan does not look like a slow model. The event is just a string in
`USER_EVENT`, so nothing in the code is about cats. About 916 tokens and half a second per frame.

## gate.ipynb — why that check

Seven pairs of frames. In four of them the camera produced a new file while the room stood still:
the same frame twice, added grain, a brightness step from the infrared light, and a re-compressed
copy. In three of them the cat actually moved. Four ways to measure the difference:

| how we measure | worst "nothing happened" | weakest real change | gap |
|---|---|---|---|
| average over the whole frame | 19.86 (brightness) | 2.01 | 0.10× |
| same, after blurring | 19.86 (brightness) | 1.07 | 0.05× |
| image hash, 256 bits | 5 (grain) | 6 | 1.20× |
| **worst square of an 8×8 grid** | **4.19** | **8.94** | **2.13×** |

A metric is only usable if the loudest thing we must ignore scores lower than the quietest thing we
must catch. Only the last one does.

The average over the frame fails badly: a light turning on scores ten times higher than a cat
walking onto a table. The hash ignores brightness completely, but at 8×8 it is too coarse to notice
the cat at all, so it had to be 16×16.

There is also a section on what to compare against. The same change was spread over 10 frames. When
each frame is compared to the one before it, the score never gets above 1.17 and the threshold of
6.6 is never reached, so the event is missed entirely. When every frame is compared to the last one
we sent, the difference builds up and the check fires once.

No API calls in this notebook, everything runs locally. Side finding: the timestamp printed in the
corner of the image changes every frame, so two frames from this camera are never identical.

## gemini.ipynb — the first attempt

Where the project started, and the only notebook written in Russian. It sends two frames at once and
asks the model to compare them: a fixed system prompt (find the difference, check whether the user's
event happened) plus a separate line for what the user is waiting for. Then five measurements:

- **two frames glued into one image vs sent separately** — glueing helps at full resolution, but at
  low resolution both get 5/5, so this is not where the win is;
- **image resolution** — the only setting that really cuts cost: 1223 tokens down to 422;
- **frame width** — changes nothing, Gemini resizes images on its own side;
- **speed** — the API refuses a timeout shorter than 10 seconds, and sending two requests at once to
  take the faster answer did not help either. The delay is on the provider's side;
- **model** — `gemini-3.1-flash-lite` is as accurate and much faster: 2.4s instead of 9.9s.

It also has a local check before sending, running requests in parallel, and retries when the API
returns 503. That local check uses the frame average, which `gate.ipynb` later showed to be the
weakest option. The threshold there was never updated.

## Running

```bash
make notebook   # Jupyter Lab on :8889
```

Keys go in `.env` (`GEMINI_API_KEY`, `GROQ_API_KEY`), see `.env.example`.
`gate.ipynb` and `frame_gate.ipynb` need no keys.
Rules for agents editing notebooks are in [`AGENTS.md`](../AGENTS.md).
