# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
python app.py
```

Starts a Flask-SocketIO server on `http://0.0.0.0:5000`. Requires `ffmpeg` and `ffprobe` to be available in PATH.

## Dependencies

No `requirements.txt` exists. Core packages needed:

```
flask flask-socketio openai-whisper torch werkzeug
```

Install with: `pip install flask flask-socketio openai-whisper torch`

## Architecture

Single-file backend (`app.py`) + single Jinja template (`templates/index.html`).

**Transcription pipeline:**

1. Client uploads MP3 via `POST /upload` (max 500 MB, enforced again at 300 MB in JS)
2. Client emits Socket.IO event `transcribir` with the saved filename
3. Server splits the audio into 360-second chunks (with 8 s overlap) using `ffmpeg`, resampled to 16 kHz mono WAV
4. Each chunk is transcribed with Whisper `small` model; segments are filtered with `detect_ngram_loop()` to catch Whisper hallucination loops — looping segments are re-transcribed via `recover_segment()` with looser params
5. Valid segments are accumulated and flushed every 30 seconds (`BLOCK_INTERVAL`) as `transcripcion_parcial` Socket.IO events, which the client appends to the result text area in real time
6. Progress is reported via `archivo_progreso` (5 % after validation, 8–96 % during chunking, 100 % on finish)

**Model loading:** lazy singleton `get_model()` — loads on first transcription request, stays in memory.

**Device:** auto-selects CUDA if available, otherwise CPU. `fp16` is enabled only on CUDA.

**Output format:** blocks formatted as:
```
S
Vocero
MM:SS - MM:SS
<transcribed text>
```

**Domain context:** tuned for Peruvian judicial hearings in Spanish — `initial_prompt` and Whisper params (`WHISPER_PARAMS`) are set accordingly. Changing the language or domain requires updating both `WHISPER_PARAMS` and the `recover_segment()` call inside `app.py`.

**Frontend (index.html):** vanilla JS with Socket.IO client. Two modes — *Individual* (one file at a time) and *Cola* (queue, processed sequentially). All state lives in JS variables (`queue`, `transcriptions`, `editingStates`). No build step.
