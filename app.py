import os
import re
import json
import math
import time
import wave
import difflib
import threading
import subprocess
import unicodedata
from uuid import uuid4

import numpy as np
import torch
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import whisper
from whisper.tokenizer import get_tokenizer
from werkzeug.utils import secure_filename

import diarization

app = Flask(__name__)
app.config["SECRET_KEY"] = "clave_secreta_123"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def _start_template_watcher():
    watched: dict[str, float] = {}

    def _watch():
        while True:
            for root, _, files in os.walk("templates"):
                for fname in files:
                    if fname.endswith((".html", ".css", ".js")):
                        path = os.path.join(root, fname)
                        try:
                            mtime = os.path.getmtime(path)
                            if path in watched and watched[path] != mtime:
                                socketio.emit("dev_reload")  # sin `to` ya llega a todos los clientes
                            watched[path] = mtime
                        except Exception:
                            pass
            time.sleep(0.5)

    threading.Thread(target=_watch, daemon=True).start()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _default_model_name():
    if DEVICE == "cuda":
        try:
            if torch.cuda.get_device_properties(0).total_memory >= 6 * 1024**3:
                return "turbo"
        except Exception:
            pass
    return "small"


MODEL_NAME = os.environ.get("WHISPER_MODEL") or _default_model_name()

SAMPLE_RATE = 16000
MAX_DURATION = 10800  # 3 h
CHUNK_DURATION = 360
OVERLAP_SEC = 10
PIECE_MAX_GAP = 0.5
PIECE_MAX_SEC = 3.0
VAD_MERGE_GAP = 4.0
MAX_RECOVERIES = 5
RECOVER_PAD = 0.3
PROMPT_ECHO_MIN = 30  # caracteres seguidos copiados del prompt para considerar que Whisper lo repitió

# Prompt neutro (sirve para cualquier audio: reuniones, videos, clases, música...). Solo marca el estilo:
# español con puntuación y mayúsculas correctas. No aparece en la transcripción.
INITIAL_PROMPT = "Esta es una transcripción en español, con puntuación y mayúsculas correctas."

# Frases típicas de alucinación de Whisper (texto normalizado: minúsculas y sin tildes)
HALLUCINATION_PHRASES = (
    "amara.org",
    "subtitulos realizados por",
    "subtitulado por",
    "subtitulos por la comunidad",
    "suscribete",
    "gracias por ver el video",
    "www.",
)

# Red de seguridad: los tokens de este texto nunca se suprimen (todos los signos del español)
SPANISH_SAMPLE = (
    "¿Está usted de acuerdo? ¡Sí, claro! El pingüino y el ñandú cruzaron. "
    "Ángel, Érica, Íñigo, Óscar, Úrsula, Ñuño y GÜEMES dijeron «sin presión» a 30 °C. "
    "Es la 1º reunión y la 2ª sesión; el informe dice “conforme” y ‘visto’ — fin – listo… Pagó 50 €. "
    "Signos: á é í ó ú ü ñ Á É Í Ó Ú Ü Ñ ¿ ¡ « » ° º ª “ ” ‘ ’ — – … €"
)

# Caracteres permitidos: tab, salto de línea, ASCII imprimible, Latin-1 y – — ‘ ’ “ ” … €
_DISALLOWED_RE = re.compile(r"[^\t\n\x20-\x7e\xa0-\xff–—‘’“”…€]")

_model = None
_model_lock = threading.Lock()
_gpu_lock = threading.Lock()  # serializa todas las llamadas a model.transcribe
_suppress_cache = {}
_vad = None
_vad_lock = threading.Lock()


# ── Texto ─────────────────────────────────────────────────────────────


def clean_chars(s):
    return _DISALLOWED_RE.sub("", s)


def sanitize_text(s):
    s = clean_chars(s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r" +([,.;:?!…])", r"\1", s)
    s = re.sub(r"([¿¡]) +", r"\1", s)
    return s.strip()


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_text(s):
    return re.sub(r"\s+", " ", strip_accents(s.lower())).strip()


def normalize_word(w):
    return re.sub(r"[\W_]+", "", strip_accents(w.lower()))


_PROMPT_NORM = normalize_text(INITIAL_PROMPT)


def collapse_repeats(words):
    """Colapsa n-gramas (n=1..8) repetidos seguidos; devuelve (palabras, n.º eliminadas)."""
    keys = [normalize_word(w["word"] if isinstance(w, dict) else w) for w in words]
    kept = list(range(len(words)))
    changed = True
    while changed:
        changed = False
        for n in range(1, 9):
            min_reps = 4 if n == 1 else 3
            out, i = [], 0
            while i < len(kept):
                gram = [keys[j] for j in kept[i : i + n]]
                reps = 1
                if len(gram) == n:
                    while [keys[j] for j in kept[i + reps * n : i + (reps + 1) * n]] == gram:
                        reps += 1
                if reps >= min_reps:
                    out.extend(kept[i : i + n])
                    i += reps * n
                    changed = True
                else:
                    out.append(kept[i])
                    i += 1
            kept = out
    return [words[j] for j in kept], len(words) - len(kept)


def capitalize_sentence(text, prev_text):
    """Mayúscula inicial si la pieza anterior cerró oración (o es la primera del archivo)."""
    if prev_text is not None and not prev_text.endswith((".", "?", "!", "…")):
        return text
    i = 0
    while i < len(text) and text[i] in "¿¡\"«(":
        i += 1
    if i < len(text) and text[i].islower():
        text = text[:i] + text[i].upper() + text[i + 1 :]
    return text


def build_pieces(groups):
    """groups: una lista de palabras (tiempos absolutos) por segmento de Whisper -> piezas <= 3 s."""
    raw = []
    for words in groups:
        cur = []
        for w in words:
            if cur and (
                w["start"] - cur[-1]["end"] > PIECE_MAX_GAP
                or w["end"] - cur[0]["start"] > PIECE_MAX_SEC
            ):
                raw.append(cur)
                cur = []
            cur.append(w)
        if cur:
            raw.append(cur)
    pieces = []
    for ws in raw:
        text = sanitize_text("".join(w["word"] for w in ws))
        if text:
            pieces.append({"start": ws[0]["start"], "end": max(w["end"] for w in ws), "text": text})
    return pieces


# ── Whisper ───────────────────────────────────────────────────────────


def build_suppress_tokens(tokenizer):
    """Tokens a suprimir: los que generan caracteres fuera de lo permitido o fragmentos de bytes."""
    enc = tokenizer.encoding
    ids = set()
    for t in range(tokenizer.eot):
        try:
            b = enc.decode_single_token_bytes(t)
        except KeyError:
            continue
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            ids.add(t)
            continue
        if _DISALLOWED_RE.search(s):
            ids.add(t)
    ids -= set(enc.encode(SPANISH_SAMPLE))
    return sorted(ids)


def get_suppress_tokens(model):
    key = (model.is_multilingual, model.num_languages)
    if key not in _suppress_cache:
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language="es",
            task="transcribe",
        )
        _suppress_cache[key] = build_suppress_tokens(tokenizer)
    return _suppress_cache[key]


def whisper_params(suppress_ids, **overrides):
    params = dict(
        language="es",
        task="transcribe",
        verbose=False,
        initial_prompt=INITIAL_PROMPT,
        carry_initial_prompt=True,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        beam_size=5,
        best_of=3,
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
        fp16=DEVICE == "cuda",
        suppress_tokens=[-1] + list(suppress_ids),
    )
    params.update(overrides)
    return params


def get_model(notify=None):
    global _model
    with _model_lock:
        if _model is None:
            msg = f"Cargando modelo Whisper '{MODEL_NAME}' (la primera vez se descarga y puede tardar)..."
            print(msg)
            if notify:
                notify(msg)
            _model = whisper.load_model(MODEL_NAME, device=DEVICE)
    return _model


# ── Audio ─────────────────────────────────────────────────────────────


def get_duration(filepath):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def prepare_audio(filepath, wav_path):
    """Decodifica todo el archivo una sola vez a WAV 16 kHz mono; devuelve la duración en segundos."""
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", filepath, "-vn",
            "-af", "highpass=f=80", "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            wav_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        lines = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg no pudo preparar el audio: {lines[-1] if lines else result.returncode}")
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / SAMPLE_RATE


def plan_chunks(total, chunk_sec=CHUNK_DURATION, overlap=OVERLAP_SEC):
    """Partes con solape; la ventana keep_from/keep_until reparte el solape por su punto medio."""
    n = math.ceil(total / chunk_sec) if total > 0 else 0
    chunks = []
    for i in range(n):
        chunks.append(
            {
                "index": i + 1,
                "total": n,
                "start": 0.0 if i == 0 else i * chunk_sec - overlap,
                "end": min(total, (i + 1) * chunk_sec),
                "keep_from": 0.0 if i == 0 else i * chunk_sec - overlap / 2,
                "keep_until": (i + 1) * chunk_sec - overlap / 2 if i < n - 1 else math.inf,
            }
        )
    return chunks


def read_wav_slice(wav_path, start, end):
    with wave.open(wav_path, "rb") as wf:
        a = int(round(start * SAMPLE_RATE))
        b = min(wf.getnframes(), int(round(end * SAMPLE_RATE)))
        wf.setpos(a)
        data = wf.readframes(max(0, b - a))
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def cleanup(filepath, chunks_dir):
    try:
        os.remove(filepath)
    except OSError:
        pass
    try:
        for f in os.listdir(chunks_dir):
            os.remove(os.path.join(chunks_dir, f))
        os.rmdir(chunks_dir)
    except OSError:
        pass


# ── VAD (opcional, Silero de faster-whisper) ─────────────────────────


def _get_vad():
    # Import diferido: faster_whisper arrastra ctranslate2/transformers (~5 s)
    global _vad
    with _vad_lock:
        if _vad is None:
            try:
                from faster_whisper.vad import VadOptions, get_speech_timestamps

                _vad = (VadOptions, get_speech_timestamps)
            except Exception:
                _vad = False
    return _vad or None


def merge_speech_regions(regions, sr=SAMPLE_RATE, max_gap=VAD_MERGE_GAP):
    """Une regiones de voz (en muestras) separadas por < max_gap s; devuelve [(inicio, fin)] en segundos."""
    clips = []
    for r in regions:
        s, e = r["start"] / sr, r["end"] / sr
        if clips and s - clips[-1][1] < max_gap:
            clips[-1][1] = max(clips[-1][1], e)
        else:
            clips.append([s, e])
    return [(s, e) for s, e in clips]


def speech_clips(audio):
    """Clips con voz del chunk; None = VAD no disponible (procesar todo el chunk)."""
    vad = _get_vad()
    if vad is None:
        return None
    VadOptions, get_speech_timestamps = vad
    try:
        regions = get_speech_timestamps(
            audio,
            vad_options=VadOptions(onset=0.4, offset=0.3, min_silence_duration_ms=1000, speech_pad_ms=400),
            sampling_rate=SAMPLE_RATE,
        )
    except Exception:
        return None
    return merge_speech_regions(regions)


# ── Post-proceso de segmentos ────────────────────────────────────────


def is_junk_segment(seg):
    if seg.get("no_speech_prob", 0.0) > 0.6 and seg.get("avg_logprob", 0.0) < -0.75:
        return True
    text = normalize_text(seg.get("text", ""))
    if any(p in text for p in HALLUCINATION_PHRASES):
        return True
    match = difflib.SequenceMatcher(None, text, _PROMPT_NORM, autojunk=False).find_longest_match(
        0, len(text), 0, len(_PROMPT_NORM)
    )
    return match.size >= PROMPT_ECHO_MIN


def keep_words(seg, offset, keep_from, keep_until):
    """Palabras del segmento en tiempo absoluto, dentro de la ventana del chunk y saneadas."""
    words = seg.get("words") or [{"word": seg.get("text", ""), "start": seg["start"], "end": seg["end"]}]
    kept = []
    for w in words:
        start = w["start"] + offset
        if not keep_from <= start < keep_until:
            continue
        text = clean_chars(w["word"])  # conserva el espacio inicial para unir palabras
        if text.strip():
            kept.append({"word": text, "start": start, "end": w["end"] + offset})
    return kept


def recover_segment(model, audio, seg, chunk, suppress_ids):
    """Re-transcribe un segmento atascado en bucle; devuelve grupos de palabras o None."""
    s = max(0.0, seg["start"] - RECOVER_PAD)
    e = min(len(audio) / SAMPLE_RATE, seg["end"] + RECOVER_PAD)
    piece = audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)]
    if len(piece) == 0:
        return None
    params = whisper_params(
        suppress_ids,
        initial_prompt=None,
        carry_initial_prompt=False,
        temperature=(0.0, 0.2, 0.4),
        compression_ratio_threshold=2.0,
    )
    try:
        with _gpu_lock:
            result = model.transcribe(piece, **params)
    except Exception:
        return None
    groups, total, removed = [], 0, 0
    for rseg in result.get("segments", []):
        if is_junk_segment(rseg):
            continue
        words = keep_words(rseg, chunk["start"] + s, chunk["keep_from"], chunk["keep_until"])
        words, rem = collapse_repeats(words)
        total += len(words) + rem
        removed += rem
        if words:
            groups.append(words)
    if total == 0 or removed > 0.2 * total:
        return None
    return groups


def postprocess_chunk(model, audio, chunk, segments, suppress_ids):
    groups = []
    recoveries = 0
    for seg in segments:
        if is_junk_segment(seg):
            continue
        words = keep_words(seg, chunk["start"], chunk["keep_from"], chunk["keep_until"])
        if not words:
            continue
        collapsed, removed = collapse_repeats(words)
        if len(words) >= 6 and removed > 0.5 * len(words) and recoveries < MAX_RECOVERIES:
            recoveries += 1
            recovered = recover_segment(model, audio, seg, chunk, suppress_ids)
            if recovered is not None:
                groups.extend(recovered)
                continue
        groups.append(collapsed)
    return build_pieces(groups)


def transcribe_chunk(model, audio, chunk, suppress_ids):
    clips = speech_clips(audio)
    if clips is not None and not clips:
        return []  # sin voz: no se usa Whisper
    params = whisper_params(suppress_ids)
    if clips:
        # Whisper solo tiene len(audio)//160 frames de mel: un fin de clip más allá hace que transcribe() no termine nunca
        limit = (len(audio) // 160) / 100.0
        params["clip_timestamps"] = [round(min(t, limit), 3) for clip in clips for t in clip]
    with _gpu_lock:
        result = model.transcribe(audio, **params)
    return postprocess_chunk(model, audio, chunk, result.get("segments", []), suppress_ids)


# ── Rutas y Socket.IO ────────────────────────────────────────────────


def display_name(filename):
    return re.sub(r"^[0-9a-f]{8}_", "", filename)


def parse_num_speakers(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 1 <= value <= 10 else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "archivo" not in request.files:
        return {"error": "No hay archivo"}, 400
    archivo = request.files["archivo"]
    if not archivo.filename:
        return {"error": "Archivo vacío"}, 400
    filename = f"{uuid4().hex[:8]}_{secure_filename(archivo.filename) or 'audio'}"
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    archivo.save(filepath)
    return {"filename": filename, "size": os.path.getsize(filepath)}


@socketio.on("transcribir")
def handle_transcripcion(data):
    data = data if isinstance(data, dict) else {}
    filename = secure_filename(str(data.get("filename") or ""))
    file_id = data.get("fileId") or filename
    num_speakers = parse_num_speakers(data.get("numSpeakers"))
    name = display_name(filename)
    try:
        sid, namespace = request.sid, request.namespace or "/"
    except RuntimeError:  # llamada fuera de un evento de Socket.IO (pruebas)
        sid, namespace = None, "/"

    def client_gone():
        # El cliente marca el trabajo como error al desconectarse: seguir solo gastaría GPU
        return sid is not None and not socketio.server.manager.is_connected(sid, namespace)

    def log(tipo, mensaje):
        emit("log", {"tipo": tipo, "mensaje": mensaje})

    def progress(porcentaje):
        emit("archivo_progreso", {"fileId": file_id, "porcentaje": porcentaje})

    if not filename:
        log("error", "Nombre de archivo inválido")
        emit("archivo_error", {"fileId": file_id})
        return

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    chunks_dir = os.path.join(app.config["UPLOAD_FOLDER"], f"chunks_{filename}")

    try:
        log("info", f"[{name}] Validando...")
        duration = get_duration(filepath)

        if duration > MAX_DURATION:
            log("error", f"[{name}] Excede 3 horas")
            emit("archivo_error", {"fileId": file_id})
            os.remove(filepath)
            return

        log("success", f"[{name}] Duración: {fmt_time(duration)}")
        log("info", f"Dispositivo: {DEVICE.upper()}")
        progress(5)

        model = get_model(notify=lambda msg: log("info", msg))
        suppress_ids = get_suppress_tokens(model)
        log("success", f"Modelo listo ({MODEL_NAME})")

        diarizer = None
        try:
            diarization.get_embedder()
            diarizer = diarization.SpeakerDiarizer(num_speakers)
        except Exception as e:
            log("warning", f"[{name}] Diarización no disponible: {e}")

        if client_gone():
            print(f"[{name}] Cliente desconectado: trabajo cancelado")
            return
        log("info", f"[{name}] Preparando audio...")
        os.makedirs(chunks_dir, exist_ok=True)
        wav_path = os.path.join(chunks_dir, "prepared.wav")
        chunks = plan_chunks(prepare_audio(filepath, wav_path))
        log("success", f"[{name}] {len(chunks)} partes")

        next_id = 0
        prev_text = None
        hablantes = None
        for chunk in chunks:
            if client_gone():
                print(f"[{name}] Cliente desconectado: trabajo cancelado")
                return
            idx, total = chunk["index"], chunk["total"]
            progress(8 + int((idx - 1) / total * 88))
            log("info", f"[{name}] Parte {idx}/{total}...")

            audio = read_wav_slice(wav_path, chunk["start"], chunk["end"])
            pieces = transcribe_chunk(model, audio, chunk, suppress_ids)

            segmentos = []
            for p in pieces:
                p["text"] = capitalize_sentence(p["text"], prev_text)
                prev_text = p["text"]
                segmentos.append(
                    {"id": next_id, "start": round(p["start"], 2), "end": round(p["end"], 2), "text": p["text"]}
                )
                next_id += 1

            hablantes = None
            if diarizer is not None:
                try:
                    for p in pieces:
                        emb = diarization.embed_piece(audio, p["start"] - chunk["start"], p["end"] - chunk["start"])
                        diarizer.add(p["start"], p["end"], emb)
                    hablantes = diarizer.labels()
                except Exception as e:
                    log("warning", f"[{name}] Diarización desactivada: {e}")
                    diarizer = None
                    hablantes = None

            emit(
                "transcripcion_segmentos",
                {"fileId": file_id, "segmentos": segmentos, "hablantes": hablantes},
            )
            progress(8 + int(idx / total * 88))
            log("success", f"[{name}] Parte {idx}/{total} ✓ ({len(segmentos)} segmentos)")

        if diarizer is not None and hablantes:
            log("success", f"[{name}] Hablantes detectados: {len(set(hablantes))}")
        progress(100)
        log("success", f"[{name}] Completado ✓")
        emit("archivo_completado", {"fileId": file_id})

    except Exception as e:
        log("error", f"[{name}] Error: {e}")
        emit("archivo_error", {"fileId": file_id})
    finally:
        cleanup(filepath, chunks_dir)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    print(f"Dispositivo: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Modelo Whisper: {MODEL_NAME}")
    # WERKZEUG_RUN_MAIN is set by the reloader in the worker process; start the
    # template watcher only there so it doesn't run twice.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _start_template_watcher()
    socketio.run(app, debug=True, use_reloader=True, allow_unsafe_werkzeug=True, host="0.0.0.0", port=5000)
