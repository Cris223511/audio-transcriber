import os
import subprocess
import json
import math
import time
import threading
import torch
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import whisper
from werkzeug.utils import secure_filename

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
                                socketio.emit("dev_reload", broadcast=True)
                            watched[path] = mtime
                        except OSError:
                            pass
            time.sleep(0.5)

    threading.Thread(target=_watch, daemon=True).start()

_model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_DURATION = 360
OVERLAP_SEC = 8
BLOCK_INTERVAL = 30

WHISPER_PARAMS = dict(
    language="es",
    verbose=False,
    initial_prompt=(
        "Audiencia judicial peruana. Magistrado, fiscal, defensa, imputado, testigo. "
        "Vocabulario legal: diligencia, acta, disposición fiscal, RENIEC, PNP, requerimiento acusatorio."
    ),
    condition_on_previous_text=False,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold=1.8,
    logprob_threshold=-0.8,
    no_speech_threshold=0.45,
    beam_size=3,
    fp16=DEVICE == "cuda",
)


def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("small", device=DEVICE)
    return _model


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


def split_audio(filepath, total_duration, chunks_dir):
    os.makedirs(chunks_dir, exist_ok=True)
    num_chunks = math.ceil(total_duration / CHUNK_DURATION)
    chunks = []
    for i in range(num_chunks):
        is_first = i == 0
        start = max(0.0, i * CHUNK_DURATION - (OVERLAP_SEC if not is_first else 0))
        duration = CHUNK_DURATION + (OVERLAP_SEC if not is_first else 0)
        chunk_path = os.path.join(chunks_dir, f"chunk_{i:04d}.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                filepath,
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                chunk_path,
            ],
            capture_output=True,
            check=True,
        )
        chunks.append(
            {
                "path": chunk_path,
                "offset": start,
                "skip": OVERLAP_SEC if not is_first else 0.0,
                "index": i + 1,
                "total": num_chunks,
            }
        )
    return chunks


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


def detect_ngram_loop(text, min_reps=3):
    words = text.split()
    n = len(words)
    if n < 6:
        return False
    for seq_len in range(2, n // min_reps + 1):
        seq = words[:seq_len]
        i, reps = 0, 0
        while i + seq_len <= n:
            if words[i : i + seq_len] == seq:
                reps += 1
                i += seq_len
            else:
                break
        if reps >= min_reps:
            return True
    return False


def recover_segment(model, chunk_path, start_sec, end_sec):
    tmp = chunk_path.replace(".wav", "_rec.wav")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                chunk_path,
                "-ss",
                str(start_sec),
                "-t",
                str(end_sec - start_sec),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                tmp,
            ],
            capture_output=True,
            check=True,
        )
        result = model.transcribe(
            tmp,
            language="es",
            verbose=False,
            initial_prompt=WHISPER_PARAMS["initial_prompt"],
            condition_on_previous_text=False,
            temperature=(0.6, 0.8, 1.0),
            compression_ratio_threshold=1.5,
            logprob_threshold=-0.6,
            no_speech_threshold=0.35,
            beam_size=3,
        )
        texts = [
            seg["text"].strip()
            for seg in result.get("segments", [])
            if seg["text"].strip() and not detect_ngram_loop(seg["text"].strip())
        ]
        return " ".join(texts) if texts else None
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def process_segments(model, chunk, segments):
    skip = chunk["skip"]
    offset = chunk["offset"]
    recovered = []
    for seg in segments:
        if seg["start"] < skip:
            continue
        text = seg["text"].strip()
        if not text:
            continue
        abs_start = seg["start"] + offset
        abs_end = seg["end"] + offset
        if detect_ngram_loop(text):
            recovered_text = recover_segment(
                model, chunk["path"], seg["start"], seg["end"]
            )
            if recovered_text:
                recovered.append((abs_start, abs_end, recovered_text))
        else:
            recovered.append((abs_start, abs_end, text))
    return recovered


def group_into_blocks(segments):
    if not segments:
        return []
    blocks = []
    blk_start, blk_end = segments[0][0], segments[0][1]
    blk_texts = [segments[0][2]]
    for start, end, text in segments[1:]:
        if start - blk_start >= BLOCK_INTERVAL and blk_texts:
            blocks.append(
                f"S\nVocero\n{fmt_time(blk_start)} - {fmt_time(blk_end)}\n{' '.join(blk_texts)}\n\n"
            )
            blk_start, blk_texts = start, []
        blk_texts.append(text)
        blk_end = end
    if blk_texts:
        blocks.append(
            f"S\nVocero\n{fmt_time(blk_start)} - {fmt_time(blk_end)}\n{' '.join(blk_texts)}\n\n"
        )
    return blocks


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
    filename = secure_filename(archivo.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    archivo.save(filepath)
    return {"filename": filename, "size": os.path.getsize(filepath)}


@socketio.on("transcribir")
def handle_transcripcion(data):
    filename = data["filename"]
    file_id = data.get("fileId", filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    chunks_dir = os.path.join(app.config["UPLOAD_FOLDER"], f"chunks_{filename}")

    try:
        emit("log", {"tipo": "info", "mensaje": f"[{filename}] Validando..."})
        duration = get_duration(filepath)

        if duration > 10800:
            emit("log", {"tipo": "error", "mensaje": f"[{filename}] Excede 3 horas"})
            emit("archivo_error", {"fileId": file_id})
            os.remove(filepath)
            return

        emit(
            "log",
            {
                "tipo": "success",
                "mensaje": f"[{filename}] Duración: {fmt_time(duration)}",
            },
        )
        emit("log", {"tipo": "info", "mensaje": f"Dispositivo: {DEVICE.upper()}"})
        emit("archivo_progreso", {"fileId": file_id, "porcentaje": 5})

        model = get_model()
        emit("log", {"tipo": "success", "mensaje": "Modelo listo"})

        emit("log", {"tipo": "info", "mensaje": f"[{filename}] Segmentando..."})
        chunks = split_audio(filepath, duration, chunks_dir)
        emit(
            "log",
            {"tipo": "success", "mensaje": f"[{filename}] {len(chunks)} segmentos"},
        )

        for chunk in chunks:
            idx, total = chunk["index"], chunk["total"]
            progreso = 8 + int((idx / total) * 88)
            emit("archivo_progreso", {"fileId": file_id, "porcentaje": progreso})
            emit(
                "log",
                {"tipo": "info", "mensaje": f"[{filename}] Parte {idx}/{total}..."},
            )

            result = model.transcribe(chunk["path"], **WHISPER_PARAMS)
            raw_segments = result.get("segments", [])
            processed = process_segments(model, chunk, raw_segments)
            blocks = group_into_blocks(processed)

            for block in blocks:
                emit("transcripcion_parcial", {"fileId": file_id, "texto": block})

            emit(
                "log",
                {"tipo": "success", "mensaje": f"[{filename}] Parte {idx}/{total} ✓"},
            )

        emit("archivo_progreso", {"fileId": file_id, "porcentaje": 100})
        emit("log", {"tipo": "success", "mensaje": f"[{filename}] Completado ✓"})
        emit("archivo_completado", {"fileId": file_id})

    except Exception as e:
        emit("log", {"tipo": "error", "mensaje": f"[{filename}] Error: {e}"})
        emit("archivo_error", {"fileId": file_id})
    finally:
        cleanup(filepath, chunks_dir)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    print(f"Dispositivo: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    # WERKZEUG_RUN_MAIN is set by the reloader in the worker process; start the
    # template watcher only there so it doesn't run twice.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _start_template_watcher()
    socketio.run(app, debug=True, use_reloader=True, allow_unsafe_werkzeug=True, host="0.0.0.0", port=5000)
