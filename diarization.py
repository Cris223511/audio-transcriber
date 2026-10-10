"""Diarización: embeddings de hablante (WeSpeaker ResNet34-LM, ONNX) + clustering jerárquico."""

import os
import threading

import numpy as np

SAMPLE_RATE = 16000
EMBED_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
EMBED_FILE = "voxceleb_resnet34_LM.onnx"

MIN_EMBED_SEC = 0.4  # piezas más cortas no reciben embedding
TARGET_EMBED_SEC = 1.0  # piezas cortas se amplían a este mínimo
TINY_CLUSTER_SEC = 4.0  # clusters con menos voz total se reasignan
SMOOTH_MAX_SEC = 1.0  # piezas cortas "sándwich" toman la etiqueta vecina
SMOOTH_MAX_GAP = 1.0
# Calibrado con voces reales (piezas de 1-3 s): 0.6 separa de más a un mismo hablante; 0.75 mezcla voces
SPEAKER_DIST_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.7"))

_embedder = None
_embedder_lock = threading.Lock()


class SpeakerEmbedder:
    def __init__(self, model_path):
        import onnxruntime as ort
        import torchaudio.compliance.kaldi as kaldi

        self._kaldi = kaldi
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name  # "feats" [B, T, 80]

    def embed(self, audio):
        """Embedding L2-normalizado (256,) de un array float32 mono a 16 kHz."""
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        feats = self._kaldi.fbank(
            wav * 32768,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=SAMPLE_RATE,
            window_type="hamming",
            use_energy=False,
        )
        feats = feats - feats.mean(dim=0, keepdim=True)  # CMN por enunciado
        feats = feats.unsqueeze(0).numpy().astype(np.float32)
        emb = self.session.run(None, {self.input_name: feats})[0][0]
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm == 0.0:
            return None
        return (emb / norm).astype(np.float32)


def get_embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(EMBED_REPO, EMBED_FILE)
            _embedder = SpeakerEmbedder(path)
    return _embedder


def embed_piece(audio, start, end, sr=SAMPLE_RATE):
    """Embedding de la pieza [start, end] (segundos relativos al array) o None si es muy corta."""
    if end - start < MIN_EMBED_SEC:
        return None
    if end - start < TARGET_EMBED_SEC:
        mid = (start + end) / 2
        start, end = mid - TARGET_EMBED_SEC / 2, mid + TARGET_EMBED_SEC / 2
    a = max(0, int(start * sr))
    b = min(len(audio), int(end * sr))
    if b - a < int(MIN_EMBED_SEC * sr):
        return None
    return get_embedder().embed(audio[a:b])


class SpeakerDiarizer:
    def __init__(self, num_speakers=None):
        self.num_speakers = num_speakers
        self.starts = []
        self.ends = []
        self.embs = []

    def add(self, start, end, embedding):
        self.starts.append(float(start))
        self.ends.append(float(end))
        self.embs.append(embedding)

    def labels(self):
        n = len(self.embs)
        if n == 0:
            return []
        starts = np.asarray(self.starts)
        ends = np.asarray(self.ends)
        durs = ends - starts
        emb_idx = [i for i, e in enumerate(self.embs) if e is not None]
        if len(emb_idx) <= 1:
            return [0] * n

        labels = self._cluster(emb_idx, durs)
        labels = self._fill_missing(labels, emb_idx, starts, ends)
        labels = self._smooth(labels, starts, ends, durs)
        return self._renumber(labels, starts)

    def _cluster(self, emb_idx, durs):
        from scipy.cluster.hierarchy import fcluster, linkage

        X = np.stack([self.embs[i] for i in emb_idx]).astype(np.float64)
        Z = linkage(X, method="average", metric="cosine")
        if self.num_speakers:
            raw = fcluster(Z, t=self.num_speakers, criterion="maxclust")
        else:
            raw = fcluster(Z, t=SPEAKER_DIST_THRESHOLD, criterion="distance")

        # Clusters diminutos (< 4 s de voz) -> cluster no diminuto más cercano (coseno al centroide)
        _, inv = np.unique(raw, return_inverse=True)
        total = np.bincount(inv, weights=durs[emb_idx])
        centroids = np.zeros((len(total), X.shape[1]))
        np.add.at(centroids, inv, X)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
        big = np.flatnonzero(total >= TINY_CLUSTER_SEC)
        tiny = np.flatnonzero(total < TINY_CLUSTER_SEC)
        if len(big) and len(tiny):
            target = np.arange(len(total))
            target[tiny] = big[np.argmax(centroids[tiny] @ centroids[big].T, axis=1)]
            inv = target[inv]

        labels = [None] * len(self.embs)
        for i, lab in zip(emb_idx, inv):
            labels[i] = int(lab)
        return labels

    @staticmethod
    def _fill_missing(labels, emb_idx, starts, ends):
        # Piezas sin embedding: etiqueta de la pieza con embedding más cercana en el tiempo
        mids = (starts + ends) / 2
        emb_mids = mids[emb_idx]
        order = np.argsort(emb_mids, kind="stable")
        sorted_mids = emb_mids[order]
        for i, lab in enumerate(labels):
            if lab is not None:
                continue
            pos = int(np.searchsorted(sorted_mids, mids[i]))
            cands = [p for p in (pos - 1, pos) if 0 <= p < len(sorted_mids)]
            best = min(cands, key=lambda p: abs(sorted_mids[p] - mids[i]))
            labels[i] = labels[emb_idx[order[best]]]
        return labels

    @staticmethod
    def _smooth(labels, starts, ends, durs):
        out = list(labels)
        for i in range(1, len(labels) - 1):
            prev_lab, next_lab = labels[i - 1], labels[i + 1]
            if (
                durs[i] < SMOOTH_MAX_SEC
                and prev_lab == next_lab != labels[i]
                and starts[i] - ends[i - 1] < SMOOTH_MAX_GAP
                and starts[i + 1] - ends[i] < SMOOTH_MAX_GAP
            ):
                out[i] = prev_lab
        return out

    @staticmethod
    def _renumber(labels, starts):
        mapping = {}
        for i in np.argsort(starts, kind="stable"):
            mapping.setdefault(labels[i], len(mapping))
        return [mapping[lab] for lab in labels]
