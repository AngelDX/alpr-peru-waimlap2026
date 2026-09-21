"""Deduplicación de placas: prioriza track_id de YOLO; fallback por texto fuzzy."""
import difflib
import re
import time

import config


_NORMALIZE_RE = re.compile(r"[^A-Z0-9\-]")


def normalize_plate(text: str) -> str:
    if not text:
        return ""
    return _NORMALIZE_RE.sub("", text.upper().strip())


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)


class Deduper:
    """Suprime placas repetidas mientras el vehículo siga presente.

    La ventana de supresión se mide desde la ÚLTIMA VEZ QUE SE VIO la placa, no
    desde la última vez que se emitió: cada detección válida refresca el timestamp
    (`register`), aunque esa detección se suprima. Así un carro que se detiene en
    cuadro durante más de SUPPRESSION_SECONDS se emite UNA sola vez (la ventana solo
    expira cuando la placa estuvo AUSENTE SUPPRESSION_SECONDS).

    Estrategia:
      1) Si la detección trae track_id estable, basta con ese id para suprimir.
      2) Si no hay track_id (o cambió), comparar la placa normalizada y, opcionalmente,
         usar similitud fuzzy para tolerar pequeñas variaciones del OCR.
    """

    def __init__(self):
        self._tracks: dict[int, float] = {}   # track_id -> last_seen_ts
        self._plates: dict[str, float] = {}   # plate_norm -> last_seen_ts
        self._last_purge = 0.0

    def should_emit(self, plate_norm: str, track_id: int | None) -> bool:
        now = time.time()
        self._maybe_purge(now)

        if track_id is not None:
            last = self._tracks.get(track_id)
            if last is not None and now - last < config.SUPPRESSION_SECONDS:
                return False

        if plate_norm:
            last = self._plates.get(plate_norm)
            if last is not None and now - last < config.SUPPRESSION_SECONDS:
                return False
            if config.USE_FUZZY:
                for seen, ts in self._plates.items():
                    if now - ts >= config.SUPPRESSION_SECONDS:
                        continue
                    if difflib.SequenceMatcher(None, plate_norm, seen).ratio() >= config.FUZZY_SIMILARITY:
                        return False
        return True

    def register(self, plate_norm: str, track_id: int | None):
        """Refresca la 'última vez visto'. Llamar en CADA detección válida (emitida o
        suprimida) para que la ventana no expire mientras el carro siga en cuadro."""
        now = time.time()
        if track_id is not None:
            self._tracks[track_id] = now
        if plate_norm:
            self._plates[plate_norm] = now

    def _maybe_purge(self, now: float):
        if now - self._last_purge < config.PURGE_INTERVAL_S:
            return
        self._last_purge = now
        cutoff = now - config.RETENTION_SECONDS
        self._tracks = {k: v for k, v in self._tracks.items() if v >= cutoff}
        self._plates = {k: v for k, v in self._plates.items() if v >= cutoff}
