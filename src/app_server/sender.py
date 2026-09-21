"""Envío asíncrono al endpoint con reintentos y cola persistente en disco."""
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import requests

import config

log = logging.getLogger(__name__)


class Sender:
    """Encola POSTs y los envía desde un thread worker. Persiste fallos en disco."""

    _STOP = object()

    def __init__(self):
        maxsize = config.SENDER_QUEUE_MAXSIZE if config.SENDER_QUEUE_MAXSIZE > 0 else 0
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._session = requests.Session()
        if config.API_TOKEN:
            self._session.headers["Authorization"] = f"Bearer {config.API_TOKEN}"
        self._failed_dir = Path(config.DIR_FAILED_QUEUE)
        self._failed_dir.mkdir(parents=True, exist_ok=True)
        self._last_replay = 0.0
        self._stopping = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._replay_failed()  # reintento inmediato de lo que quedó pendiente de una corrida previa

    def enqueue(self, payload: dict):
        # Si la cola en RAM está llena (servidor caído, envíos más lentos que las
        # detecciones), persiste a disco en vez de bloquear o crecer sin límite.
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            log.warning("Cola del Sender llena; persistiendo a disco: %s", payload.get("number"))
            self._persist_failed(payload)

    def close(self):
        # Flag de parada + sentinela. El flag corta aunque la cola esté llena y no
        # acepte el _STOP (put_nowait best-effort para no bloquear el cierre).
        self._stopping.set()
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            pass
        self._worker.join(timeout=5)

    def _run(self):
        while True:
            # get() con timeout para poder reintentar failed_posts/ aunque no entren
            # envíos nuevos (p. ej. el servidor volvió tras una caída).
            try:
                item = self._queue.get(timeout=config.FAILED_REPLAY_INTERVAL_S)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                self._maybe_replay_failed()
                continue
            if item is self._STOP:
                return
            try:
                self._send_with_retries(item)
            except Exception as e:
                log.exception("Fallo definitivo enviando placa: %s", e)
                self._persist_failed(item)
            self._maybe_replay_failed()

    def _send_with_retries(self, payload: dict):
        backoff = 1.0
        plate = payload.get("number")
        for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
            try:
                r = self._session.post(
                    config.API_URL, json=payload, timeout=config.HTTP_TIMEOUT_S
                )
                r.raise_for_status()
                body = self._format_body(r)
                log.info("ENVIADA    %s  -> %s  %s", plate, r.status_code, body)
                return
            except requests.RequestException as e:
                if attempt == config.HTTP_MAX_RETRIES:
                    log.error("ERROR      %s  -> %s", plate, e)
                    raise
                time.sleep(backoff)
                backoff *= 2

    @staticmethod
    def _format_body(response):
        try:
            return response.json()
        except ValueError:
            text = response.text.strip().replace("\n", " ")
            return text[:200]

    def _persist_failed(self, payload: dict):
        fname = self._failed_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
        try:
            fname.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.error("No se pudo persistir el envío fallido: %s", e)

    def _maybe_replay_failed(self):
        now = time.time()
        if now - self._last_replay < config.FAILED_REPLAY_INTERVAL_S:
            return
        self._last_replay = now
        self._replay_failed()

    def _replay_failed(self):
        for path in sorted(self._failed_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                log.warning("No se pudo releer %s: %s", path, e)
                continue
            # Si la cola está llena, deja el JSON en disco y reintenta en la próxima ronda.
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                break
            try:
                os.remove(path)
            except OSError as e:
                log.warning("No se pudo borrar %s tras reencolar: %s", path, e)
