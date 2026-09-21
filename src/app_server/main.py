"""Pipeline de detección de placas: RTSP -> YOLO -> OCR -> dedup -> POST async."""
import logging
import os
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path

# Silenciar verbosidad de librerías ANTES de importarlas
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_logtostderr", "0")
warnings.filterwarnings("ignore")

# Habilitar secuencias ANSI en consolas de Windows (no-op en Linux)
if os.name == "nt":
    os.system("")

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"


def color_for(score: float) -> str:
    """Verde >0.95, amarillo 0.90-0.95, rojo <0.90."""
    if score > 0.95:
        return ANSI_GREEN
    if score >= 0.90:
        return ANSI_YELLOW
    return ANSI_RED


def colorize(score: float, fmt: str = "{:.2f}") -> str:
    return f"{color_for(score)}{fmt.format(score)}{ANSI_RESET}"

import cv2 as cv
import torch
from ultralytics import YOLO

import config
from dedup import Deduper, normalize_plate, safe_filename
from ocr import Ocr
from sender import Sender


def setup_logging():
    """Sólo nuestros mensajes; silenciamos a terceros."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("plates").setLevel(logging.INFO)
    logging.getLogger("sender").setLevel(logging.INFO)
    for noisy in ("ultralytics", "ppocr", "paddle", "urllib3", "requests", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


setup_logging()
log = logging.getLogger("plates")


def month_folder(base: str) -> Path:
    folder = Path(base) / datetime.now().strftime("%Y-%m")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def open_capture():
    cap = cv.VideoCapture(config.RTSP_URL, cv.CAP_FFMPEG)
    # Buffer mínimo: quédate con el frame más reciente y evita que se acumulen
    # frames viejos (latencia/RAM crecientes) si el procesamiento va más lento.
    if config.CAP_BUFFER_SIZE > 0:
        cap.set(cv.CAP_PROP_BUFFERSIZE, config.CAP_BUFFER_SIZE)
    if not cap.isOpened():
        log.error("No se pudo abrir el stream RTSP")
    return cap


def reset_tracker(model):
    """Reinicia el estado interno del tracker de YOLO (acota RAM en corridas largas)."""
    try:
        for tracker in getattr(model.predictor, "trackers", []) or []:
            tracker.reset()
    except Exception as e:  # versión sin trackers o aún sin inicializar
        log.warning("No se pudo reiniciar el tracker: %s", e)


def extract_detections(detection):
    """Devuelve lista de (x1, y1, x2, y2, conf, track_id)."""
    boxes = detection.boxes
    if boxes is None or boxes.xyxy is None or boxes.conf is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(confs)
    out = []
    for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
        out.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(tid) if tid is not None else None))
    return out


def crop_roi(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def save_images(frame, plate_roi, plate_text, plate_score, detect_score):
    ts = int(time.time())
    fh, fw = frame.shape[:2]
    ph, pw = plate_roi.shape[:2]
    safe = safe_filename(plate_text)
    score_str = f"{detect_score:.2f}"
    full = month_folder(config.DIR_VEHICLES) / f"{ts}_{score_str}_{fw}x{fh}_{safe}.jpg"
    crop = month_folder(config.DIR_PLATES) / f"{ts}_{score_str}_{pw}x{ph}_{safe}.jpg"
    cv.imwrite(str(full), frame)
    cv.imwrite(str(crop), plate_roi)
    return str(crop)


def main():
    model = YOLO(config.YOLO_WEIGHTS).to(config.YOLO_DEVICE)
    ocr = Ocr()
    sender = Sender()
    dedup = Deduper()

    cap = open_capture()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(config.RECONNECT_DELAY_S)
                cap = open_capture()
                continue

            frame_idx += 1
            if config.FRAME_SKIP > 1 and (frame_idx % config.FRAME_SKIP) != 0:
                continue

            if config.TRACKER_RESET_FRAMES > 0 and (frame_idx % config.TRACKER_RESET_FRAMES) == 0:
                reset_tracker(model)

            results = model.track(frame, stream=False, verbose=False, persist=True)

            for detection in results:
                for x1, y1, x2, y2, det_conf, track_id in extract_detections(detection):
                    if det_conf < config.DETECT_CONF_THRESHOLD:
                        continue

                    plate_roi = crop_roi(frame, x1, y1, x2, y2)
                    if plate_roi is None:
                        continue

                    np_text, np_score = ocr.read_license_plate(plate_roi)
                    if np_text is None or np_score < config.OCR_SCORE_THRESHOLD:
                        continue

                    plate_norm = normalize_plate(np_text)
                    if len(plate_norm) != config.PLATE_LENGTH:
                        continue

                    emit = dedup.should_emit(plate_norm, track_id)
                    # Refresca "última vez visto" SIEMPRE (aunque se suprima), para
                    # que mientras el carro siga en cuadro la ventana no expire y NO
                    # se reenvíe la misma placa. Sin esto, un carro detenido más de
                    # SUPPRESSION_SECONDS se emitía 2-3 veces.
                    dedup.register(plate_norm, track_id)
                    if not emit:
                        continue

                    save_images(frame, plate_roi, plate_norm, np_score, det_conf)
                    dimension = f"{plate_roi.shape[1]}x{plate_roi.shape[0]}"
                    veh_score = int(plate_roi.size)
                    sender.enqueue({
                        # Clave de idempotencia: única por detección. Viaja igual en
                        # cada reintento y en los replays desde failed_posts/, para que
                        # el servidor deduplique y NO inserte la misma placa varias veces
                        # aunque el POST se repita (p. ej. tras un 500 que ya insertó).
                        "event_id": uuid.uuid4().hex,
                        "number": plate_norm,
                        "ocrscore": float(np_score),
                        "platescore": round(float(det_conf), 4),
                        "vehiclescore": veh_score,
                        "dimension": dimension,
                    })
                    log.info(
                        "DETECTADA  %s   ocr=%s  placa=%s  vehiculo=%s",
                        plate_norm,
                        colorize(float(np_score)),
                        colorize(float(det_conf)),
                        f"{veh_score:.2f}",
                    )
    except KeyboardInterrupt:
        log.info("Interrumpido por usuario")
    finally:
        cap.release()
        cv.destroyAllWindows()
        ocr.close()
        sender.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
