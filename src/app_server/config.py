"""Configuración centralizada. Sobreescribe con variables de entorno cuando convenga."""
import os

# --- Modelo ---
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "best_upeu_yolo8_100ep.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda")

# --- OCR (PaddleOCR, reconocimiento puro en GPU) ---
# PP-OCRv4_server_rec: 88.0% placa correcta en el test 250 (vs 85.6% del mobile), 0 vacíos.
# En Linux torch (YOLO) y paddle conviven en el mismo proceso, así que el OCR va in-process.
OCR_MODEL_NAME = os.getenv("OCR_MODEL_NAME", "PP-OCRv4_server_rec")
OCR_DEVICE = os.getenv("OCR_DEVICE", "gpu")  # "gpu" o "cpu"
# Corrector por formato peruano + heurística O/0 e I/1. Medido +6.0pp (88.0->94.0%) sobre el
# server model (test 250). Desactivar con OCR_APPLY_CORRECTOR=0.
OCR_APPLY_CORRECTOR = os.getenv("OCR_APPLY_CORRECTOR", "1") == "1"

# --- Cámara IP ---
RTSP_URL = os.getenv(
    "RTSP_URL",
    "rtsp://usuario:clave@IP_CAMARA:554/Streaming/Channels/1",
)

# --- Servidor ---
API_URL = os.getenv("API_URL", "https://tu-servidor.example.com/api/plates")
API_TOKEN = os.getenv("API_TOKEN", "")
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "5"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
# Tope de la cola en RAM del Sender. Si se llena (p. ej. el servidor está caído y
# las detecciones entran más rápido de lo que se envían), los envíos nuevos se
# persisten directo a disco en vez de acumularse en memoria. 0 = sin límite.
SENDER_QUEUE_MAXSIZE = int(os.getenv("SENDER_QUEUE_MAXSIZE", "1000"))
# Cada cuánto el worker reintenta los JSON de failed_posts/ (además del arranque).
FAILED_REPLAY_INTERVAL_S = float(os.getenv("FAILED_REPLAY_INTERVAL_S", "60"))

# --- Umbrales ---
DETECT_CONF_THRESHOLD = float(os.getenv("DETECT_CONF_THRESHOLD", "0.60"))
OCR_SCORE_THRESHOLD = float(os.getenv("OCR_SCORE_THRESHOLD", "0.85"))

# --- Deduplicación ---
SUPPRESSION_SECONDS = int(os.getenv("SUPPRESSION_SECONDS", "10"))
RETENTION_SECONDS = int(os.getenv("RETENTION_SECONDS", "600"))
PURGE_INTERVAL_S = int(os.getenv("PURGE_INTERVAL_S", "30"))
FUZZY_SIMILARITY = float(os.getenv("FUZZY_SIMILARITY", "0.85"))
USE_FUZZY = os.getenv("USE_FUZZY", "1") == "1"

# --- Throttling / rendimiento ---
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))  # procesar 1 de cada N frames
RECONNECT_DELAY_S = float(os.getenv("RECONNECT_DELAY_S", "1.0"))
# Buffer interno de OpenCV/FFMPEG para el stream RTSP. 1 = quédate siempre con el
# frame más reciente (evita que se acumulen frames viejos y crezca la latencia/RAM
# si el procesamiento va más lento que la cámara).
CAP_BUFFER_SIZE = int(os.getenv("CAP_BUFFER_SIZE", "1"))
# Reinicia el estado del tracker de YOLO cada N frames PROCESADOS para acotar la RAM
# que acumula el tracker en corridas largas. 0 = desactivado (actívalo si ves la RAM
# crecer; la dedup por texto de placa cubre la pérdida momentánea de track_id).
TRACKER_RESET_FRAMES = int(os.getenv("TRACKER_RESET_FRAMES", "0"))

# --- Almacenamiento ---
DIR_VEHICLES = os.getenv("DIR_VEHICLES", "data_vehicles")
DIR_PLATES = os.getenv("DIR_PLATES", "data_plates")
DIR_FAILED_QUEUE = os.getenv("DIR_FAILED_QUEUE", "failed_posts")

# --- Formato de placa peruana (XXX-XXX) ---
PLATE_REGEX = r"([A-Z0-9]{3}-[A-Z0-9]{3})"
PLATE_LENGTH = 7
