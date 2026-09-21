# Detección de placas — app de servidor (Linux + GPU)

Pipeline: **RTSP → YOLO (detección) → PaddleOCR (reconocimiento) → dedup → POST**.

OCR con **PP-OCRv4_server_rec** + corrector de formato peruano/heurística O-0 e I-1 = **94.0%
placa correcta** en el test de 250 (vs 88.0% sin corrector, 85.6% del modelo mobile; 0 lecturas
vacías). En Linux torch (YOLO) y paddle conviven en un mismo proceso, así que el OCR corre
**in-process** (sin worker ni subprocesos).

El corrector (`correct_plate6` en `ocr.py`) aplica el formato `[L][A][A]-[N][N][N]` (pos1
letra, pos4-6 dígito) y la heurística medida sobre SUNARP: la confusión O/0 e I/1 en pos2 es
siempre el carácter cero/uno (100%) y en pos3 es la letra O/I (~82-85%).

## Entorno objetivo (verificado en `server_info`)

- Ubuntu 22.04, Python 3.10+
- GPU NVIDIA **RTX 3080** (Ampere, sm_86), driver 550.x (**CUDA 12.4**)
- `nvidia-smi` debe responder

## Instalación

```bash
cd app_server
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 1) PyTorch con CUDA 12.4 (coincide con el driver 550; sirve para YOLO en GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) PaddlePaddle GPU 3.3.1 (runtime CUDA 12.6; corre sobre el driver 12.4 por
#    compatibilidad de versión menor de CUDA). Trae kernels sm_86 (Ampere).
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 3) Resto de dependencias de la app
pip install -r requirements.txt
```

> Si paddle se queja del driver (CUDA demasiado nuevo para el driver), usa la rueda cu118
> como alternativa conservadora:
> `pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/`

### Verificar GPU

```bash
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import paddle; print('paddle', paddle.__version__, paddle.is_compiled_with_cuda()); paddle.utils.run_check()"
```

## Archivos necesarios (copiar al servidor)

- **`best_upeu_yolo12_100ep.pt`** — pesos de YOLO (ponerlos junto a `main.py` o ajustar `YOLO_WEIGHTS`).
- El modelo de OCR (`PP-OCRv4_server_rec`) se **descarga solo** la primera vez a `~/.paddlex/official_models/`.

## Configuración

Todo se ajusta por variables de entorno (ver `config.py`). Las más comunes:

| Variable | Defecto | Descripción |
|---|---|---|
| `RTSP_URL` | (cámara UPEU) | URL del stream RTSP |
| `API_URL` | `https://tu-servidor.example.com/api/plates` | Endpoint de envío |
| `API_TOKEN` | (vacío) | Bearer token si aplica |
| `YOLO_WEIGHTS` | `best_upeu_yolo12_100ep.pt` | Pesos YOLO |
| `YOLO_DEVICE` | `cuda` | Dispositivo de YOLO |
| `OCR_MODEL_NAME` | `PP-OCRv4_server_rec` | Modelo de reconocimiento |
| `OCR_DEVICE` | `gpu` | `gpu` o `cpu` |
| `OCR_APPLY_CORRECTOR` | `1` | Corrector formato peruano + heurística O/0 e I/1 (+6.0pp). `0` para desactivar |
| `DETECT_CONF_THRESHOLD` | `0.60` | Confianza mínima de detección |
| `OCR_SCORE_THRESHOLD` | `0.85` | Score mínimo de OCR para emitir |
| `FRAME_SKIP` | `2` | Procesa 1 de cada N frames |
| `OCR_DEVICE` | `gpu` | Acepta índice: `gpu:1` para la 2ª GPU |
| `CAP_BUFFER_SIZE` | `1` | Buffer RTSP; `1` = quédate con el frame más reciente (evita latencia/RAM creciente) |
| `SENDER_QUEUE_MAXSIZE` | `1000` | Tope de la cola de envío en RAM; al llenarse persiste a disco. `0` = sin límite |
| `FAILED_REPLAY_INTERVAL_S` | `60` | Cada cuánto el worker reintenta `failed_posts/` |
| `TRACKER_RESET_FRAMES` | `0` | Reinicia el tracker YOLO cada N frames (acota RAM en corridas largas). `0` = off |

Ejemplo:

```bash
export RTSP_URL="rtsp://usuario:clave@IP/Streaming/Channels/1"
export API_TOKEN="..."
python main.py
```

## Ejecutar

```bash
source .venv/bin/activate
python main.py
```

Salida: una línea `DETECTADA  ABC-123  ocr=0.99 ...` por placa nueva; los envíos van en un thread
aparte con reintentos y cola persistente en `failed_posts/` (se reintentan al arrancar).

## Selección de GPU (servidor con 2 GPUs)

El pipeline de **una sola cámara** es secuencial (YOLO → OCR, frame por frame), así que repartir
ese stream en 2 GPUs **no sube el throughput**: las etapas se ejecutan una tras otra. Las 2 GPUs
sirven para (a) aterrizar en la GPU más libre, (b) repartir la VRAM, o (c) correr varias cámaras.

Primero, mira cuál está más libre:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

### A) Usar solo la 2ª GPU (la más libre) — recomendado

Expón únicamente la GPU 1 al proceso; torch y paddle la ven como la única GPU y los defaults
(`cuda` / `gpu`) funcionan sin cambios:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py
```

### B) Repartir la VRAM: YOLO en una GPU, OCR en otra

Con índices explícitos (no uses `CUDA_VISIBLE_DEVICES` aquí, o tapas la otra GPU):

```bash
# todo en la GPU 1
YOLO_DEVICE=cuda:1  OCR_DEVICE=gpu:1  python main.py

# separados: detección en GPU 0, OCR en GPU 1
YOLO_DEVICE=cuda:0  OCR_DEVICE=gpu:1  python main.py
```

### C) Aprovechar las 2 GPUs de verdad: una cámara por GPU

Un proceso por cámara, cada uno fijado a una GPU (escala el throughput casi lineal):

```bash
CUDA_VISIBLE_DEVICES=0  RTSP_URL="rtsp://cam1..."  python main.py
CUDA_VISIBLE_DEVICES=1  RTSP_URL="rtsp://cam2..."  python main.py
```

> `CUDA_VISIBLE_DEVICES` **no es permanente**: vale solo para esa terminal. Para que sobreviva
> reinicios, fíjalo en el servicio systemd (`Environment=...`, abajo) o en tu script de arranque.

## Servicio systemd (arranque permanente + GPU fija)

`/etc/systemd/system/placas.service`:

```ini
[Unit]
Description=Deteccion de placas (app_server)
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/ruta/a/app_server
Environment=CUDA_VISIBLE_DEVICES=1
# Opcionales: pega aquí las que necesites de la tabla de Configuración
# Environment=YOLO_DEVICE=cuda:1
# Environment=OCR_DEVICE=gpu:1
# Environment=API_TOKEN=...
# Environment=RTSP_URL=rtsp://usuario:clave@IP/Streaming/Channels/1
ExecStart=/ruta/a/app_server/.venv/bin/python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now placas.service
sudo journalctl -u placas.service -f      # ver logs en vivo
```
