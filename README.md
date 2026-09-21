# ALPR en tiempo real para placas peruanas — WAIMLAP 2026

Sistema ALPR (reconocimiento automático de placas vehiculares) en tiempo real para
videovigilancia urbana, orientado a placas peruanas. El flujo completo es
**cámara IP (RTSP) → detección YOLO → OCR (PaddleOCR) → deduplicación → envío asíncrono →
almacenamiento y visualización en tiempo real (Laravel + WebSockets)**.

Este repositorio acompaña al artículo:

> A. R. Condori Coaquira, G. Quea Lopez, D. Mamani-Pari, M. E. Humpiri Flores y E. Tocto Cano,
> *"Sistema ALPR en tiempo real para videovigilancia urbana: detección y reconocimiento de
> matrículas de vehículos peruanos con YOLO y PaddleOCR"*, WAIMLAP 2026, Lima, Perú.
> Springer (serie Proceedings).

Contiene el **código**, los **descriptores de los conjuntos de datos**, los **scripts de
experimentos** y los **resultados** necesarios para reproducir los hallazgos del artículo.
Las **imágenes** de los datasets no se incluyen aquí (ver [`DATASET.md`](DATASET.md)): el
conjunto propio se comparte **bajo solicitud previa razonable** y el conjunto genérico se
descarga de su fuente pública con su licencia.

---

## Estructura del repositorio

```
alpr-peru-waimlap2026/
├── README.md                 Este archivo
├── DATASET.md                Datasets: origen, creador, licencia y acceso bajo solicitud
├── LICENSE                   Licencia del CÓDIGO (MIT)
├── CITATION.cff              Cómo citar este trabajo
├── requirements.txt          Dependencias consolidadas
├── src/
│   ├── app_server/           Pipeline de producción (RTSP→YOLO→OCR→dedup→POST)
│   └── dataset_collector/    Variante que además recolecta "hard cases" + corrector OCR
├── experiments/
│   ├── crossdomain/          Experimento controlado de generalización (in/out-domain)
│   └── end_to_end/           Evaluación extremo a extremo (latencia y exactitud en campo)
└── results/
    ├── crossdomain/          Métricas JSON de todos los modelos evaluados
    ├── training_curves/      results.csv, curvas PR y matrices de confusión
    └── end_to_end/           Resultados de la sesión real (854 vehículos)
```

---

## Requisitos

- Python 3.10–3.12, GPU NVIDIA con CUDA.
- **PyTorch (CUDA)** y **PaddlePaddle-GPU** se instalan primero desde sus índices oficiales
  (no viven en PyPI con build de GPU). Luego el resto:

```bash
# 1) PyTorch CUDA (elige la rueda para tu GPU; ej. CUDA 12.8 para Blackwell/RTX 50xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) PaddlePaddle-GPU (según tu versión de CUDA; ver https://www.paddlepaddle.org.cn)
pip install paddlepaddle-gpu
# 3) Resto de dependencias
pip install -r requirements.txt
```

> GPUs Blackwell (RTX 50xx, sm_120) requieren `torch>=2.7` con ruedas CUDA 12.8+.

---

## Reproducir los resultados

### 1) Detección: selección de versión de YOLO (in-domain)

Se entrenaron YOLOv8n/10n/11n/12n con transferencia desde COCO e hiperparámetros idénticos
(ver `results/training_curves/*/args.yaml`). Se **adoptó YOLOv8n** para el despliegue por el
mejor tiempo de inferencia (8,61 ms; 117 FPS) con exactitud estadísticamente indistinguible
del mejor modelo.

### 2) Generalización entre dominios (experimento controlado)

Cierra el sesgo de evaluación in-domain: **misma arquitectura (YOLOv8n), solo cambian los
datos** de entrenamiento (solo cámara vs. corpus fusionado). Desde `experiments/crossdomain/`
(ajusta `path:` en los `.yaml` a tu copia local del dataset):

```bash
python run_crossdomain.py --train --arch yolov8n.pt --data camonly.yaml --name yolov8n_camonly
python run_crossdomain.py --train --arch yolov8n.pt --data fused.yaml   --name yolov8n_fused
# o, si ya tienes pesos: --weights ruta/best.pt --name <nombre>
```

Cada corrida evalúa en los dos test sets independientes (`test_camara.yaml`, `test_movil.yaml`)
y escribe `crossdomain_<name>.json` con el mismo esquema que los de `results/crossdomain/`.

**Resultado principal (YOLOv8n, misma arquitectura):**

| Entrenamiento | mAP@50-95 in-domain | mAP@50-95 out-domain | Retención |
|---|---|---|---|
| Solo cámara | 0,942 | 0,143 | **15,1 %** |
| Fusionado   | 0,932 | 0,598 | **64,1 %** |

La retención fuera de dominio sube 4,2× al fusionar el conjunto público, mientras el
desempeño in-domain se conserva. Como la arquitectura e hiperparámetros son idénticos, la
mejora es atribuible exclusivamente a la fusión de datos.

### 3) OCR y corrector de formato peruano

`src/dataset_collector/ocr.py` implementa el corrector determinista basado en el formato de
placa peruano (letra en posición 1, dígitos en 4–6, desambiguación O/0 e I/1 por posición).
Sobre un conjunto independiente de 250 placas, la coincidencia exacta sube del **88,0 % al
94,0 %** (+6,0 pp).

### 4) Evaluación extremo a extremo

`experiments/end_to_end/medir_e2e.py` cruza la verdad de terreno (placa + hora de la cámara)
contra la exportación de la tabla `plates` de MySQL, calcula exactitud y latencia de registro
(usa `verdad_plantilla.csv` como plantilla). Resultados de la sesión real (854 vehículos) en
`results/end_to_end/`.

---

## Datos

El detalle completo de origen, creador, licencia y **acceso bajo solicitud previa** está en
[`DATASET.md`](DATASET.md). En resumen:

- **Conjunto propio (in-domain)** — placas peruanas capturadas con cámara IP fija. Creado por
  los autores (Universidad Peruana Unión, Juliaca). Contiene datos personales (matrículas), por
  lo que se comparte **bajo solicitud previa razonable** al autor de correspondencia y bajo un
  acuerdo de uso responsable. Roboflow: `dataset1-zxf4z/test2-c07ld`.
- **Conjunto genérico (bootstrap)** — *ALPR Dataset* de **VehicleDetection** (Roboflow Universe,
  **CC BY 4.0**), usado para arrancar la detección. Debe citarse a su creador.
- **Conjunto público fuera de dominio (out-domain)** — *Detección de Placas Peruanas* de
  **Proyecto de Tesis** (Roboflow Universe, **CC BY 4.0**), usado para la evaluación de
  generalización. https://universe.roboflow.com/proyecto-de-tesis/deteccion-de-placas-peruanas-ybiq9

---

## Cómo citar

Si usas este código o los datos, cita el artículo (ver [`CITATION.cff`](CITATION.cff)) y, para
el dataset genérico, atribuye a **VehicleDetection (Roboflow Universe, CC BY 4.0)**.

## Licencia

- **Código:** MIT (ver [`LICENSE`](LICENSE)).
- **Datos:** cada conjunto conserva su propia licencia/condiciones (ver [`DATASET.md`](DATASET.md)).

## Autores y contacto

Universidad Peruana Unión — Escuela Profesional de Ingeniería de Sistemas, Juliaca, Perú.
Autor de correspondencia: **Angel R. Condori Coaquira** — angelrc2@upeu.edu.pe
