# Resultados

## 1. Generalización entre dominios (`crossdomain/`)

Cada `crossdomain_<modelo>.json` reporta el desempeño del mismo modelo en dos conjuntos de
prueba independientes: **in-domain** (`test_camara`, 1500 img, cámara fija) y **out-domain**
(`test_movil`, 323 img, móvil/web). `retencion_mAP50_95 = out-domain / in-domain`.

### Experimento controlado (misma arquitectura YOLOv8n — cierra el confound)

| Modelo (JSON) | Entrenamiento | mAP@50-95 in | mAP@50-95 out | Retención |
|---|---|---|---|---|
| `crossdomain_yolov8n_camonly` | Solo cámara | 0,942 | 0,143 | 15,1 % |
| `crossdomain_yolov8n_fused`   | Fusionado   | 0,932 | 0,598 | **64,1 %** |

Misma arquitectura e hiperparámetros; solo cambian los datos → la mejora (4,2×) es atribuible
a la fusión del conjunto público.

### Corridas adicionales (referencia)

| Modelo (JSON) | Entrenamiento | mAP@50-95 in | mAP@50-95 out | Retención |
|---|---|---|---|---|
| `crossdomain_yolov8s_camara`      | Solo cámara (YOLOv8s) | 0,908 | 0,157 | 17,3 % |
| `crossdomain_yolo11n_mobile`      | Fusionado (YOLO11n)   | 0,915 | 0,592 | 64,7 % |
| `crossdomain_yolo11n_mobile_onnx` | Fusionado (YOLO11n, ONNX) | 0,906 | 0,575 | 63,5 % |

## 2. Curvas de entrenamiento (`training_curves/`)

Para cada corrida controlada (`yolov8n_camonly`, `yolov8n_fused`): `results.csv` (métricas por
época), `args.yaml` (configuración exacta), curva PR y matriz de confusión normalizada.

## 3. Evaluación extremo a extremo (`end_to_end/`)

`Resultados_end_to_end_13jul.md`: sesión real de un día (854 vehículos) — 99,8 % de lecturas
con formato válido, exactitud de lectura 97,7 % sobre muestra verificada, latencia de registro
≈ 1–2 s. Reproducible con `experiments/end_to_end/medir_e2e.py`.
