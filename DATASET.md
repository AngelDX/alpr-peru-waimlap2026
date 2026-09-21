# Conjuntos de datos — origen, creador, licencia y acceso

Este repositorio incluye **descriptores** de los datos (archivos `.yaml`, `build_report.json`
y las **listas de partición** en `experiments/crossdomain/splits/`), pero **no** incluye las
imágenes. A continuación se explica el origen de cada conjunto, su creador, la licencia y cómo
obtenerlo.

---

## 1. Conjunto propio dentro del dominio (in-domain) — placas peruanas, cámara fija

- **Creador:** Angel R. Condori Coaquira y coautores, Universidad Peruana Unión (UPeU),
  Juliaca, Perú. **Debe atribuirse a los autores** en cualquier uso.
- **Descripción:** imágenes de vehículos con placa peruana capturadas durante seis meses con
  una cámara IP fija (HIKVISION, 1280×720) en un punto de acceso vehicular, bajo condiciones
  no controladas (día, noche, lluvia, viento con polvo). Anotación manual de una sola clase
  (`placa`) en Roboflow.
- **Ubicación (Roboflow):** espacio de trabajo `dataset1-zxf4z`, proyecto `test2-c07ld`.
- **Licencia / acceso — BAJO SOLICITUD PREVIA:** las imágenes contienen matrículas
  (dato personal), por lo que **no se publican abiertamente**. Se comparten **bajo solicitud
  previa razonable** dirigida al autor de correspondencia
  (**angelrc2@upeu.edu.pe**), sujeto a un acuerdo de **uso responsable** que exige:
  1. uso exclusivamente académico/de investigación;
  2. no redistribución de las imágenes a terceros sin autorización;
  3. no intento de reidentificación de personas o propietarios;
  4. **cita del artículo y atribución a los autores como creadores del conjunto**.
  Para solicitarlo, escribir indicando afiliación, propósito de investigación y aceptación de
  estas condiciones.

## 2. Conjunto genérico de arranque (bootstrap) — público

- **Creador:** **VehicleDetection** — *ALPR Dataset (License Plate Recognition)*,
  Roboflow Universe, 2024.
- **Licencia:** **CC BY 4.0** (uso permitido con atribución al creador).
- **Uso en este trabajo:** entrenar un primer modelo genérico que permitió detectar y recopilar
  de forma automatizada las placas peruanas para construir el conjunto propio.
- **Enlace:** https://universe.roboflow.com/vehicledetection-tj7kn/alpr-eypnb
- **Atribución requerida:** al reutilizarlo, citar a VehicleDetection (Roboflow Universe, CC BY 4.0).

## 3. Conjunto público fuera de dominio (out-domain) — placas peruanas, móvil/web

- **Nombre:** *Detección de Placas Peruanas Dataset*.
- **Creador:** **Proyecto de Tesis** (Roboflow Universe). **Debe atribuirse a su autor.**
- **Licencia:** **CC BY 4.0** (uso permitido con atribución).
- **Descripción:** 6 813 imágenes de placas peruanas (clases `placa_carro`, `placa_moto`)
  capturadas con teléfonos móviles y cámaras web (dispositivos y perspectivas distintos de la
  cámara fija). Se usa **solo para la evaluación de generalización** (test independiente
  `test_movil`) y como parte del corpus fusionado.
- **Enlace:** https://universe.roboflow.com/proyecto-de-tesis/deteccion-de-placas-peruanas-ybiq9

BibTeX (atribución requerida):

```bibtex
@misc{ deteccion-de-placas-peruanas-ybiq9_dataset,
  title  = { Detección de Placas Peruanas Dataset },
  type   = { Open Source Dataset },
  author = { Proyecto de Tesis },
  howpublished = { \url{ https://universe.roboflow.com/proyecto-de-tesis/deteccion-de-placas-peruanas-ybiq9 } },
  url     = { https://universe.roboflow.com/proyecto-de-tesis/deteccion-de-placas-peruanas-ybiq9 },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year   = { 2026 },
  month  = { feb },
  note   = { visited on 2026-09-15 },
}
```

---

## 4. Corpus fusionado y protocolo de partición

El corpus fusionado combina el conjunto in-domain (cámara) con el out-domain (móvil/web) para
estudiar la generalización. Detalles reproducibles en `data/build_report.json`:

- **Semilla fija:** 42. **Anonimización:** datos del propietario difuminados.
- **Muestreo por placa única (no por fotograma):** evita que fotogramas casi idénticos de una
  misma secuencia de captura de un vehículo caigan en particiones distintas (elimina el sesgo
  por imágenes correlacionadas).
- **Particiones (cámara):** de 23 943 / 2 993 / 2 992 placas únicas disponibles se muestrearon
  **3 000 / 600 / 1 500** imágenes (entrenamiento / validación / prueba in-domain). El tamaño
  del área de placa (px²) de cada partición se ajustó a la distribución de referencia del corpus
  (mediana ≈ 5 184 px²).
- **Particiones (móvil):** 1 508 / 323 (entrenamiento / validación) + **323** de prueba
  out-domain independiente.
- **Dos test sets independientes** que nunca se ven en entrenamiento: in-domain (1 500,
  cámara fija) y out-domain (323, móvil/web).

### Listas de partición

`experiments/crossdomain/splits/` contiene las listas de imágenes por condición, con rutas
relativas a la raíz del dataset:

| Archivo | Contenido |
|---|---|
| `train_camonly.txt` / `val_camonly.txt` | Solo cámara (prefijo `cam_`): 3 000 / 600 |
| `train_fused.txt` / `val_fused.txt` | Fusionado (cámara + móvil): 4 508 / 923 |

Los `.yaml` de `experiments/crossdomain/` referencian estas listas y los test sets. Ajusta la
clave `path:` de cada `.yaml` a la ruta local donde descargues las imágenes.

---

## Cómo obtener las imágenes

1. **In-domain:** solicítalo al autor de correspondencia (ver punto 1) y expórtalo desde
   Roboflow en formato YOLO.
2. **Genérico (CC BY 4.0):** descárgalo del enlace del punto 2.
3. Coloca las imágenes/labels respetando la estructura `train/`, `valid/`, `test_camara/`,
   `test_movil/` (cada una con `images/` y `labels/`) y apunta `path:` en los `.yaml`.
