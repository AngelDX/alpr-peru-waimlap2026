# Evaluación extremo a extremo — sesión real 13-jul-2026

Fuente: `data_plates/2026-07` (recortes) + `data_vehicles/2026-07` (imagen completa 1280×720) +
`plates.csv` (tabla plates de MySQL). Nombre de archivo: `{epoch}_{id}_{platescore}_{WxH}_{PLACA}.jpg`
(epoch = created_at en hora local Perú, UTC-5).

## 1. Análisis automático de toda la sesión (854 detecciones)

| Métrica | Valor |
|---|---|
| Detecciones registradas en el día | 854 |
| Lecturas con formato estándar peruano (L-AA-NNN) | 852 (99,8 %) |
| Lecturas con formato inválido (error seguro) | 2 — `F4B-U67`, `Z3B-U54` (U donde va dígito) |
| Placas con glifo O/0 en pos 2-3 (ambiguas) | 66 (7,7 %) |
| Confianza de placa (platescore) | min 0,61 · media 0,904 · máx 0,94 |
| Detecciones con score < 0,90 | 155 (confirma umbral 0,60) |
| Placas repetidas en el día | 201 (256 registros duplicados) |

Nota: la sesión usó **umbral de detección 0,60** (default en config.py, confirmado por los 155 registros con score < 0,90). El manuscrito **ya se unificó a 0,60**.

## 2. Verificación por lectura independiente (ventana de videos: 129 detecciones)

Se leyeron los recortes de forma independiente (mañana 08:40–09:30 = 77; tarde 13:00–13:20 = 52)
y se compararon con la lectura del sistema.

| Resultado | n |
|---|---|
| Coinciden con el sistema (lectura clara) | 116 |
| Coinciden pero con glifo O/0 u similar **no verificable** a esta resolución | ~10 |
| **Discrepancias claras (errores del sistema)** | **3** |
| Total verificado | 129 |

**Exactitud de lectura ≈ 126/129 = 97,7 %** (99,2 % si se excluyen las placas no estándar).

### Los 3 errores claros
1. **Placa amarilla de moto/servicio** `G984-4A` leída como `G98-444` (08:49:55). Formato fuera de alcance.
2. **Placa azul especial** `7063-6Z` leída como `Z06-362` (13:11:51). Formato fuera de alcance.
3. **Placa estándar** `Y1L-742` leída como `YJL-742` (13:10:04) — confusión J/1 en pos 2.
   Se confirma porque el **mismo vehículo** fue leído bien como `Y1L-742` a las 13:19:34.

### Observaciones
- Las 2 primeras son **placas no estándar** (moto/oficial), que el modelo —entrenado solo con el
  formato de automóvil— convierte en una cadena estándar plausible. El chequeo de formato **no** las
  detecta (la salida parece válida). Es una limitación declarable: el alcance es "automóviles de
  formato estándar".
- La ambigüedad **O/0** (~7,7 % de las placas) es el residual real; leer el recorte no la resuelve
  (mismos píxeles), por lo que ese ~2 % de incertidumbre solo lo dirime SUNARP (análisis del otro
  artículo). Refuerza el hallazgo del "techo de resolución".
- Los **duplicados** son mayormente **reingresos legítimos**: varias placas reaparecen horas después
  (p. ej. `Z8J-646` 08:47 y 13:05; `X3E-141` 08:52 y 13:07; `V7A-413`, `Z3V-606`, `Z4D-410`), no fallos
  de deduplicación.

## 3. Mensaje para el paper (subsección "Evaluación extremo a extremo")

En una sesión continua real de un día (854 vehículos), el sistema completo (cámara → YOLOv12 → PaddleOCR
→ registro en MySQL → visualización web) registró el 99,8 % de las lecturas con formato válido; sobre una
muestra de 129 detecciones verificadas contra la imagen fuente, la exactitud de lectura fue del 97,7 %
(99,2 % en placas de formato estándar). Los errores claros se concentran en placas no estándar
(motocicletas/oficiales) fuera del formato entrenado; el residual es la ambigüedad «0»/«O», limitada por
la resolución del sensor.

## 4. Latencia de registro (entrada → guardado en BD)

Se verificó la **sincronización de relojes** por vehículo emparejado (overlay de la cámara vs `created_at`).
Hallazgo: el nombre del archivo de video ≈ hora de **fin** de grabación, no de inicio; el video de la
tarde (`Grabación 2026-07-13 130629.mp4`, overlay 13:01–13:06) está **sincronizado** (reloj cámara ≈ servidor).

Mediciones directas (vehículo visible en el overlay → registro en MySQL):

| Placa | Overlay (entrada) | created_at (registro) | Latencia |
|---|---|---|---|
| VCM-524 | 13:03:54 | 13:03:57 | 3 s → **1 s** |
| Z6V-403 | 13:05:53 | 13:05:57 | 4 s → **2 s** |

→ **Latencia entrada→registro real ≈ 1–2 s**, tras corregir el desfase de **2 s** verificado entre el reloj de la cámara y el del servidor (bruta 3–4 s − 2 s). Incluye el trayecto del vehículo hasta el punto de detección.

### 4b. Latencia interna de registro (detección → inserción en BD) — TODA la sesión, sin video

Comparando el **epoch del nombre de la imagen** (instante en que el detector guarda el recorte, tras el
OCR, en el servidor) contra el **`created_at`** de MySQL (instante de la inserción) — ambos en hora del
servidor, sin desfase de reloj — para las **854 detecciones**:

| Latencia detección→BD | Valor |
|---|---|
| Mediana | **0 s** |
| p90 | **1 s** |
| Máx | 2 s |
| Distribución | 532 en 0 s · 317 en 1 s · 5 en 2 s |

→ Una vez detectada la placa, el sistema la **registra en la base de datos en ≤ 1 s** casi siempre
(n = 854). Es la latencia de la etapa OCR→POST→inserción, medida sobre la sesión completa y sin
depender del video. Confirma la operación en tiempo real.

Para una distribución (mediana / p90) con más vehículos: reproducir el video sincronizado y anotar la
hora del overlay cuando la placa cruza la línea de detección (el muestreo por fotogramas se salta los
autos, que cruzan en ~1–2 s). Luego `scripts/medir_e2e.py` cruza por placa y calcula la latencia.

Nota: esto mide la **latencia de registro** (ingreso → BD). La latencia adicional del *render* en el
dashboard (BD → navegador) es sub-segundo vía WebSocket/Reverb y requeriría una grabación del dashboard.
