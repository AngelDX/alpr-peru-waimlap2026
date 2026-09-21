@echo off
REM ==== Experimento cross-domain controlado (RTX 5060 Ti 16 GB) ====
REM Ejecutar desde: F:\ARTICULOS\WAIMLAP\MobileALPR_dataset\crossdomain_exp
REM cache=disk : rapido y SIN reventar la RAM (ram cachea todo en memoria y satura equipos con poca RAM).
REM Si el disco va justo, usa --cache False. batch 48 va sobrado de VRAM (el cuello era la RAM, no la GPU).

set BATCH=48
set WORKERS=8
set CACHE=disk

echo [1/2] YOLOv8n SOLO camara  (batch=%BATCH%, cache=%CACHE%)...
python run_crossdomain.py --train --arch yolov8n.pt --data camonly.yaml --name yolov8n_camonly --batch %BATCH% --workers %WORKERS% --cache %CACHE%

echo [2/2] YOLOv8n FUSIONADO   (batch=%BATCH%, cache=%CACHE%)...
python run_crossdomain.py --train --arch yolov8n.pt --data fused.yaml   --name yolov8n_fused   --batch %BATCH% --workers %WORKERS% --cache %CACHE%

echo.
echo ==== (OPCIONAL) 2x2 completo con YOLO11n ====
REM python run_crossdomain.py --train --arch yolo11n.pt --data camonly.yaml --name yolo11n_camonly --batch %BATCH% --workers %WORKERS% --cache %CACHE%
REM python run_crossdomain.py --train --arch yolo11n.pt --data fused.yaml   --name yolo11n_fused   --batch %BATCH% --workers %WORKERS% --cache %CACHE%

echo.
echo Listo. JSONs: crossdomain_yolov8n_camonly.json y crossdomain_yolov8n_fused.json
pause
