# -*- coding: utf-8 -*-
"""
Experimento controlado de generalizacion cross-domain (cierra el confound del articulo WAIMLAP).

OBJETIVO
--------
Aislar el efecto de FUSIONAR el dataset publico movil, manteniendo la MISMA arquitectura.
Para eso se entrena el mismo backbone bajo dos condiciones de datos y se evalua cada modelo
resultante en los DOS test sets independientes:

    In-domain  = test_camara (1500 img, camara fija, el objetivo real)
    Out-domain = test_movil  (323 img, celular/web, dominio publico)

MATRIZ A COMPLETAR (2x2, misma arquitectura en cada fila):
                       entrenado SOLO camara   entrenado FUSIONADO
    YOLOv8n                  (A1)                    (A2)
    YOLO11n  (opcional)      (B1)                    (B2)

La comparacion A1 vs A2 (mismo YOLOv8n, solo cambian los datos) es la que CIERRA el argumento:
como la arquitectura es identica, la diferencia se debe exclusivamente a la fusion de datos.

USO
---
1) Entrenar + evaluar (lo que falta). Ejemplo para YOLOv8n:
     python run_crossdomain.py --train --arch yolov8n.pt --data camonly.yaml --name yolov8n_camonly
     python run_crossdomain.py --train --arch yolov8n.pt --data fused.yaml   --name yolov8n_fused
   (opcional, para el 2x2 completo)
     python run_crossdomain.py --train --arch yolo11n.pt --data camonly.yaml --name yolo11n_camonly
     python run_crossdomain.py --train --arch yolo11n.pt --data fused.yaml   --name yolo11n_fused

2) Solo evaluar un peso YA entrenado (rapido, sin reentrenar):
     python run_crossdomain.py --weights ruta/al/best.pt --name yolov8_fused_existente

Cada corrida escribe crossdomain_<name>.json con el MISMO esquema que tus archivos actuales,
listo para pegar en la Tabla 4 del articulo.

REQUISITOS: ultralytics instalado, GPU (RTX 5060 Ti 16 GB). Corre desde esta carpeta
(crossdomain_exp/), o pasa rutas absolutas a los .yaml.
RENDIMIENTO: batch 48 + cache=ram + workers 8 exprimen la tarjeta (por defecto en run_all.bat).
Blackwell (sm_120) necesita torch>=2.7 con CUDA 12.8+ (whl/cu128).
"""
import argparse, json, os, sys
from pathlib import Path


def eval_split(model, data_yaml, imgsz, device, batch=64):
    """Evalua el modelo en el split 'test' del yaml dado y devuelve las 4 metricas."""
    r = model.val(data=data_yaml, split="test", imgsz=imgsz, device=device, batch=batch, verbose=False)
    return {
        "mAP50_95": round(float(r.box.map), 4),
        "mAP50":    round(float(r.box.map50), 4),
        "precision": round(float(r.box.mp), 4),
        "recall":    round(float(r.box.mr), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-domain controlado (WAIMLAP)")
    ap.add_argument("--train", action="store_true", help="entrena antes de evaluar")
    ap.add_argument("--arch", default="yolov8n.pt", help="peso base para entrenar (yolov8n.pt, yolo11n.pt, ...)")
    ap.add_argument("--data", default="fused.yaml", help="yaml de entrenamiento (camonly.yaml | fused.yaml)")
    ap.add_argument("--weights", default=None, help="modo solo-eval: ruta a un best.pt ya entrenado")
    ap.add_argument("--name", required=True, help="nombre de la corrida (define el JSON de salida)")
    ap.add_argument("--test-camara", default="test_camara.yaml")
    ap.add_argument("--test-movil", default="test_movil.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32,
                    help="batch. 16 GB (RTX 5060 Ti) aguanta 32-64 en YOLOn@640. Mantenlo IGUAL en todas las condiciones.")
    ap.add_argument("--workers", type=int, default=8, help="hilos de carga de datos (sube util de GPU)")
    ap.add_argument("--cache", default="ram", help="cachea imagenes: ram | disk | False (ram acelera mucho con 16 GB)")
    ap.add_argument("--seed", type=int, default=0, help="semilla de entrenamiento; fijala igual en todas las condiciones")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="det_runs_crossdomain")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics no esta instalado. Instala con: pip install ultralytics")

    # Info de GPU (util para verificar que la RTX 5060 Ti Blackwell esta activa)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[gpu] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")
        else:
            print("[gpu] AVISO: torch no ve la GPU. La RTX 5060 Ti (Blackwell sm_120) necesita "
                  "torch>=2.7 con wheels CUDA 12.8+. Instala: pip install -U torch --index-url https://download.pytorch.org/whl/cu128")
    except Exception:
        pass

    cache = args.cache
    if isinstance(cache, str) and cache.lower() in ("false", "no", "0"):
        cache = False

    if args.train:
        model = YOLO(args.arch)
        # Hiperparametros identicos a la Tabla 1 del articulo (comparacion justa entre condiciones).
        model.train(
            data=args.data, epochs=args.epochs, patience=args.patience,
            imgsz=args.imgsz, batch=args.batch, workers=args.workers, cache=cache,
            optimizer="SGD",
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3, seed=args.seed, deterministic=True, amp=True,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
            fliplr=0.5, mosaic=1.0, close_mosaic=10, erasing=0.4,
            box=7.5, cls=0.5, dfl=1.5,
            project=args.project, name=args.name, exist_ok=True,
        )
        # save_dir REAL del entrenador (ultralytics guarda en su runs_dir configurado,
        # que puede no coincidir con el cwd ni con --project relativo).
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        weights_path = str(best)
        print(f"[train] pesos guardados en: {weights_path}")
        model = YOLO(weights_path)
    else:
        if not args.weights:
            sys.exit("Sin --train debes pasar --weights ruta/al/best.pt")
        weights_path = args.weights
        model = YOLO(weights_path)

    print(f"[eval] in-domain  (test_camara) ...")
    indom = eval_split(model, args.test_camara, args.imgsz, args.device)
    print(f"[eval] cross-domain (test_movil) ...")
    cross = eval_split(model, args.test_movil, args.imgsz, args.device)

    ret = round(cross["mAP50_95"] / indom["mAP50_95"], 4) if indom["mAP50_95"] else 0.0
    out = {
        "modelo": os.path.abspath(weights_path),
        "nombre": args.name,
        "imgsz": args.imgsz,
        "resultados": {"in_domain_camara": indom, "cross_domain_movil": cross},
        "brecha_de_dominio": {
            "delta_mAP50_95": round(indom["mAP50_95"] - cross["mAP50_95"], 4),
            "delta_mAP50":    round(indom["mAP50"] - cross["mAP50"], 4),
            "delta_recall":   round(indom["recall"] - cross["recall"], 4),
            "retencion_mAP50_95": ret,
        },
    }
    out_path = f"crossdomain_{args.name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n===== RESULTADO =====")
    print(f"  Modelo         : {args.name}")
    print(f"  In-domain  mAP50-95={indom['mAP50_95']:.4f}  mAP50={indom['mAP50']:.4f}")
    print(f"  Cross-dom  mAP50-95={cross['mAP50_95']:.4f}  mAP50={cross['mAP50']:.4f}")
    print(f"  Retencion mAP50-95 : {ret*100:.1f}%")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
