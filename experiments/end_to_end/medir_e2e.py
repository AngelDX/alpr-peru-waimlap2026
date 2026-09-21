# -*- coding: utf-8 -*-
"""
Evaluacion extremo a extremo (end-to-end) del sistema ALPR.

Cruza la VERDAD DE TERRENO (placas + hora de la camara, anotadas del video)
contra el EXPORT de la tabla `plates` de MySQL (con created_at), clasifica cada
vehiculo, calcula las metricas y la LATENCIA DE REGISTRO, y rellena la plantilla
Excel.

USO:
    python medir_e2e.py --verdad verdad.csv --sistema plates.csv \
        --plantilla "Plantilla_evaluacion_end_to_end.xlsx" \
        --salida "Evaluacion_end_to_end_LLENA.xlsx" --duracion 47 --auto-desfase

verdad.csv : placa_real, hora_camara        (hh:mm:ss del overlay de la camara)
plates.csv : number, ocrscore, platescore, created_at [, dimension]

Latencia de registro = created_at - hora_camara  (emparejado por placa).
Si los relojes camara/servidor difieren, usa --desfase <seg> o --auto-desfase.
"""
import argparse, csv, re, sys
from statistics import median
from collections import Counter


def norm_core(s):
    if s is None:
        return ""
    return re.sub(r'[^A-Za-z0-9]', '', str(s)).upper()


def edit_distance(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return la or lb
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_time_to_sec(s):
    if s is None or str(s).strip() == "":
        return None
    txt = str(s).strip()
    if " " in txt:
        txt = txt.split(" ")[-1]
    if "T" in txt:
        txt = txt.split("T")[-1]
    txt = txt.split(".")[0].split("+")[0]
    parts = txt.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    else:
        return None
    return h * 3600 + m * 60 + sec


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        sample = f.read(2048); f.seek(0)
        delim = ';' if sample.count(';') > sample.count(',') else ','
        return list(csv.DictReader(f, delimiter=delim))


def pick(row, *names):
    low = {k.lower().strip(): v for k, v in row.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _f(v):
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return ""


def main():
    ap = argparse.ArgumentParser(description="Evaluacion end-to-end ALPR")
    ap.add_argument('--verdad', required=True)
    ap.add_argument('--sistema', required=True)
    ap.add_argument('--plantilla', default="Plantilla_evaluacion_end_to_end.xlsx")
    ap.add_argument('--salida', default="Evaluacion_end_to_end_LLENA.xlsx")
    ap.add_argument('--duracion', type=float, default=0, help="minutos de la sesion (throughput)")
    ap.add_argument('--umbral', type=int, default=2, help="max. dist. de edicion para 'error de OCR'")
    ap.add_argument('--desfase', type=float, default=0,
                    help="desfase de reloj camara->servidor en segundos (se resta a la latencia)")
    ap.add_argument('--auto-desfase', action='store_true',
                    help="estima el desfase desde los datos (percentil 5 de created_at-hora_camara) y lo resta")
    ap.add_argument('--col-placa-real', default='placa_real')
    ap.add_argument('--col-hora', default='hora_camara')
    ap.add_argument('--col-number', default='number')
    ap.add_argument('--col-ocr', default='ocrscore')
    ap.add_argument('--col-placascore', default='platescore')
    ap.add_argument('--col-created', default='created_at')
    args = ap.parse_args()

    gt_rows = read_csv(args.verdad)
    sys_rows = read_csv(args.sistema)

    sistema = []
    for r in sys_rows:
        num = pick(r, args.col_number, 'number', 'placa', 'plate')
        core = norm_core(num)
        if not core:
            continue
        sistema.append({
            'core': core, 'texto': (num or '').strip().upper(),
            'ocr': _f(pick(r, args.col_ocr, 'ocrscore', 'ocr')),
            'placa': _f(pick(r, args.col_placascore, 'platescore', 'placa_score')),
            'created': pick(r, args.col_created, 'created_at', 'fecha', 'hora'),
            'claimed': False,
        })
    core_counts = Counter(s['core'] for s in sistema)

    resultados = []
    for i, g in enumerate(gt_rows, 1):
        real = pick(g, args.col_placa_real, 'placa_real', 'placa', 'plate')
        hora = pick(g, args.col_hora, 'hora_camara', 'hora_video', 'hora', 'tiempo')
        rcore = norm_core(real)
        if not rcore:
            continue
        match = None
        for s in sistema:
            if not s['claimed'] and s['core'] == rcore:
                match = s; break
        if match is None:
            best, bestd = None, 99
            for s in sistema:
                if s['claimed']:
                    continue
                d = edit_distance(rcore, s['core'])
                if d < bestd:
                    best, bestd = s, d
            if best is not None and bestd <= args.umbral:
                match = best
        detectada = match is not None
        if detectada:
            match['claimed'] = True
        placa_sys = match['texto'] if match else ""
        correcta = detectada and norm_core(placa_sys) == rcore
        dup = "Si" if (match and core_counts[match['core']] > 1) else ("No" if detectada else "")
        created_txt = match['created'] if match else None
        tc, ts = parse_time_to_sec(hora), parse_time_to_sec(created_txt)
        delta = (ts - tc) if (tc is not None and ts is not None) else None
        resultados.append({
            'n': i, 'hora': hora or "", 'real': (real or '').strip().upper(),
            'det': "Si" if detectada else "No", 'sys': placa_sys,
            'correcta': "Si" if correcta else ("No" if detectada else ""),
            'created': (created_txt or "") if match else "", 'dup': dup,
            'placa': match['placa'] if match else "", 'ocr': match['ocr'] if match else "",
            'delta': delta, 'lat': "",
        })

    # desfase de reloj y latencia final
    deltas = sorted(r['delta'] for r in resultados if r['delta'] is not None)
    offset = args.desfase
    if args.auto_desfase and deltas:
        offset = deltas[max(0, int(round(0.05 * (len(deltas) - 1))))]
        print("[auto-desfase] desfase de reloj estimado: %.0f s (~%.1f min)" % (offset, offset / 60))
    for r in resultados:
        if r['delta'] is not None:
            r['lat'] = round(max(0.0, r['delta'] - offset), 1)

    total = len(resultados)
    det = sum(1 for r in resultados if r['det'] == "Si")
    cor = sum(1 for r in resultados if r['correcta'] == "Si")
    dups = len(sistema) - len(core_counts)
    extra = sum(1 for s in sistema if not s['claimed'])
    lats = [r['lat'] for r in resultados if isinstance(r['lat'], (int, float))]

    def pct(a, b):
        return ("%.1f%%" % (100 * a / b)) if b else "-"
    print("\n===== RESUMEN END-TO-END =====")
    print("Total de vehiculos (verdad)      : %d" % total)
    print("Detectados                       : %d  (%s)" % (det, pct(det, total)))
    print("No detectados (omitidos)         : %d" % (total - det))
    print("Placas correctas (E2E)           : %d" % cor)
    print("Exactitud extremo a extremo      : %s" % pct(cor, total))
    print("Exactitud de OCR (sobre detect.) : %s" % pct(cor, det))
    print("Registros duplicados (sistema)   : %d" % dups)
    print("Registros extra sin emparejar    : %d" % extra)
    if lats:
        ls = sorted(lats)
        p90 = ls[min(len(ls) - 1, int(round(0.9 * (len(ls) - 1))))]
        print("Latencia registro (s)            : mediana %.1f | p90 %.1f | max %.1f" % (median(lats), p90, max(lats)))
    if args.duracion:
        print("Throughput                       : %.2f veh/min" % (total / args.duracion))
    print("================================\n")

    try:
        import openpyxl
    except ImportError:
        print("openpyxl no instalado; instala con: pip install openpyxl", file=sys.stderr)
        return
    wb = openpyxl.load_workbook(args.plantilla)
    rg = wb['Registro']
    for row in range(3, 1003):
        for col in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
            rg.cell(row, col).value = None
    for k, r in enumerate(resultados):
        row = 3 + k
        rg.cell(row, 1, r['n']); rg.cell(row, 2, r['hora']); rg.cell(row, 3, r['real'])
        rg.cell(row, 4, r['det']); rg.cell(row, 5, r['sys']); rg.cell(row, 7, r['created'])
        rg.cell(row, 8, r['dup'])
        if r['placa'] not in (None, ""):
            rg.cell(row, 9, r['placa'])
        if r['ocr'] not in (None, ""):
            rg.cell(row, 10, r['ocr'])
        if r['lat'] not in (None, ""):
            rg.cell(row, 11, r['lat'])
    if args.duracion:
        wb['Resumen']['B4'] = args.duracion
    wb.save(args.salida)
    print("Excel rellenado -> %s" % args.salida)
    print("Abrelo en Excel/LibreOffice: la hoja 'Resumen' recalcula sola.")


if __name__ == "__main__":
    main()
