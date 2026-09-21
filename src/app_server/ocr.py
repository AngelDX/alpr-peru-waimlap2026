"""Lectura OCR de placas con PaddleOCR (reconocimiento puro) en GPU.

El recorte que llega ya es la placa (lo entrega el detector YOLO), así que usamos
TextRecognition (sin detección): es lo correcto en ALPR de 2ª etapa y más rápido.
En Linux torch (YOLO) y paddle conviven en un mismo proceso, así que el OCR corre
in-process (sin worker). Modelo PP-OCRv4_server_rec: 88.0% placa, 0 vacíos en el test 250.
"""
import re

import config

# Mapas de confusión OCR usados sólo en las posiciones de tipo fijo del formato peruano.
_D2L = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "Z", "8": "B"}  # dígito->letra (pos1)
_L2D = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
        "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}                      # letra->dígito (pos4-6)
# Símbolos que el OCR confunde con la V de pos1; como NO son alfanuméricos, _extract_plate los
# descartaría -> hay que mapearlos a letra ANTES de extraer (str.translate sobre el texto crudo).
_SYM2L = str.maketrans({"/": "V", "\\": "V"})  # '/' o '\' -> V


def correct_plate6(p6):
    """Corrige una placa de 6 chars (sin guion) con el formato peruano y la heurística O/0.

    Formato `[L][A][A]-[N][N][N]`: pos1 siempre letra, pos4-6 siempre dígito (mapea la
    confusión típica si hace falta). pos2-3 son alfanuméricas, pero medido sobre SUNARP:
    cuando aparece O/0, pos2 es 0 (cero) el 100% y pos3 es O (letra) el 85% → se resuelve así.
    También I/1: pos2 es 1 (uno) el 100% y pos3 es I (letra) el ~82% (mismo patrón que O/0).
    Medido (server model, test 250): +6.0pp acumulado (88.0->94.0%); O/0 pos2-3 32%->89%.
    """
    if len(p6) != 6:
        return p6
    c = list(p6)
    if c[0].isdigit():                       # pos1: letra
        c[0] = _D2L.get(c[0], c[0])
    for i in (3, 4, 5):                       # pos4-6: dígito
        if c[i].isalpha():
            c[i] = _L2D.get(c[i], c[i])
    if c[1] in ("O", "0"):                    # pos2: O/0 -> 0 (cero). D NO se mapea: hay D reales en
        c[1] = "0"                            # pos2 (CDU/CDD/CDG... SUNARP); D->0 medido net -6, revertido.
    if c[1] in ("I", "1"):                    # pos2: I/1 -> 1 (uno)
        c[1] = "1"
    if c[2] in ("O", "0"):                    # pos3: O/0 -> O (letra)
        c[2] = "O"
    if c[2] in ("I", "1"):                    # pos3: I/1 -> I (letra)
        c[2] = "I"
    return "".join(c)


class Ocr:
    def __init__(self):
        import paddle
        from paddleocr import TextRecognition

        paddle.set_device(config.OCR_DEVICE)
        self.rec = TextRecognition(model_name=config.OCR_MODEL_NAME, device=config.OCR_DEVICE)
        self._pattern = re.compile(config.PLATE_REGEX, re.IGNORECASE)

    def read_license_plate(self, plate_crop):
        """Devuelve (placa, score). (None, 0.0) si no se reconoce nada utilizable."""
        result = self.rec.predict(plate_crop)
        if not result:
            return None, 0.0
        r0 = result[0]  # dict-like de TextRecognition (una entrada por imagen)
        text = str(r0.get("rec_text", "")).strip().upper()
        score = float(r0.get("rec_score", 0.0))

        plate = self._extract_plate(text)
        if plate is None:
            return None, 0.0
        if config.OCR_APPLY_CORRECTOR:
            p6 = correct_plate6(plate.replace("-", ""))
            plate = f"{p6[:3]}-{p6[3:]}"
        return plate, score

    def close(self):
        """Simetría con main.py; el modelo se libera al terminar el proceso."""
        self.rec = None

    def _extract_plate(self, text):
        """Saca la placa LLL-NNN del texto. None si no encaja en 6 alfanuméricos."""
        text = text.translate(_SYM2L)  # '/'/'\\' -> V antes de descartar no-alfanuméricos
        m = self._pattern.search(text)
        if m:
            return m.group(1).upper()
        compact = re.sub(r"[^A-Z0-9]", "", text)  # rescate si el guion no se reconoció
        if len(compact) == 6:
            return f"{compact[:3]}-{compact[3:]}"
        return None
