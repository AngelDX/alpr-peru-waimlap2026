from paddleocr import PaddleOCR
import gc

class Ocr:
    def __init__(self):
        # Inicializa PaddleOCR una sola vez
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            use_gpu=True,
            ocr_version='PP-OCRv4',  # Agrega esto si es necesario
            # ec_model_dir="path_to_lighter_model"  # Asegúrate de que la ruta sea correcta
        )

    def read_license_plate(self, license_plate_crop):    
        result = self.ocr.ocr(license_plate_crop, cls=True)  # Usa self.ocr, no ocr
        
        if result and result[0] is not None:
            result = result[0]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            
            license_plate = txts[0] if txts else "FFF-999"
            accuracy = float(scores[0]) if scores else 0.0

            # Limpieza de variables innecesarias
            del result
            gc.collect()
            return license_plate, accuracy
        
        gc.collect()
        return "FFF-999", 0.0
