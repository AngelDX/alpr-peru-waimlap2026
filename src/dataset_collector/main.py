import cv2 as cv
#from glob import glob
from ultralytics import YOLO
from paddleocr import PaddleOCR,draw_ocr
from matplotlib import pyplot as plt
import time
import psutil
import gc
import torch
from send import Send
from ocrIA import Ocr

np_model = YOLO('best.pt').to("cuda") #Por el momento el mejor resultado


URL="rtsp://usuario:password@ip/Streaming/Channels/1"

send=Send()
ocr=Ocr()

def iniciar_captura():
    cap = cv.VideoCapture(URL,cv.CAP_FFMPEG)
    if cap.isOpened():
        print("Video abierto correctamente")
    else:
        print("Error al abrir el video")
    return cap
    
try:
    cap=iniciar_captura()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede leer el stream")
            cap.release()
            time.sleep(1)
            cap=iniciar_captura()
            continue

        detections = np_model.track(frame, stream=False, verbose=False)
        #detections = np_model.track(frame, stream=False)
        # Conversión de formato de color
        #frame = cv.cvtColor(frame, cv.COLOR_YUV2RGB_I420)  # Alternativa: COLOR_YUV2RGB
        #print(f"Memoria usada: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
        for detection in detections:
            boxes = detection.boxes.cpu().numpy()
            for box in boxes:
                if box.conf < 0.90:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                plate_roi = frame[y1:y2, x1:x2]
                np_text, np_score = ocr.read_license_plate(plate_roi)
                if np_score >= 0.85:
                        timestamp = str(time.time())
                        dimension = f"{plate_roi.shape[1]}x{plate_roi.shape[0]}"
                        plate_score = box.conf[0]
                        
                        #cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        #cv.putText(frame,str(round(box.conf[0],2)),[x1, y1],cv.FONT_HERSHEY_SIMPLEX,0.8,(204, 255, 0),2)
                    
                        # 6. Escritura asíncrona de imágenes
                        cv.imwrite(f'data_vehicles_train2/{timestamp}_{str(round(plate_score,2))}_{str(frame.shape)}_{np_text}.jpg', frame)
                        cv.imwrite(f'data_plates_train2/{timestamp}_{str(round(plate_score,2))}_{str(plate_roi.shape)}_{np_text}.jpg', plate_roi)
                        
                        # 7. Envío no bloqueante
                        #send.send_post(np_text,np_score,str(box.conf),dimension)
                        send.send_post(np_text,np_score,str(box.conf[0]),str(round(plate_roi.size,0)),dimension)
                        
                del plate_roi
            del detection
            del boxes
        #cv.imshow('CamaraIP', frame)
        del frame
        if cv.waitKey(1) == ord('q'):
            break
finally:
    cap.release()
    cv.destroyAllWindows()
    torch.cuda.empty_cache()
    


