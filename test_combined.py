import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import os

print("Program başlatılıyor...")

try:
    # Load YOLO model
    model_path = 'runs/detect/train5/weights/best.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    model = YOLO(model_path)
    print("YOLO modeli yüklendi.")

    # Open video file
    video_path = 'Violence detection.mp4'
    print(f"Video dosyası açılıyor: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video dosyası açılamadı!")

    print("Video başarıyla açıldı.")
    print(f"Video özellikleri:")
    print(f"- FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"- Genişlik: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"- Yükseklik: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    # Create window
    window_name = 'Violence Detection.mp4'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video bitti veya frame okunamadı.")
            break

        # YOLO Detection
        results = model(frame)
        
        # Draw YOLO detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Q tuşuna basıldı. Program sonlandırılıyor...")
            break

except Exception as e:
    print(f"Bir hata oluştu: {str(e)}")
    import traceback
    print("Hata detayı:")
    print(traceback.format_exc())

finally:
    print("Kaynaklar serbest bırakılıyor...")
    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()
    print("Program sonlandı.") 