# Video Test Uygulaması
# Bu script, videolarda şiddet, silah, poz ve yüz tespiti yapar
# Yazar: [Adınız]
# Tarih: [Tarih]

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import os

def process_video(video_path):
    """
    Videoyu işleyerek şiddet, silah, poz ve yüz tespiti yapar.
    
    Args:
        video_path (str): İşlenecek video dosyasının yolu
    """
    # YOLO modellerini yükle
    violence_model = YOLO('runs/detect/train5/weights/best.pt')
    weapon_model = YOLO('runs/detect/train6/weights/best.pt')
    
    # MediaPipe ayarları
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    
    # Poz ve yüz detektörleri
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
    
    # Video yakalama nesnesini başlat
    print(f"Video yükleniyor: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video açılamadı: {video_path}")
    
    # Video özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video özelliklerini göster
    print(f"Video özellikleri:")
    print(f"- Genişlik: {width}")
    print(f"- Yükseklik: {height}")
    print(f"- FPS: {fps}")
    print(f"- Toplam kare: {total_frames}")
    
    # Pencereyi oluştur
    window_name = 'Video Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # FPS sayacı için değişkenler
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    try:
        while cap.isOpened():
            # Kareyi oku
            ret, frame = cap.read()
            if not ret:
                print("Video bitti.")
                break
            
            # FPS hesapla
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Kareyi işle
            results_violence = violence_model(frame)
            results_weapon = weapon_model(frame)
            
            # MediaPipe işlemleri
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            face_results = face_detector.process(rgb_frame)
            
            # Görselleştirme için kopya oluştur
            output_frame = frame.copy()
            frame_height, frame_width = output_frame.shape[:2]
            
            # Şiddet tespiti (Kırmızı) - Sağ üst
            for result in results_violence:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f'Şiddet: {conf:.2f}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = frame_width - text_size[0] - 10
                    cv2.putText(output_frame, text, (text_x, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Silah tespiti (Mavi) - Sol üst
            for result in results_weapon:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(output_frame, f'Silah: {conf:.2f}', (10, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Poz tespiti (Yeşil)
            if pose_results.pose_landmarks:
                mp_draw.draw_landmarks(
                    output_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
            
            # Yüz tespiti (Sarı)
            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = output_frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width_box = int(bbox.width * w)
                    height_box = int(bbox.height * h)
                    cv2.rectangle(output_frame, (x, y), (x + width_box, y + height_box),
                                (0, 255, 255), 2)
            
            # FPS göster
            cv2.putText(output_frame, f'FPS: {current_fps}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Renk açıklamalarını ekle
            legend_y = 60
            # Açıklamaları karşılıklı yerleştir
            cv2.putText(output_frame, 'Kırmızı: Şiddet', (frame_width - 150, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(output_frame, 'Mavi: Silah', (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(output_frame, 'Yeşil: İnsan Hareketi', (10, legend_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output_frame, 'Sarı: Yüz', (10, legend_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Sonucu göster
            cv2.imshow(window_name, output_frame)
            
            # Q tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Program kullanıcı tarafından sonlandırıldı.")
                break
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    
    finally:
        print("İşlem tamamlandı.")
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Ana program fonksiyonu
    """
    try:
        # Kullanıcıdan video yolunu al
        video_path = "silah.mp4"
        
        # Dosyanın varlığını kontrol et
        if not os.path.exists(video_path):
            raise ValueError(f"Video dosyası bulunamadı: {video_path}")
        
        # Videoyu işle
        process_video(video_path)
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

# Programın ana giriş noktası
if __name__ == "__main__":
    main() 