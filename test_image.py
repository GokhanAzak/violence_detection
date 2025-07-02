# Resim Test Uygulaması
# Bu script, resimlerde şiddet, silah, poz ve yüz tespiti yapar
# Yazar: [Adınız]
# Tarih: [Tarih]

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

def process_image(image_path):
    """
    Resmi işleyerek şiddet, silah, poz ve yüz tespiti yapar.
    
    Args:
        image_path (str): İşlenecek resim dosyasının yolu
    
    Returns:
        numpy.ndarray: İşlenmiş resim
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
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
    
    # Resmi oku
    print(f"Resim yükleniyor: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Resim yüklenemedi: {image_path}")
    
    # Resmi işle
    results_violence = violence_model(image)
    results_weapon = weapon_model(image)
    
    # MediaPipe işlemleri
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_image)
    face_results = face_detector.process(rgb_image)
    
    # Sonuçları görselleştir
    output_image = image.copy()
    
    # Şiddet tespiti (Kırmızı)
    for result in results_violence:
        boxes = result.boxes
        for box in boxes:
            # Tespit kutusu koordinatlarını al
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # Tespit kutusunu ve etiketi çiz
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(output_image, f'Şiddet: {conf:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Silah tespiti (Mavi)
    for result in results_weapon:
        boxes = result.boxes
        for box in boxes:
            # Tespit kutusu koordinatlarını al
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # Tespit kutusunu ve etiketi çiz
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output_image, f'Silah: {conf:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Poz tespiti (Yeşil)
    if pose_results.pose_landmarks:
        # Poz noktalarını ve bağlantılarını çiz
        mp_draw.draw_landmarks(
            output_image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
    
    # Yüz tespiti (Sarı)
    if face_results.detections:
        for detection in face_results.detections:
            # Yüz kutusu koordinatlarını al
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = output_image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            # Yüz kutusunu çiz
            cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 255), 2)
    
    # Renk açıklamalarını ekle
    legend_y = 30
    cv2.putText(output_image, 'Kırmızı: Şiddet', (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(output_image, 'Mavi: Silah/Bıçak', (10, legend_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(output_image, 'Yeşil: İnsan Hareketi', (10, legend_y + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output_image, 'Sarı: Yüz', (10, legend_y + 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return output_image

def main():
    """
    Ana program fonksiyonu
    """
    try:
        # Test edilecek resim
        image_path = "1.mp4"  # Test edeceğiniz resmin yolunu buraya yazın
        
        # Resmi işle
        result_image = process_image(image_path)
        
        # Sonucu göster
        window_name = 'Image Test'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.imshow(window_name, result_image)
        
        # Sonucu kaydet
        output_path = "output_" + image_path
        cv2.imwrite(output_path, result_image)
        print(f"Sonuç kaydedildi: {output_path}")
        
        # Pencereyi kapat
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

# Programın ana giriş noktası
if __name__ == "__main__":
    main() 