# Gelişmiş Tespit Sistemi
# Bu script, videolarda şiddet, silah, insan ve kalabalık tespiti yapar
# Yazar: [Adınız]
# Tarih: [Tarih]

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

class AdvancedDetectionSystem:
    """
    Gelişmiş tespit sistemi sınıfı.
    Bu sınıf, farklı modelleri kullanarak video karelerinde çeşitli tespitler yapar.
    """
    
    def __init__(self):
        """
        Sınıfın başlatıcı metodu.
        Tüm modelleri ve gerekli ayarları yükler.
        """
        # YOLO modellerini yükle
        self.violence_model = YOLO('runs/detect/train5/weights/best.pt')  # Şiddet tespit modeli
        self.weapon_model = YOLO('runs/detect/train6/weights/best.pt')    # Silah/bıçak tespit modeli
        
        # MediaPipe ayarları
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        
        # Poz ve yüz detektörleri
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_detector = self.mp_face.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Detectron2 ayarları
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.detectron2_predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def process_frame(self, frame):
        """
        Video karesini işler ve tespit sonuçlarını görselleştirir.
        
        Args:
            frame (numpy.ndarray): İşlenecek video karesi
            
        Returns:
            numpy.ndarray: İşlenmiş ve tespit sonuçları eklenmiş video karesi
        """
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        # 1. YOLO Şiddet Tespiti (Kırmızı renk)
        violence_results = self.violence_model(frame)
        for result in violence_results:
            boxes = result.boxes
            for box in boxes:
                # Tespit kutusu koordinatlarını al
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{self.violence_model.names[cls]} {conf:.2f}'
                
                # Tespit kutusunu ve etiketi çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 2. YOLO Silah/Bıçak Tespiti (Mavi renk)
        weapon_results = self.weapon_model(frame)
        for result in weapon_results:
            boxes = result.boxes
            for box in boxes:
                # Tespit kutusu koordinatlarını al
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{self.weapon_model.names[cls]} {conf:.2f}'
                
                # Tespit kutusunu ve etiketi çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 3. MediaPipe İnsan Tespiti (Yeşil renk)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Poz tespiti
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            # Poz noktalarını ve bağlantılarını çiz
            self.mp_draw.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        
        # Yüz tespiti (maske kontrolü için)
        face_results = self.face_detector.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                # Yüz kutusu koordinatlarını al
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Yüz kutusunu ve güven skorunu çiz
                cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'Yüz {detection.score[0]:.2f}',
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 4. Detectron2 Kalabalık Tespiti (Mor renk)
        detectron2_outputs = self.detectron2_predictor(frame)
        instances = detectron2_outputs["instances"].to("cpu")
        
        # Sadece person sınıfını al (COCO dataset'inde person class_id=0)
        person_indices = (instances.pred_classes == 0).nonzero().squeeze(1)
        if len(person_indices) > 0:
            person_boxes = instances.pred_boxes[person_indices]
            person_scores = instances.scores[person_indices]
            
            # Kalabalık tespiti (3 veya daha fazla kişi varsa)
            if len(person_indices) >= 3:
                cv2.putText(annotated_frame, f'Kalabalık Tespit Edildi: {len(person_indices)} kişi',
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Her bir kişi için tespit kutusunu çiz
            for box, score in zip(person_boxes, person_scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(annotated_frame, f'Kişi {score:.2f}',
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return annotated_frame

    def analyze_scene(self, frame):
        """
        Sahnedeki durumu analiz eder ve metin olarak döndürür.
        
        Args:
            frame (numpy.ndarray): Analiz edilecek video karesi
            
        Returns:
            str: Sahne analizi sonucu (tespit edilen nesneler ve durumlar)
        """
        analysis = []
        
        # Şiddet analizi
        violence_results = self.violence_model(frame)
        for result in violence_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    analysis.append(f"{self.violence_model.names[cls]}")
        
        # Silah/bıçak analizi
        weapon_results = self.weapon_model(frame)
        for result in weapon_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    analysis.append(f"{self.weapon_model.names[cls]}")
        
        # Kalabalık analizi
        detectron2_outputs = self.detectron2_predictor(frame)
        instances = detectron2_outputs["instances"].to("cpu")
        person_count = (instances.pred_classes == 0).sum().item()
        if person_count >= 3:
            analysis.append(f"Kalabalık ({person_count} kişi)")
        
        # Yüz/maske analizi
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detector.process(rgb_frame)
        if face_results.detections:
            analysis.append(f"Yüz tespit edildi ({len(face_results.detections)})")
        
        return " | ".join(analysis) if analysis else "Normal durum"

def main():
    """
    Ana program fonksiyonu
    """
    # Video dosyası yolu
    video_path = 'Violence detection.mp4'
    
    print(f"Video dosyası açılıyor: {video_path}")
    
    # Video capture başlat
    cap = cv2.VideoCapture(video_path)
    
    # Video açılıp açılmadığını kontrol et
    if not cap.isOpened():
        print(f"Hata: Video dosyası açılamadı: {video_path}")
        return
        
    # Video özelliklerini göster
    print("Video başarıyla açıldı!")
    print(f"Video özellikleri:")
    print(f"- Genişlik: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"- Yükseklik: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"- FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
    
    # Pencere oluştur ve boyutlandır
    window_name = 'Advanced Detection System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    try:
        # Tespit sistemini başlat
        detector = AdvancedDetectionSystem()
        
        while cap.isOpened():
            # Kareyi oku
            ret, frame = cap.read()
            if not ret:
                print("Video bitti veya frame okunamadı.")
                break
                
            # Frame'i işle
            processed_frame = detector.process_frame(frame)
            
            # Sahne analizi
            analysis = detector.analyze_scene(frame)
            
            # Analiz sonucunu ekrana yaz
            cv2.putText(processed_frame, analysis, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # FPS bilgisini ekrana yaz
            cv2.putText(processed_frame, f'FPS: {int(cap.get(cv2.CAP_PROP_FPS))}',
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Renk açıklamalarını ekle
            legend_y = 90
            cv2.putText(processed_frame, 'Kırmızı: Şiddet', (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(processed_frame, 'Mavi: Silah/Bıçak', (10, legend_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(processed_frame, 'Yeşil: İnsan Hareketi', (10, legend_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(processed_frame, 'Sarı: Yüz', (10, legend_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(processed_frame, 'Mor: Kalabalık', (10, legend_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Sonucu göster
            cv2.imshow(window_name, processed_frame)
            
            # 'q' tuşuna basılırsa çık (30ms bekle - daha yavaş oynatım için)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                print("Program kullanıcı tarafından sonlandırıldı.")
                break
    
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
        import traceback
        print("Hata detayı:")
        print(traceback.format_exc())
    
    finally:
        print("Program kapatılıyor...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 