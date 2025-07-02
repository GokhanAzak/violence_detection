import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time

class EnhancedPoseDetector:
    def __init__(self, yolo_model_path='runs/detect/train5/weights/best.pt'):
        # YOLO modelini yükle
        self.yolo_model = YOLO(yolo_model_path)
        
        # MediaPipe pose detektörünü ayarla
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def process_frame(self, frame):
        # YOLO tespiti
        yolo_results = self.yolo_model(frame)
        
        # MediaPipe pose tespiti
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        # Sonuçları görselleştir
        annotated_frame = frame.copy()
        
        # YOLO tespitlerini çiz
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{self.yolo_model.names[cls]} {conf:.2f}'
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # MediaPipe pose noktalarını çiz
        if pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        return annotated_frame

def main():
    # Video capture başlat
    cap = cv2.VideoCapture(0)  # Webcam için 0, video dosyası için dosya yolu
    detector = EnhancedPoseDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Frame'i işle
        processed_frame = detector.process_frame(frame)
        
        # Sonucu göster
        cv2.imshow('Enhanced Pose Detection', processed_frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 