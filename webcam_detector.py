# Webcam Test Uygulaması
# Bu script, webcam görüntüsünde şiddet ve silah tespiti yapar
# Video kaydetme özelliği ve kullanıcı girişi kaldırılmıştır.
# Yazar: [Adınız]
# Tarih: [Tarih]

import cv2
from ultralytics import YOLO
import time
import os
import torch

def main():
    """
    Ana program fonksiyonu
    """
    # Sabitlenmiş değişkenler
    model_path = 'runs/detect/train5/weights/best.pt' # Model dosyanın yolunu buraya yaz
    confidence_threshold = 0.5                      # Tespit eşiğini buraya yaz (0.0 ile 1.0 arası)

    try:
        # Model dosyasının varlığını kontrol et
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hata: Model dosyası bulunamadı: {model_path}\nLütfen 'model_path' değişkenini doğru yola ayarlayın.")
        
        # CUDA kullanımını devre dışı bırak
        torch.backends.cudnn.enabled = False

        print("Model yükleniyor...")
        # Her iki tespit için aynı modeli kullan
        violence_model = YOLO(model_path)
        weapon_model = YOLO(model_path)
        # CPU moduna geç
        violence_model.cpu = True
        weapon_model.cpu = True
        print("Model yüklendi!")

        # Webcam'i başlat
        print("Webcam başlatılıyor...")
        cap = cv2.VideoCapture(0) # 0, varsayılan webcam'i temsil eder
        
        if not cap.isOpened():
            raise ValueError("Hata: Webcam açılamadı!\nLütfen webcam'in bağlı ve kullanılabilir olduğundan emin olun.")

        # Pencereyi oluştur
        window_name = 'Webcam Test'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        print("Webcam başlatıldı ve pencere oluşturuldu!")

        # FPS sayacı için değişkenler
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0 

        print("Program başlatıldı! Çıkmak için 'q' tuşuna basın...")

        while True:
            # Kareyi oku
            ret, frame = cap.read()
            if not ret:
                print("Webcam'den kare alınamıyor! Akış sonlanıyor...")
                break

            # FPS hesapla
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            # Kareyi işle
            # conf parametresi ile tespit eşiğini doğrudan modele gönderiyoruz
            results_violence = violence_model(frame, device='cpu', conf=confidence_threshold)
            results_weapon = weapon_model(frame, device='cpu', conf=confidence_threshold)

            # Görselleştirme için kopya oluştur
            output_frame = frame.copy()
            frame_height, frame_width = output_frame.shape[:2]

            # Şiddet tespiti (Kırmızı) - Sağ üst
            for result in results_violence:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    # cls = int(box.cls[0]) # Bu değişkeni kullanmadığınız için silebiliriz
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f'Şiddet: {conf:.2f}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = frame_width - text_size[0] - 10
                    cv2.putText(output_frame, text, (text_x, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Silah tespiti (Mavi) - Sol üst
            for result in results_weapon:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    # cls = int(box.cls[0]) # Bu değişkeni kullanmadığınız için silebiliriz
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(output_frame, f'Silah: {conf:.2f}', (10, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

            # Sonucu göster
            cv2.imshow(window_name, output_frame)

            # Q tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Program kullanıcı tarafından sonlandırıldı.")
                break

    except FileNotFoundError as e:
        print(f"\nHata: {str(e)}")
    except ValueError as e:
        print(f"\nHata: {str(e)}")
    except Exception as e:
        print(f"\nBeklenmeyen bir hata oluştu: {str(e)}")
        print("Hata detayları:", e.__class__.__name__)

    finally:
        print("\nİşlem tamamlandı.")
        try:
            cap.release()
        except NameError: # 'cap' tanımlı olmayabilir
            pass
        cv2.destroyAllWindows()

# Programın ana giriş noktası
if __name__ == "__main__":
    main()