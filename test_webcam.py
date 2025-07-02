# Webcam Test Uygulaması
# Bu script, webcam görüntüsünde şiddet ve silah tespiti yapar
# Yazar: [Adınız]
# Tarih: [Tarih]

import cv2
from ultralytics import YOLO
import time
import os
import torch
from datetime import datetime

def create_output_folder():
    """
    İşlenen webcam kayıtlarının kaydedileceği klasörü oluşturur.
    
    Returns:
        str: Oluşturulan klasörün yolu
    """
    output_folder = "processed_videos"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Çıktı klasörü oluşturuldu: {output_folder}")
    return output_folder

def get_output_filename(output_folder):
    """
    Webcam kaydı için benzersiz bir dosya adı oluşturur.
    
    Args:
        output_folder (str): Çıktı klasörünün yolu
    
    Returns:
        str: Oluşturulan dosya yolu
    """
    # Zaman damgası ekle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_folder, f"webcam_recording_{timestamp}.mp4")

def main():
    """
    Ana program fonksiyonu
    """
    try:
        # CUDA kullanımını devre dışı bırak
        torch.backends.cudnn.enabled = False
        
        # Model dosyasının varlığını kontrol et
        model_path = 'runs/detect/train5/weights/best.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
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
        cap = cv2.VideoCapture("1.mp4")
        
        if not cap.isOpened():
            raise ValueError("Hata: Video açılamadı!")
        
        # Video özelliklerini al
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Çıktı klasörünü oluştur ve dosya yolunu al
        output_folder = create_output_folder()
        output_path = get_output_filename(output_folder)
        
        # Video yazıcıyı ayarla
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # Pencereyi oluştur
        window_name = 'Webcam Test'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        print("Webcam başlatıldı ve pencere oluşturuldu!")
        
        # FPS sayacı için değişkenler
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        print("Program başlatıldı! Çıkmak için 'q' tuşuna basın...")
        
        while True:
            # Kareyi oku
            ret, frame = cap.read()
            if not ret:
                print("Webcam'den kare alınamıyor!")
                break
            
            # FPS hesapla
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Kareyi işle
            results_violence = violence_model(frame, device='cpu')
            results_weapon = weapon_model(frame, device='cpu')
            
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
            
            # FPS göster
            cv2.putText(output_frame, f'FPS: {fps}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Renk açıklamalarını ekle
            legend_y = 60
            # Açıklamaları karşılıklı yerleştir
            cv2.putText(output_frame, 'Kırmızı: Şiddet', (frame_width - 150, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(output_frame, 'Mavi: Silah', (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Sonucu kaydet ve göster
            out.write(output_frame)
            cv2.imshow(window_name, output_frame)
            
            # Q tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Program kullanıcı tarafından sonlandırıldı.")
                break
    
    except FileNotFoundError as e:
        print(f"\nHata: {str(e)}")
        print("Lütfen model dosyasının doğru konumda olduğunu kontrol edin.")
    except Exception as e:
        print(f"\nBeklenmeyen bir hata oluştu: {str(e)}")
        print("Hata detayları:", e.__class__.__name__)
    
    finally:
        print("\nİşlem tamamlandı.")
        print(f"Video kaydedildi: {output_path}")
        try:
            cap.release()
            out.release()
        except:
            pass
        cv2.destroyAllWindows()

# Programın ana giriş noktası
if __name__ == "__main__":
    main() 