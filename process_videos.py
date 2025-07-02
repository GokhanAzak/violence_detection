# Video İşleme Uygulaması
# Bu script, videolarda şiddet ve silah tespiti yapar
# Yazar: [Adınız]
# Tarih: [Tarih]

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
from pathlib import Path

def create_output_folder():
    """
    İşlenen videoların kaydedileceği klasörü oluşturur.
    Returns:
        str: Oluşturulan klasörün yolu
    """
    output_folder = "islenen_videolar"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Çıktı klasörü oluşturuldu: {output_folder}")
    return output_folder

def get_output_filename(original_video_path, output_folder):
    """
    İşlenen video için benzersiz bir dosya adı oluşturur.
    
    Args:
        original_video_path (str): Orijinal video dosyasının yolu
        output_folder (str): Çıktı klasörünün yolu
    
    Returns:
        str: Oluşturulan yeni dosya yolu
    """
    # Orijinal dosya adını al
    base_name = os.path.basename(original_video_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Zaman damgası ekle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Yeni dosya adını oluştur
    new_filename = f"{name_without_ext}_{timestamp}.mp4"
    
    # Çıktı klasörü ile birleştir
    return os.path.join(output_folder, new_filename)

def process_video(video_path, show_preview=True):
    """
    Videoyu işler ve şiddet/silah tespiti yapar.
    
    Args:
        video_path (str): İşlenecek video dosyasının yolu
        show_preview (bool): Önizleme penceresini gösterip göstermeme
    """
    # Çıktı klasörünü oluştur ve dosya yolunu al
    output_folder = create_output_folder()
    output_path = get_output_filename(video_path, output_folder)
    
    # YOLO modelini CPU ile yükle
    model = YOLO('runs/detect/train5/weights/best.pt')  # Şiddet ve silah tespiti modeli
    model.to('cpu')
    
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
    
    # Video yazıcıyı ayarla
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Önizleme penceresini ayarla
    if show_preview:
        window_name = 'İşleme Önizlemesi'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    
    # İlerleme takibi için değişkenler
    frame_count = 0
    start_time = time.time()
    violence_detected = False
    weapon_detected = False
    
    try:
        while cap.isOpened():
            # Kareyi oku
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # İlerleme durumunu hesapla ve göster
            progress = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\rİşleniyor: {progress:.1f}% | Kare {frame_count}/{total_frames} | FPS: {fps_processing:.1f}", end="")
            
            # Kareyi işle
            results = model(frame)  # Şiddet ve silah tespiti
            
            # Görselleştirme için kopya oluştur
            output_frame = frame.copy()
            
            # Tespit sonuçlarını işle ve görselleştir
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Tespit kutusu koordinatlarını al
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Güven skoru kontrolü
                    if conf > 0.5:
                        class_name = result.names[cls]
                        
                        # Sınıfa göre renk ve durum belirleme
                        if "Violence" in class_name:
                            color = (0, 0, 255)  # Kırmızı - Şiddet
                            violence_detected = True
                        elif "Weapon" in class_name or "weapon" in class_name.lower():
                            color = (255, 0, 0)  #  - Silah
                            weapon_detected = True
                        elif "Knife" in class_name or "knife" in class_name.lower():
                            color = (255, 165, 0)  #  - Bıçak
                            weapon_detected = True
                        else:
                            color = (0, 255, 0)  
                        
                        # Tespit kutusunu çiz
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Etiket metnini hazırla
                        label = f'{class_name}: {conf:.2f}'
                        
                        # Etiket arka planını çiz
                        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(output_frame, (x1, y1-20), (x1 + label_width, y1), color, -1)
                        
                        # Etiket metnini yaz
                        cv2.putText(output_frame, label, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # İşlenmiş kareyi kaydet
            out.write(output_frame)
            
            # Önizleme penceresini göster
            if show_preview:
                cv2.imshow(window_name, output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nKullanıcı tarafından iptal edildi.")
                    break
    
    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")
    
    finally:
        # İşlem sonuçlarını göster
        print("\nİşlem tamamlandı.")
        if violence_detected:
            print("UYARI: Videoda şiddet içeriği tespit edildi!")
        if weapon_detected:
            print("UYARI: Videoda silah/bıçak tespit edildi!")
        if not violence_detected and not weapon_detected:
            print("Videoda şiddet veya silah/bıçak tespit edilmedi.")
        print(f"Video kaydedildi: {output_path}")
        
        # Kaynakları serbest bırak
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

def main():
    """
    Ana program fonksiyonu
    """
    try:
        # Varsayılan video dosyasını kullan
        video_path = "helinmamajan.mp4"
        
        # Dosyanın varlığını kontrol et
        if not os.path.exists(video_path):
            raise ValueError(f"Video dosyası bulunamadı: {video_path}")
        
        # Videoyu işle
        process_video(video_path, show_preview=True)
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

# Programın ana giriş noktası
if __name__ == "__main__":
    main() 