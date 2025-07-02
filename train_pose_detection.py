# İnsan Pozu Tespiti Modeli Eğitim Scripti
# Bu script, YOLOv8 modelini insan pozu tespiti için eğitir
# Yazar: [Adınız]
# Tarih: [Tarih]

from ultralytics import YOLO

# Temel YOLOv8 poz tespit modelini yükle
model = YOLO('yolov8n-pose.pt')  # Anahtar nokta tespiti için uygun model

# Modeli eğit
model.train(
    # Veri seti yapılandırma dosyası
    data='datasets/pose-detection-full-body-final.v1i.yolov8/data.yaml',
    
    # Eğitim parametreleri
    epochs=10,  # Eğitim tur sayısı - İstediğiniz kadar artırabilirsiniz
    imgsz=640,  # Giriş görüntü boyutu
    device='cpu',  # İşlem birimi - CUDA yoksa CPU kullanılır
    patience=3  # Erken durdurma - 3 epoch boyunca gelişme olmazsa durdur
) 