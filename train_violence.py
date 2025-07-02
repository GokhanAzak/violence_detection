# Şiddet Tespiti Modeli Eğitim Scripti
# Bu script, YOLOv8 modelini şiddet tespiti için eğitir
# Yazar: [Adınız]
# Tarih: [Tarih]

from ultralytics import YOLO

# Temel YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # Nesne tespiti için uygun model

# Modeli eğit
model.train(
    # Veri seti yapılandırma dosyası
    data='datasets/Violence - Weapon Detection.v1i.yolov8/data.yaml',
    
    # Eğitim parametreleri
    epochs=10,  # Eğitim tur sayısı - İstediğiniz kadar artırabilirsiniz
    imgsz=640,  # Giriş görüntü boyutu
    device='cpu',  # İşlem birimi - CUDA yoksa CPU kullanılır
    patience=3  # Erken durdurma - 3 epoch boyunca gelişme olmazsa durdur
) 