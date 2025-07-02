# Şiddet ve Silah Tespiti Modeli Eğitim Scripti
# Bu script, YOLOv8 modelini şiddet ve silah tespiti için eğitir
# Yazar: [Adınız]
# Tarih: [Tarih]

from ultralytics import YOLO

# CUDA kullanımı otomatik olarak PyTorch tarafından yapılır (eğer GPU varsa)
# Temel YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # İsterseniz yolov8s.pt, yolov8m.pt, yolov8l.pt kullanabilirsiniz

# Modeli eğit
model.train(
    # Veri seti yapılandırma dosyası
    data='datasetler/Violence - Weapon Detection.v1i.yolov8/data.yaml',
    
    # Eğitim parametreleri
    epochs=10,  # Eğitim tur sayısı - İstediğiniz kadar artırabilirsiniz
    imgsz=640,  # Giriş görüntü boyutu
    device='cpu'  # İşlem birimi - CUDA yoksa CPU kullanılır
) 