from ultralytics import YOLO

# Eğitilmiş Violence-Weapon Detection modelini yükle
model = YOLO('runs/detect/train5/weights/best.pt')

# Video üzerinde test et
results = model('İçerde - Akıllarda İz Bırakan Dövüş Sahnesi.mp4', show=True, save=True)  # show=True ile gerçek zamanlı gösterim, save=True ile kaydetme 