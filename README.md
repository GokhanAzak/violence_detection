# violence_detection

This project is a real-time detection system built using YOLOv8. It is capable of identifying the following classes:

- Gun
- Knife
- NoViolence (normal human behavior)
- Violence (aggressive or harmful behavior)

## Technologies Used

- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- PyTorch  

## File Descriptions

- `train_*.py` – Scripts used to train models for pose, violence, and weapon detection  
- `test_*.py` – Scripts for testing videos, webcam streams, or images  
- `webcam_detector.py` – Real-time detection via webcam  
- `yolov8n.pt` – YOLOv8 model weights  
- `yolov8n-pose.pt` – Model weights for pose detection

## Sample Videos

The following sample videos are included in the repository:

- `videos/esnaf.mp4`  
- `videos/mamajan.mp4`  

These files demonstrate how the model performs in different scenarios.
