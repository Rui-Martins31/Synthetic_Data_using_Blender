# Training

Run this command to start training the YOLO Model using your dataset:
```bash
yolo segment train data=synthetic_data/data.yaml model=yolo11n-seg.pt imgsz=640 device=mps
```