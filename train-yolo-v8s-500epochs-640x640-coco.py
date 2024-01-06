from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(
    data="../datasets/SplitPrivateDatasetRemapped.yaml",
    epochs=500,
    batch=-1, # auto batch
    device="cuda:0",
    imgsz=640
)