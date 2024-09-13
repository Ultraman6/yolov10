from ultralytics import YOLO

# Load the pre-trained YOLOv10-N model
model = YOLO("ultralytics/weights/yolov10s.pt")
results = model("ultralytics/assets/bus.jpg")
results[0].show()