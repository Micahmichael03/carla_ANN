from ultralytics import YOLO
# Load a model
model = YOLO('last.pt')  # pretrained YOLOv8n model

success = model.export(format='onnx')

print(success)