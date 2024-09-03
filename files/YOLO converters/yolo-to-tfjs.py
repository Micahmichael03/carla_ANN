from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Export the model to TF.js format
model.export(format="tfjs")  # creates '/yolov8n_web_model'

# Load the exported TF.js model
tfjs_model = YOLO("./yolov8n_web_model")

# Run inference
results = tfjs_model("https://ultralytics.com/images/bus.jpg")