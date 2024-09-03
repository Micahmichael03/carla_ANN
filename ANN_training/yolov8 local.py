# #nvidia-smi

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=10, batch=1, optimizer="Adam")  # train the model
  
# import torch
# from ultralytics import YOLO

# # Check if CUDA is available and set the device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# # Load the YOLO model
# model = YOLO("yolov8n.pt").to(device)

# # Train the model
# results = model.train(data="config.yaml", epochs=10, batch=36, optimizer="Adam", device=device)