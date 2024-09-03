# import tensorflow as tf 

# # Load the SavedModel from the correct directory path
# model = tf.saved_model.load(r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO to TFsavedmodel\saved_model")

# # Create a TFLiteConverter object from the loaded model
# converter = tf.lite.TFLiteConverter.from_saved_model(r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO to TFsavedmodel\saved_model")

# # Perform optimizations (optional but recommended)
# # Post-training quantization is useful for reducing model size and improving performance
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # Specify the supported operations (this allows conversion of more complex models)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# # Specify the supported types (for example, tf.float32)
# converter.target_spec.supported_types = [tf.float32]

# # Convert the model to TFLite format
# tflite_model = converter.convert()

# # Save the converted TFLite model to a file
# output_path = r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO to TFsavedmodel\model.tflite"
# with open(output_path, 'wb') as f:
#     f.write(tflite_model)


from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolov8n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("yolov8n_float32.tflite")

# Run inference
results = tflite_model("https://ultralytics.com/images/bus.jpg")