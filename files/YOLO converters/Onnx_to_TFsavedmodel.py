# import onnx
# from onnx_tf.backend import prepare


# model_path = r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO converters\last    .onnx"
# onnx_model = onnx.load(model_path)


# tf_rep = prepare(onnx_model)

# tf_rep.export_graph(r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO converters\saved_model")


# import onnx
# from onnx_tf.backend import prepare

# # Load the ONNX model
# onnx_model = onnx.load(r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO converters\best.onnx")

# # Convert the ONNX model to TensorFlow format using onnx-tf
# tf_rep = prepare(onnx_model)

# # Save the converted model to TensorFlow SavedModel format
# tf_rep.export_graph(r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO converters\saved_model")

# # Print a confirmation message after saving the model
# print("Model has been successfully converted and saved to TensorFlow SavedModel format.")


from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO converters\best.pt")

# Export the model to TF SavedModel format
model.export(format="saved_model")  # creates '/yolov8n_saved_model'

# Load the exported TF SavedModel for inference
tf_savedmodel_model = YOLO("./yolov8n_saved_model")
results = tf_savedmodel_model("https://ultralytics.com/images/bus.jpg")