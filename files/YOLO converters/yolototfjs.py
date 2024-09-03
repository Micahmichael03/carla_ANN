import os
import tensorflow as tf

# Load the SavedModel from the correct directory path
model = tf.saved_model.load(r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO to TFsavedmodel\saved_model")

# Convert the SavedModel to TensorFlow.js format using the command line
output_dir = r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO to TFsavedmodel\tfjs_model"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Construct the command without using f-strings
input_path = r"C:\CARLA_0.9.5\PythonAPI\examples\yolo-model\YOLO to TFsavedmodel\saved_model"
command = "tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model " + input_path + " " + output_dir

# Run the conversion using the TensorFlow.js converter
os.system(command)

print(f"Model successfully converted to TensorFlow.js and saved to {output_dir}")
