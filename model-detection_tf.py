#=============================================================================================================
# model-detection_tf-FILE-for-RGB-Detection-from-simulation-by-Michael_Micah,ENGR.DR.MARTINS_OBASEKI (2024)-----
#=============================================================================================================

import pathlib
import sys
import glob
import os
import numpy as np
import tensorflow as tf
import time
import cv2
import random
import pygame  # Import Pygame for display and handling graphics

# Import CARLA
try:
    # Add Carla's Python API to the system path based on the Python version and OS
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla  # Import CARLA simulator API for vehicle and environment simulation

# Initialize global variables
actor_list = []  # List to store actors in the CARLA simulator
running = True  # Control variable for the main loop
detection_model = None  # Placeholder for the object detection model
category_index = None  # Placeholder for category labels (class names)

# Initialize Pygame for displaying camera views
pygame.init()
screen_width, screen_height = 1440, 960  # Set the size of the display window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("CARLA Camera Views")  # Set the window title

# Global variable to store images from each of the 9 cameras
camera_images = [None] * 9

# Function to process images from the CARLA simulator's cameras
def process_image(image, idx, infer=True):
    global running, camera_images
    # Convert the image to raw format for processing
    image.convert(carla.ColorConverter.Raw)
    # Convert the raw image data to a numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # Reshape the array to the desired image dimensions and extract RGB channels
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Extract RGB channels

    # Perform object detection inference if enabled
    if infer:
        processed_image = show_inference(detection_model, array)
    else:
        processed_image = array

    # Store the processed image in the global list
    camera_images[idx] = processed_image

    # Check if all camera images are available
    if all(img is not None for img in camera_images):
        # Combine the images into a single display
        combined_image = combine_images(camera_images)
        combined_surface = pygame.surfarray.make_surface(combined_image.swapaxes(0, 1))
        screen.blit(combined_surface, (0, 0))
        pygame.display.flip()

# Function to combine images from multiple cameras into a single image
def combine_images(images):
    h, w, _ = images[0].shape
    combined_image = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)

    # Top Row (Front Views)
    combined_image[0:h, 0:w] = images[0]  # Front left camera
    combined_image[0:h, w:w*2] = images[1]  # Front center camera
    combined_image[0:h, w*2:] = images[2]  # Front right camera

    # Middle Row (Side and Rear Views)
    combined_image[h:h*2, 0:w] = images[3]  # Left side camera
    combined_image[h*2:, w:w*2] = images[4]  # Top back camera
    combined_image[h:h*2, w*2:] = images[5]  # Right side camera

    # Bottom Row (Rear Views)
    combined_image[h*2:, 0:w] = images[8]  # Rear left camera
    combined_image[h:h*2, w:w*2] = images[7]  # Rear center camera
    combined_image[h*2:, w*2:] = images[6]  # Rear right camera

    return combined_image

# Function to clean up actors and resources when exiting
def cleanup():
    global actor_list
    for actor in actor_list:
        actor.destroy()  # Destroy each actor to free resources
    print("All actors cleaned up!")
    pygame.quit()  # Quit Pygame

# Function to run object detection inference on a single image
def show_inference(model, frame):
    if frame is None:
        print("Error: Frame is None in show_inference.")
        return None

    output_dict = run_inference_for_single_image(model, frame)
    frame = visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=1)

    return frame

# Function to load a TensorFlow model from a specified directory
def load_model(model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        print(f'Error: Model directory {model_dir} does not exist.')
        return None

    try:
        model = tf.saved_model.load(str(model_dir))  # Load the TensorFlow model
        print('Model loaded successfully.')
        return model
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        return None

# Define the model directory and load the object detection model
model_dir = r"path\to\your\saved_model"
detection_model = load_model(model_dir)
if detection_model is None:
    sys.exit('Model could not be loaded. Exiting.')

# Function to create a category index from a label map file
def create_category_index_from_labelmap(label_map_file_path):
    if not os.path.exists(label_map_file_path):
        raise FileNotFoundError(f'Label map file not found: {label_map_file_path}')

    label_map = {}
    with open(label_map_file_path, 'r') as f:
        label = None
        for line in f:
            if 'id:' in line:
                id = int(line.split('id:')[1].strip())
            if 'name:' in line:
                name = line.split('name:')[1].strip().replace("'", "")
                label_map[id] = {'id': id, 'name': name}
    return label_map

# Load the label map file and create a category index
label_map_file_path = r"path\to\your\label_map.pbtxt"
category_index = create_category_index_from_labelmap(label_map_file_path)

# Function to run inference on a single image using the loaded model
def run_inference_for_single_image(model, image):
    if model is None:
        raise ValueError("Model is not loaded. Please check the model path and load the model correctly.")

    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

# Function to reframe detection masks to fit the image size
def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
    def reframe_box_masks_to_image_masks_default(box_masks, boxes, image_height, image_width):
        num_boxes = boxes.shape[0]
        image_masks = np.zeros([num_boxes, image_height, image_width], dtype=np.float32)
        for i in range(num_boxes):
            box = boxes[i]
            box_mask = cv2.resize(box_masks[i], (int(box[3] - box[1]), int(box[2] - box[0])))
            image_masks[i, int(box[0]):int(box[2]), int(box[1]):int(box[3])] = box_mask
        return image_masks

    box_masks = tf.image.resize(box_masks, [image_height, image_width])
    return reframe_box_masks_to_image_masks_default(box_masks, boxes, image_height, image_width)

# Define a color mapping for each class ID for visualizing detections
class_color_map = {
    1: (255, 0, 0),    # Car - Red
    2: (0, 255, 0),    # Pedestrian - Green
    3: (0, 0, 255),    # Traffic light (Green left) - Blue
    4: (0, 255, 255),  # Traffic light (Green) - Yellow
    5: (255, 0, 255),  # Traffic light (Red) - Magenta
    6: (255, 165, 0),  # Traffic light (Red left) - Orange
    7: (0, 0, 0),      # Traffic light - Black
    8: (128, 0, 128),  # Truck - Purple
    9: (255, 192, 203),# Biker - Pink
    10: (255, 255, 0), # Traffic light (Yellow) - Cyan
    11: (128, 128, 0)  # Traffic light (Yellow left) - Olive
}

# Function to draw bounding boxes on the image based on detections
def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, display_str,
                                     class_id, thickness=1):
    if image is None:
        print("Error: Image is None in draw_bounding_box_on_image_array.")
        return None

    # Make the array writable
    image = np.ascontiguousarray(image)

    # Get the color based on the class ID
    color = class_color_map.get(class_id, (255, 255, 255))  # Default to white if class ID not found

    (im_width, im_height) = image.shape[1], image.shape[0]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    # Draw the bounding box
    pygame.draw.rect(image, color, pygame.Rect(int(left), int(top), int(right-left), int(bottom-top)), thickness)
    font = pygame.font.SysFont(None, 24)
    text_surface = font.render(display_str, True, color)
    image.blit(text_surface, (int(left), int(top) - 20))
    return image

# Function to visualize detection boxes and labels on the image
def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,
                                              instance_masks=None, use_normalized_coordinates=False,
                                              line_thickness=1):
    if image is None:
        print("Error: Image is None in visualize_boxes_and_labels_on_image_array.")
        return None

    for i in range(boxes.shape[0]):
        if scores[i] > 0.1:  # Set score threshold for displaying detections
            box = tuple(boxes[i].tolist())
            class_id = classes[i]
            if class_id not in category_index:
                continue
            class_name = category_index[class_id]['name']
            display_str = f'{class_name}: {int(100 * scores[i])}%'
            image = draw_bounding_box_on_image_array(image, box[0], box[1], box[2], box[3], display_str, class_id)
    return image

# Main function to initialize and run the CARLA simulation
def main():
    global actor_list, detection_model, category_index, running

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()  # Get the simulation world
    blueprint_library = world.get_blueprint_library()  # Access the blueprint library for actors
    
    # Get the blueprint for a Tesla Model 3 vehicle and set the spawn point
    vehicle_bp = blueprint_library.filter('model3')[0]
    # spawn_point = carla.Transform(carla.Location(x=110, y=8.2, z=0.5), carla.Rotation(yaw=0))
    spawn_point = random.choice(world.get_map().get_spawn_points())
    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)  # Spawn the vehicle
    
    if vehicle is None:
        print("Vehicle could not be spawned.")
        return
    actor_list.append(vehicle)  # Add the vehicle to the list of actors

    # Set the vehicle to autopilot mode
    vehicle.set_autopilot(True)

    # Define camera attributes
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '150')
    camera_bp.set_attribute('image_size_y', '150')
    camera_bp.set_attribute('fov', '90')

    # Define the camera setup for different views around the vehicle
    camera_transforms = [
        carla.Transform(carla.Location(x=2.0, y=-0.5, z=1.4), carla.Rotation(yaw=-45)),  # Front left camera
        carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0)),            # Front center camera
        carla.Transform(carla.Location(x=2.0, y=0.5, z=1.4), carla.Rotation(yaw=45)),    # Front right camera

        carla.Transform(carla.Location(x=1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-90)), # Left side camera
        carla.Transform(carla.Location(x=-1.0, y=0.75, z=1.4), carla.Rotation(yaw=135)), # Top back camera
        carla.Transform(carla.Location(x=1.0, y=0.75, z=1.4), carla.Rotation(yaw=90)),   # Right side camera

        carla.Transform(carla.Location(x=-1.0, z=1.4), carla.Rotation(yaw=180)),         # Rear center camera
        carla.Transform(carla.Location(x=-1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-135)),# Rear left camera
        carla.Transform(carla.Location(x=-6.0, z=5.0), carla.Rotation(pitch=-30)),       # Rear right camera
    ] 

    # Spawn and listen to each camera
    camera_list = []
    for idx, transform in enumerate(camera_transforms):
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        camera.listen(lambda image, idx=idx: process_image(image, idx, infer=True))  # Process each camera's image
        actor_list.append(camera)
        camera_list.append(camera)
        
    try:
        while running:  # Main loop for the simulation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Handle quitting the application
                    running = False
            # Sleep to reduce CPU usage
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        running = False
    finally:
        cleanup()  # Clean up actors and resources when exiting

if __name__ == '__main__':
    main()  # Entry point for script execution
