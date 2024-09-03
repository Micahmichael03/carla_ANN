import pathlib
import sys
import glob
import os
import numpy as np
import cv2
import tensorflow as tf
import time
import random
import argparse
import math
# from agents.navigation.basic_agent import BasicAgent
# from agents.navigation.behavior_agent import BehaviorAgent

# Import CARLA
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# from agents.navigation.behavior_agent import BehaviorAgent
import carla



# Initialize global variables
actor_list = []
running = True
detection_model = None
category_index = None
# Example initialization



# def get_random_spawn_point(world):
#     """Gets a random spawn point from the world."""
#     spawn_points = world.get_map().get_spawn_points()
#     random_index = random.randint(0, len(spawn_points) - 1)
#     return spawn_points[random_index]

def get_specific_spawn_point(world, x=150, y=10, z=0.5, yaw=0.0, pitch=0.0, roll=0.0):
  """Gets a spawn point at a specific location and orientation.
    x=150, y=10, z=0.5: This places the vehicle at a position 150 meters forward, 
    10 meters to the right, and 0.5 meters above the ground. 
    Adjust these values based on your desired location.
    yaw=0.0, pitch=0.0, roll=0.0: 
    This sets the initial orientation to face forward without any tilt.
  """
  return carla.Transform(location=carla.Location(x=x, y=y, z=z),
                         rotation=carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))
  
def get_final_destination(x=200, y=10, z=0.5):
    """Sets the final destination coordinates.

    Args:
        x: The x-coordinate of the destination. Defaults to 300.
        y: The y-coordinate of the destination. Defaults to -50.
        z: The z-coordinate of the destination. Defaults to 0.5.

    Returns:
        A carla.Location object representing the final destination.
    """
    return carla.Location(x=x, y=y, z=z)

def has_reached_destination(vehicle, destination_location, tolerance=2.0):
    """Check if the vehicle has reached the destination within a tolerance distance."""
    vehicle_location = vehicle.get_location()
    distance = math.sqrt((vehicle_location.x - destination_location.x) ** 2 +
                         (vehicle_location.y - destination_location.y) ** 2 +
                         (vehicle_location.z - destination_location.z) ** 2)
    return distance <= tolerance

def draw_route(world, current_location, destination_location):
    """Draws a route from the current location to the destination on the CARLA map."""
    debug = world.debug
    waypoint = world.get_map().get_waypoint(current_location)
    route = [waypoint]
    
    # Get the route waypoints from the current location to the destination
    while waypoint.transform.location.distance(destination_location) > 2.0:
        waypoint = waypoint.next(2.0)[0]
        route.append(waypoint)
    
    # Draw the waypoints on the map
    for wp in route:
        debug.draw_string(wp.transform.location, '|', draw_shadow=False,
                          color=carla.Color(r=0, g=255, b=0), life_time=2.0,
                          persistent_lines=True)
        debug.draw_string(destination_location, 'XXXXX', draw_shadow=False,
                      color=carla.Color(r=255, g=0, b=0), life_time=2.0,
                      persistent_lines=True)

# Add global variable to store images from each camera
camera_images = [None] * 4
 
def process_image(image, idx):
    global running, camera_images
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Extract RGB channels
    processed_image = show_inference(detection_model, array)

    # Store the processed image in the global list
    camera_images[idx] = processed_image

    # Check if all images are available (not None)
    if all(img is not None for img in camera_images):
        combined_image = combine_images(camera_images)
        cv2.imshow('Combined Camera Feed', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

def combine_images(images):
    # Ensure all images are the same size
    h, w, _ = images[0].shape
    combined_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # Place images in the 2x2 grid
    combined_image[0:h, 0:w] = images[0]  # Top-left
    combined_image[0:h, w:w * 2] = images[1]  # Top-right
    combined_image[h:h * 2, 0:w] = images[2]  # Bottom-left
    combined_image[h:h * 2, w:w * 2] = images[3]  # Bottom-right
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return
    
    return combined_image


def cleanup():
    global actor_list
    for actor in actor_list:
        actor.destroy()
    print("All actors cleaned up!")
    cv2.destroyAllWindows()

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

def load_model(model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        print(f'Error: Model directory {model_dir} does not exist.')
        return None

    try:
        model = tf.saved_model.load(str(model_dir))
        print('Model loaded successfully.')
        return model
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        return None

model_dir = r"C:\Carla\CARLA_0.9.8\WindowsNoEditor\PythonAPI\examples\saved_model"
detection_model = load_model(model_dir)
if detection_model is None:
    sys.exit('Model could not be loaded. Exiting.')

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

label_map_file_path = r"C:\Carla\CARLA_0.9.8\WindowsNoEditor\PythonAPI\examples\driving_label_map.pbtxt"
category_index = create_category_index_from_labelmap(label_map_file_path)

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

# Define a color mapping for each class ID
class_color_map = {
    1: (255, 0, 0),  # car - Red
    2: (0, 255, 0),  # pedestrian - Green
    3: (0, 0, 255),  # trafficLight-GreenLeft - Blue
    4: (0, 255, 255),  # trafficLight-Green - Yellow
    5: (255, 0, 255),  # trafficLight-Red - Magenta
    6: (255, 165, 0),  # trafficLight-RedLeft - Orange
    7: (0, 0, 0),  # trafficLight - Black
    8: (128, 0, 128),  # truck - Purple
    9: (255, 192, 203),  # biker - Pink
    10: (255, 255, 0),  # trafficLight-Yellow - Cyan
    11: (128, 128, 0)   # trafficLight-YellowLeft - Olive
}

def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, display_str,
                                     class_id, thickness=1):
    if image is None:
        print("Error: Image is None in draw_bounding_box_on_image_array.")
        return None

    # Make the array writable
    image = np.ascontiguousarray(image)

    # Get the color based on the class_id
    color = class_color_map.get(class_id, (255, 255, 255))  # Default to white if class_id not found

    (im_width, im_height) = image.shape[1], image.shape[0]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, thickness)
    cv2.putText(image, display_str, (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

# Modify the visualize_boxes_and_labels_on_image_array function to pass class_id
def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,
                                              instance_masks=None, use_normalized_coordinates=False,
                                              line_thickness=1):
    if image is None:
        print("Error: Image is None in visualize_boxes_and_labels_on_image_array.")
        return None

    for i in range(boxes.shape[0]):
        if scores[i] > 0.1: #change the score high model prediction
            box = tuple(boxes[i].tolist())
            class_id = classes[i]
            if class_id not in category_index:
                continue
            class_name = category_index[class_id]['name']
            display_str = f'{class_name}: {int(100 * scores[i])}%'
            image = draw_bounding_box_on_image_array(image, box[0], box[1], box[2], box[3], display_str, class_id)
    return image

# Main function to initialize and run the simulation
def main():
    global actor_list, detection_model, category_index, running, agent, vehicle

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    
    # Get vehicle blueprint and spawn point
    vehicle_bp = blueprint_library.filter('cybertruck')[0]
    # spawn_points =  get_specific_spawn_point(world)
    start_location = carla.Transform(carla.Location(x=150, y=10, z=0.5))
    print(start_location)
    vehicle = world.spawn_actor(vehicle_bp, start_location)
    print(vehicle)
    
    # # Get the default final destination
    destination = get_final_destination()
    print(destination)
    # agent = BasicAgent(vehicle)
    
    # agent = BehaviorAgent(vehicle, behavior='aggressive')
    # destination_location = carla.Location(x=200, y=10, z=0.5)
    # print(destination_location)
    # agent.set_destination((destination_location.x, destination_location.y, destination_location.z))

    
    if vehicle is None:
        print("Vehicle could not be spawned.")
        return
    actor_list.append(vehicle)

    # Set autopilot
    vehicle.set_autopilot(True)

    # # Convert destination_location to a tuple of coordinates (x, y, z)
    # destination_coords = (destination_location.x, destination_location.y, destination_location.z)
    
    # # Set the destination for the agent
    # agent.set_destination(destination_coords)

    # Camera attributes
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')

    # Define camera locations and orientations relative to the vehicle
    camera_transforms = [
        carla.Transform(carla.Location(x=1.5, z=2.4)),  # Front camera
        carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180)),  # Rear camera
        carla.Transform(carla.Location(y=-1.5, x=0.0, z=2.4), carla.Rotation(yaw=-90)),  # Left-side camera
        carla.Transform(carla.Location(y=1.5, x=0.0, z=2.4), carla.Rotation(yaw=90))  # Right-side camera
    ]

    # Spawn and listen to each camera
    for idx, transform in enumerate(camera_transforms):
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        camera.listen(lambda image, idx=idx: process_image(image, idx))
        actor_list.append(camera)
        
    try:
        while running:
            # World tick for synchronization
            # world.tick()
            
            # vehicle.apply_control()
            
            # Draw the planned route on the map
            draw_route(world, vehicle.get_location(), get_final_destination)
            
            # if agent.set_destination(destination_coords):
            #     vehicle.apply_control(agent.run_step())
            #     print("The target has been reached, stopping the simulation")
            #     break
            # vehicle.apply_control(agent.run_step())
            if has_reached_destination(vehicle, get_final_destination):
                print("The target has been reached, stopping the simulation")
                break
    
    except KeyboardInterrupt:
        running = False
        
    finally:
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

if __name__ == '__main__':
    main()
    