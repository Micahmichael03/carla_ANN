import pathlib
import sys
import glob
import os
import numpy as np
import cv2
import tensorflow as tf
import time
import random
import math
 
# Import CARLA
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla 

# Initialize global variables
actor_list = []
running = True
detection_model = None
category_index = None

def get_specific_spawn_point(world, x=110, y=8.2, z=0.5, yaw=0.0, pitch=0.0, roll=0.0):
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

def draw_route(world, current_location, destination):
    """Draws a route from the current location to the destination on the CARLA map."""
    debug = world.debug
    waypoint = world.get_map().get_waypoint(current_location)
    route = [waypoint]
    
    # Get the route waypoints from the current location to the destination
    while waypoint.transform.location.distance(destination) > 2.0:
        waypoint = waypoint.next(2.0)[0]
        route.append(waypoint)
    
    # Draw the waypoints on the map
    for wp in route:
        debug.draw_string(wp.transform.location, '|', draw_shadow=False,
                          color=carla.Color(r=0, g=255, b=0), life_time=2.0,
                          persistent_lines=True)
        debug.draw_string(destination, 'XXXXX', draw_shadow=False,
                      color=carla.Color(r=255, g=0, b=0), life_time=2.0,
                      persistent_lines=True)
        
# Add global variable to store images from each camera
camera_images = [None] * 9

def process_image(image, idx, infer=True):
    global running, camera_images
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Extract RGB channels
    if infer:
        processed_image = show_inference(detection_model, array)
    else:
        processed_image = array

    # Store the processed image in the global list
    camera_images[idx] = processed_image

    # Check if all images are available (not None)
    if all(img is not None for img in camera_images):
        combined_image = combine_images(camera_images)
        cv2.imshow('Combined Camera Feed', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

def combine_images(images):
    h, w, _ = images[0].shape
    combined_image = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)

    # Top Row (Front Views)
    combined_image[0:h, 0:w] = images[0]  # Front left cam 1
    combined_image[0:h, w:w*2] = images[1]  # Front center cam 2
    combined_image[0:h, w*2:] = images[2]  # Front right cam 3

    # Middle Row (Side and Rear Views)
    combined_image[h:h*2, 0:w] = images[3]  # Left side cam 7
    combined_image[h*2:, w:w*2] = images[4]  # Top back cam 8
    combined_image[h:h*2, w*2:] = images[5]  # Right side cam 6

    # Bottom Row (Top Back and Rear Views)
    combined_image[h*2:, 0:w] = images[8]  # Rear left cam 5
    combined_image[h:h*2, w:w*2] = images[7]  # Rear center cam 4
    combined_image[h*2:, w*2:] = images[6]  # Rear right cam 9

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
    global actor_list, detection_model, category_index, running

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    # Get vehicle blueprint and spawn point
    vehicle_bp = blueprint_library.filter('model3')[0]
    # spawn_points =  get_specific_spawn_point(world)
    start_points = get_specific_spawn_point(world)
    print(start_points)
    
    # Get the default final destination
    destination = get_final_destination()
    print(destination)
    
    vehicle = world.spawn_actor(vehicle_bp, start_points)
    print(vehicle)
    
    if vehicle is None:
        print("Vehicle could not be spawned.")
        return
    actor_list.append(vehicle)

    # Set autopilot
    vehicle.set_autopilot(True)

    # Camera attributes
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '480')
    camera_bp.set_attribute('image_size_y', '320')
    camera_bp.set_attribute('fov', '90')

    # # Define camera locations and orientations relative to the cybertruck vehicle
    # camera_transforms = [
    #     carla.Transform(carla.Location(x=1.5, z=2.4)),  # Front camera
    #     carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180)),  # Rear camera
    #     carla.Transform(carla.Location(y=-1.5, x=0.0, z=2.4), carla.Rotation(yaw=-90)),  # Left-side camera
    #     carla.Transform(carla.Location(y=1.5, x=0.0, z=2.4), carla.Rotation(yaw=90)),  # Right-side camera
    #     carla.Transform(carla.Location(x=-3.0, z=4.0), carla.Rotation(yaw=180)),  # Top-back view camera
    #     carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4)),  # Extra Camera 1
    #     carla.Transform(carla.Location(x=0.0, y=1.0, z=2.4)),  # Extra Camera 2
    #     carla.Transform(carla.Location(x=0.0, y=-1.0, z=2.4))  # Extra Camera 3a
    # ]
    
    # Tesla model3 Camera setup
    camera_transforms = [
        carla.Transform(carla.Location(x=2.0, y=-0.5, z=1.4), carla.Rotation(yaw=-45)),  # Front left cam 1
        carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0)),            # Front center cam 2
        carla.Transform(carla.Location(x=2.0, y=0.5, z=1.4), carla.Rotation(yaw=45)),    # Front right cam 3

        carla.Transform(carla.Location(x=1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-90)),  # Left side cam 7
        carla.Transform(carla.Location(x=-1.0, y=0.75, z=1.4), carla.Rotation(yaw=135)),   # Top back cam 8
        carla.Transform(carla.Location(x=1.0, y=0.75, z=1.4), carla.Rotation(yaw=90)),    # Right side cam 6

        carla.Transform(carla.Location(x=-1.0, z=1.4), carla.Rotation(yaw=180)),          # Rear center cam 4
        carla.Transform(carla.Location(x=-1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-135)), # Rear left cam 5    
        carla.Transform(carla.Location(x=-6.0, z=5.0), carla.Rotation(pitch=-30)),         # Rear right cam 9
    ] 

    # Spawn and listen to each camera
    camera_list = []
    for idx, transform in enumerate(camera_transforms):
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        camera.listen(lambda image, idx=idx: process_image(image, idx, infer=True))
        actor_list.append(camera)
        camera_list.append(camera)
        
    try:
        while running:
            # # World tick for synchronization
            # world.tick()
            
            # Draw the planned route on the map
            draw_route(world, vehicle.get_location(), destination)

            # Update the vehicle control
            control = carla.VehicleControl()

            # Get the current location and orientation of the vehicle
            vehicle_location = vehicle.get_location()
            vehicle_transform = vehicle.get_transform()
            vehicle_rotation = vehicle_transform.rotation
            vehicle_yaw = vehicle_rotation.yaw

            # Calculate the direction to the destination
            direction_vector = np.array([destination.x - vehicle_location.x, destination.y - vehicle_location.y])
            direction_magnitude = np.linalg.norm(direction_vector)
            if direction_magnitude != 0:
                direction_unit_vector = direction_vector / direction_magnitude
            else:
                direction_unit_vector = direction_vector

            # Calculate the current direction of the vehicle
            vehicle_direction = np.array([math.cos(math.radians(vehicle_yaw)), math.sin(math.radians(vehicle_yaw))])

            # Calculate the angle between the vehicle's direction and the direction to the destination
            dot_product = np.dot(vehicle_direction, direction_unit_vector)
            angle_to_destination = math.degrees(math.acos(dot_product))

            # Cross product to determine the direction of the turn
            cross_product = np.cross(vehicle_direction, direction_unit_vector)

            # Lane keeping logic
            waypoint = world.get_map().get_waypoint(vehicle_location)
            lane_center = waypoint.transform.location
            lane_direction = np.array([math.cos(math.radians(waypoint.transform.rotation.yaw)), 
                                    math.sin(math.radians(waypoint.transform.rotation.yaw))])

            # Calculate deviation from lane center
            lane_deviation_vector = np.array([vehicle_location.x - lane_center.x, vehicle_location.y - lane_center.y])
            lane_deviation_magnitude = np.linalg.norm(lane_deviation_vector)

            # Correct vehicle direction to stay in the lane
            lane_dot_product = np.dot(vehicle_direction, lane_direction)
            lane_cross_product = np.cross(vehicle_direction, lane_direction)
            lane_angle = math.degrees(math.acos(lane_dot_product))

            if lane_cross_product > 0:
                control.steer = -min(lane_angle / 90.0, 1.0)  # Turn left to stay in lane
            else:
                control.steer = min(lane_angle / 90.0, 1.0)  # Turn right to stay in lane

            # Obstacle detection and avoidance logic
            obstacles = world.get_actors().filter('vehicle.*')  # Get all vehicles in the simulation
            min_distance = 15.0  # Minimum distance to consider an obstacle

            for obstacle in obstacles:
                if obstacle.id != vehicle.id:  # Avoid self-detection
                    obstacle_location = obstacle.get_location()
                    distance_to_obstacle = vehicle_location.distance(obstacle_location)

                    if distance_to_obstacle < min_distance:
                        # Adjust steering and throttle to avoid the obstacle
                        avoidance_vector = np.array([obstacle_location.x - vehicle_location.x,
                                                    obstacle_location.y - vehicle_location.y])
                        avoidance_magnitude = np.linalg.norm(avoidance_vector)
                        avoidance_unit_vector = avoidance_vector / avoidance_magnitude

                        # Calculate a new direction to avoid the obstacle
                        avoidance_direction = vehicle_direction - avoidance_unit_vector
                        new_angle = math.degrees(math.atan2(avoidance_direction[1], avoidance_direction[0]))

                        if cross_product > 0:
                            control.steer = -min(new_angle / 90.0, 1.0)  # Turn left to avoid
                        else:
                            control.steer = min(new_angle / 90.0, 1.0)  # Turn right to avoid

                        control.throttle = 0.2  # Slow down to avoid collision
                        break
            else:
                # Normal steering logic based on the angle to destination
                if cross_product > 0:
                    control.steer = -min(angle_to_destination / 90.0, 1.0)  # Turn left
                else:
                    control.steer = min(angle_to_destination / 90.0, 1.0)  # Turn right

                # Throttle and brake logic based on distance to the destination
                if direction_magnitude > 20:
                    control.throttle = 0.5  # Increase speed if far from the destination
                    control.brake = 0.0
                elif direction_magnitude > 5:
                    control.throttle = 0.3  # Reduce speed as you get closer to the destination
                    control.brake = 0.0
                else:
                    control.throttle = 0.0
                    control.brake = 1.0  # Stop the vehicle when close to the destination

            # Apply control to the vehicle
            vehicle.apply_control(control)

            # Stop vehicle if it reaches the destination
            if vehicle_location.distance(destination) < 2.0:
                control.throttle = 0.0
                control.brake = 1.0  # Full stop
                vehicle.apply_control(control)
            # Check if the vehicle has reached the final destination
            if has_reached_destination(vehicle, destination):
                    print("Final destination reached!")
                    running = False
            else:
                    time.sleep(0.5)  # Sleep to reduce CPU usage
    
    except KeyboardInterrupt:
        running = False
    finally:
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

if __name__ == '__main__':
    main() 
    