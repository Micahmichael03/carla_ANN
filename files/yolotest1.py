from ultralytics import YOLO
import torch 
import pathlib
import sys
import glob
import os
import numpy as np
import cv2
import time
import math
import yaml
import random

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
# Add global variable to store images from each camera
camera_images = [None] * 9
load_model = None
# Set image dimensions for the camera.
IM_WIDTH = 360  # The width of the image.
IM_HEIGHT = 360  # The height of the image.
Show_path_trajectory = False  # A boolean flag to control whether the path trajectory should be shown.
blue = carla.Color(47, 210, 231)  # Blue color.
red = carla.Color(255, 0, 0)  # Red color.


class ModelLoader:
    def __init__(self, model_dir: str, yolo_yaml_file_path: str):
        """
        Initialize the ModelLoader class.

        :param model_dir: Directory where the YOLO model is saved.
        :param yolo_yaml_file_path: Path to the YOLO YAML file containing class labels.
        """
        self.model_dir = pathlib.Path(model_dir)
        self.yolo_yaml_file_path = yolo_yaml_file_path
        self.model = self.load_model()
        self.category_index = self.create_category_index_from_yolo_yaml()

        if self.model is None:
            sys.exit('Model could not be loaded. Exiting.')

    def load_model(self):
        """
        Load the YOLOv8 model from the specified directory.

        :return: Loaded YOLOv8 model, or None if loading fails.
        """
        if not self.model_dir.exists():
            print(f'Error: Model directory {self.model_dir} does not exist.')
            return None

        try:
            # Load the YOLOv8 model
            model = YOLO(str(self.model_dir))
            print('Model loaded successfully.')
            return model
        except Exception as e:
            print(f'Error loading model: {str(e)}')
            return None

    def create_category_index_from_yolo_yaml(self):
        """
        Create a category index mapping from the YOLO YAML file.

        :return: Dictionary mapping class IDs to class names.
        """
        if not os.path.exists(self.yolo_yaml_file_path):
            raise FileNotFoundError(f'YOLO YAML file not found: {self.yolo_yaml_file_path}')

        with open(self.yolo_yaml_file_path, 'r') as f:
            yolo_data = yaml.safe_load(f)

        label_map = {}
        for idx, name in enumerate(yolo_data['names']):
            label_map[idx] = {'id': idx, 'name': name}

        print("Category index created successfully.")
        return label_map

# Example Usage:
model_dir = r"C:\CARLA_0.9.5\PythonAPI\examples\models\weights\best.pt"
yolo_yaml_file_path = r"C:\CARLA_0.9.5\PythonAPI\examples\models\labels.yaml"
model_loader = ModelLoader(model_dir, yolo_yaml_file_path)
detection_model = model_loader.model
category_index = model_loader.category_index

# def get_specific_spawn_point(world, x=110, y=8.2, z=0.5, yaw=0.0, pitch=0.0, roll=0.0):
#     return carla.Transform(location=carla.Location(x=x, y=y, z=z),
#                         rotation=carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))

# def get_final_destination(x=200, y=10, z=0.5):
#     return carla.Location(x=x, y=y, z=z)

# def has_reached_destination(vehicle, destination_location, tolerance=2.0):
#     """Check if the vehicle has reached the destination within a tolerance distance."""
#     vehicle_location = vehicle.get_location()
#     distance = math.sqrt((vehicle_location.x - destination_location.x) ** 2 +
#                         (vehicle_location.y - destination_location.y) ** 2 +
#                         (vehicle_location.z - destination_location.z) ** 2)
#     return distance <= tolerance

# def draw_route(world, current_location, destination, waypoint_interval=2.0, route_color=carla.Color(r=0, g=255, b=0), destination_color=carla.Color(r=255, g=0, b=0)):
#     """
#     Draws a route from the current location to the destination on the CARLA map.

#     :param world: The CARLA world object.
#     :param current_location: The starting location for the route.
#     :param destination: The end location for the route.
#     :param waypoint_interval: The distance between each waypoint.
#     :param route_color: The color used to draw the route.
#     :param destination_color: The color used to draw the destination marker.
#     """
#     debug = world.debug
#     waypoint = world.get_map().get_waypoint(current_location)
#     route = [waypoint]

#     # Get the route waypoints from the current location to the destination
#     while waypoint.transform.location.distance(destination) > waypoint_interval:
#         waypoint = waypoint.next(waypoint_interval)[0]
#         route.append(waypoint)

#     # Draw the waypoints on the map
#     for wp in route:
#         debug.draw_string(wp.transform.location, '|', draw_shadow=False,
#                           color=route_color, life_time=2.0,
#                           persistent_lines=True)

#     # Draw a marker at the destination
#     debug.draw_string(destination, 'XXXXX', draw_shadow=False,
#                       color=destination_color, life_time=2.0,
#                       persistent_lines=True)

#     print("Route drawn from current location to destination.")
   
# # Function to draw an arrow indicating the direction of a transformation.
# def draw_transform(debug, trans, col=carla.Color(255, 0, 0), lt=-1):
#     debug.draw_arrow(
#     trans.location, trans.location + trans.get_forward_vector(),  # Draws an arrow from the transformation's location in the direction it's facing.
#     thickness=0.05, arrow_size=0.1, color=col, life_time=lt)  # Sets the thickness, arrow size, color, and lifetime of the arrow.

# # Function to draw a line between two waypoints (locations) with an optional point at the end.
# def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=0.5):
#     debug.draw_line(
#     w0 + carla.Location(z=0.25),  # The start point of the line with a slight elevation.
#     w1 + carla.Location(z=0.25),  # The end point of the line with a slight elevation.
#     thickness=0.1, color=color, life_time=lt, persistent_lines=False)  # Sets the thickness, color, and lifetime of the line.
#     debug.draw_point(w1 + carla.Location(z=0.25), 0.105, color, lt, False)  # Draws a point at the end of the line.

def process_image(frame, idx, infer=True):
    """
    Process a single camera frame from CARLA, optionally performing inference.
    :param frame: The CARLA camera frame.
    :param idx: The index of the camera (used for storing in a list).
    :param infer: Whether to run inference on the frame.
    """
    global running, camera_images

    # Convert the frame to an RGB image
    frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (frame.height, frame.width, 4))
    array = array[:, :, :3]  # Extract RGB channels

    # Perform inference if required
    if infer:
        processed_image = show_inference(detection_model, array)
    else:
        processed_image = array

    # Store the processed frame in the global list
    camera_images[idx] = processed_image

    # Check if all images are available (not None)
    if all(img is not None for img in camera_images):
        combined_image = combine_images(camera_images)
        cv2.imshow('Combined Camera Feed', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            
def combine_images(images):
    """
    Combine multiple camera images into a single composite image.

    :param images: A list of images to combine.
    :return: A single combined image.
    """
    h, w, _ = images[0].shape
    combined_image = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)

    # Arrange images in a 3x3 grid
    #Top Row (Front Views)
    combined_image[0:h, 0:w] = images[0]  # Front left cam 1
    combined_image[0:h, w:w*2] = images[1]  # Front center cam 2
    combined_image[0:h, w*2:] = images[2]  # Front right cam 3
    
    # Middle Row (Side and Rear Views)
    combined_image[h:h*2, 0:w] = images[7]  # Rear left cam 5
    combined_image[h:h*2, w:w*2] = images[4]  # Top back cam 8
    combined_image[h:h*2, w*2:] = images[6]  # Rear right cam 9
    
    # Bottom Row (Top Back and Rear Views)
    combined_image[h*2:, 0:w] = images[3]  # Left side cam 7
    combined_image[h*2:, w:w*2] = images[5]  # Right side cam 6
    combined_image[h*2:, w*2:] = images[8]  # Rear center cam 4

    return combined_image

def show_inference(model, image_np):
    """
    Run inference on an image using a YOLO model and visualize the results.

    :param model: The YOLO model.
    :param image_np: The input image as a NumPy array.
    :return: The image with detections drawn on it.
    """
    # Run inference on the image
    results = model(image_np, fuse=True)

    # Extract detections
    detections = results[0]  # Access the first result (batch size is 1)
    
    # Extract bounding boxes, scores, and class IDs
    boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = detections.boxes.conf.cpu().numpy()  # Confidence scores
    classes = detections.boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Visualize the detections
    image_np_with_detections = image_np.copy()
    for i, box in enumerate(boxes):
        # Draw bounding box
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 1
        image_np_with_detections = cv2.rectangle(image_np_with_detections, start_point, end_point, color, thickness)

        # Draw label
        label = f"{category_index[classes[i]]['name']}:{scores[i]:.2f}"
        position = (start_point[0], start_point[1] - 10)
        cv2.putText(image_np_with_detections, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image_np_with_detections

# def run_inference_for_single_frame(frame, model):
#     """
#     Run inference on a frame using a YOLOv8 model.

#     Args:
#         frame (numpy array): Input frame.
#         model (torch.nn.Module): YOLOv8 model.

#     Returns:
#         dict: Dictionary containing detection results.
#     """
#     # Convert frame to RGB
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Get frame dimensions
#     height, width, _ = frame.shape

#     # Convert frame to tensor
#     frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0

#     # Run inference
#     outputs = model(frame_tensor)

#     # Extract detection results
#     detection_results = []
#     for output in outputs:
#         for detection in output:
#             scores = detection['scores']
#             class_id = torch.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Confidence threshold
#                 x, y, w, h = detection['bbox']
#                 x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
#                 detection_results.append({
#                     'class_id': class_id.item(),
#                     'confidence': confidence.item(),
#                     'bbox': [x1, y1, x2, y2]
#                 })

#     # Return detection results
#     return {
#         'detection_results': detection_results,
#         'frame_height': height,
#         'frame_width': width
#     }
    
# def visualize_boxes_and_labels_on_frame_array(frame, boxes, classes, scores, category_index,
#                                               instance_masks=None, use_normalized_coordinates=False,
#                                               line_thickness=1):
#     if frame is None:
#         print("Error: Image is None in visualize_boxes_and_labels_on_image_array.")
#         return None

#     for i in range(boxes.shape[0]):
#         if scores[i] > 0.1: #change the score high model prediction
#             box = tuple(boxes[i].tolist())
#             class_id = classes[i]
#             if class_id not in category_index:
#                 continue
#             class_name = category_index[class_id]['name']
#             display_str = f'{class_name}: {int(100 * scores[i])}%'
#             frame = draw_bounding_box_on_frame_array(frame, box[0], box[1], box[2], box[3], display_str, class_id)
#     return frame

# # Define a color mapping for each class ID
# class_color_map = {
#     1: (255, 0, 0),  # car - Red
#     2: (0, 255, 0),  # pedestrian - Green
#     3: (0, 0, 255),  # trafficLight-GreenLeft - Blue
#     4: (0, 255, 255),  # trafficLight-Green - Yellow
#     5: (255, 0, 255),  # trafficLight-Red - Magenta
#     6: (255, 165, 0),  # trafficLight-RedLeft - Orange
#     7: (0, 0, 0),  # trafficLight - Black
#     8: (128, 0, 128),  # truck - Purple
#     9: (255, 192, 203),  # biker - Pink
#     10: (255, 255, 0),  # trafficLight-Yellow - Cyan
#     11: (128, 128, 0)   # trafficLight-YellowLeft - Olive
# }
# def draw_bounding_box_on_frame_array(frame, ymin, xmin, ymax, xmax, display_str,
#                                      class_id, thickness=1):
#     if frame is None:
#         print("Error: Image is None in draw_bounding_box_on_image_array.")
#         return None

#     # Make the array writable
#     frame = np.ascontiguousarray(frame)

#     # Get the color based on the class_id
#     color = class_color_map.get(class_id, (255, 255, 255))  # Default to white if class_id not found

#     (im_width, im_height) = frame.shape[1], frame.shape[0]
#     (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
#                                   ymin * im_height, ymax * im_height)
#     cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, thickness)
#     cv2.putText(frame, display_str, (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#     return frame

# def cleanup():
#     global actor_list
#     for actor in actor_list:
#         actor.destroy()
#     print("All actors cleaned up!")
#     cv2.destroyAllWindows()

def main():
    global actor_list, detection_model, category_index, running

    client = carla.Client('localhost', 2000)
    client.set_timeout(30)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    debug = world.debug  # Retrieves the debugging helper from the world.

    # Get vehicle blueprint and spawn point
    vehicle_bp = blueprint_library.filter('model3')[0] # Change the car model if needed, like cybertruck
    print(vehicle_bp)
    # start_points = get_specific_spawn_point(world)
    # print(start_points)
    start_points = random.choice(world.get_map().get_spawn_points())
    
    # # Get the default final destination
    # destination = get_final_destination()
    # print(destination)
    
    vehicle = world.spawn_actor(vehicle_bp, start_points)
    print(vehicle)
    
    actor_list.append(vehicle)
    
    # Set autopilot
    vehicle.set_autopilot(True)
    
    # Camera attributes
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f"{IM_WIDTH}")
    camera_bp.set_attribute('image_size_y', f"{IM_HEIGHT}")
    camera_bp.set_attribute('fov', '110')

    # # This section is used to Define camera locations and orientations relative to the cybertruck vehicle
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
        
    # if Show_path_trajectory:  # If the flag to show the path trajectory is set to True.
    #     current_ = vehicle.get_location()  # Gets the current location of the vehicle.
    #     while True:
    #         next_ = vehicle.get_location()  # Continuously gets the updated location of the vehicle.
    #         # vector = vehicle.get_velocity()

    #         draw_waypoint_union(debug, current_, next_, blue, 30)  # Draws the path between the current and next location.
    #         debug.draw_string(current_, str('%15.0f' % (math.sqrt((next_.x - current_.x)**2 + (next_.y - current_.y)**2 + (next_.z - current_.z)**2))), False, red, 30)  # Draws the distance between the two points.

    #         current_ = next_  # Updates the current location to the next location.
    #         time.sleep(1)  # Waits for 1 second before the next iteration.
    # running = True
    
    try:
        while running:
            
            # Draw the planned route on the map
            # draw_route(world, vehicle.get_location(), destination)
            
            # # Check if the vehicle has reached the final destination
            # if has_reached_destination(vehicle, destination):
            #     print("Final destination reached!")
            #     # Stop the vehicle
            #     vehicle.set_autopilot(False)
            #     running = False
            #     break

            time.sleep(0.01)

    except KeyboardInterrupt:
        running = False

    finally:
        for actor in actor_list:
            actor.destroy()
        print("All actors cleaned up!")
            

if __name__ == '__main__':
    main()
    
    