#=============================================================================================================
# Recording-FILE-for-RGB-Detection-from-simulation-by-Michael_Micah,ENGR.DR.MARTINS_OBASEKI (2024)-----------
#=============================================================================================================

""" 

"""

# Importing necessary libraries for object detection, simulation, video recording, and display.
from ultralytics import YOLO
import glob
import os
import sys
import random
import time
import numpy as np
import pygame
import logging
import argparse
import cv2  # OpenCV added for video recording capabilities

#===============================
# Calling Carla simulator-------
#===============================

try:
    # Adding Carla's Python API to the system path dynamically based on the system's Python version and OS.
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla  # Importing the Carla simulator API for vehicle and environment simulation.

#================================
# Initialize-global-variables----
#================================

# Defining image dimensions for the RGB camera.
IM_WIDTH = 640
IM_HEIGHT = 480

# Defining the path for saving the video recorded from the simulation.
SAVE_PATH = "c:/mydataset"
video_save_path = os.path.join(SAVE_PATH, "front_camera_video.mp4")

# Setting thresholds for object detection confidence and bounding box thickness for visualizations.
CONFIDENCE_THRESHOLD = 0.3
BOUNDING_BOX_THICKNESS = 1

# Function to initialize the video writer for recording the simulation.
def initialize_video_writer(path, width, height, fps=20.0, codec='mp4v'):
    fourcc = cv2.VideoWriter_fourcc(*codec)  # Define the codec for the video writer.
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))  # Create a video writer object.
    return video_writer

# Load a pre-trained YOLOv8 model for object detection in the simulated environment.
model = YOLO(r"C:\CARLA_0.9.5\PythonAPI\examples\models\train2\weights\best.pt")
print("Model successfully loaded")

# Mapping custom class IDs to labels for detected objects (e.g., speed limits, vehicles, traffic lights).
custom_classes = {
    0: '30',           # Speed limit 30
    1: '60',           # Speed limit 60
    2: '90',           # Speed limit 90
    3: 'bike',         # Bicycle
    4: 'bus',          # Bus
    5: 'car',          # Car
    6: 'green_light',  # Green traffic light
    7: 'pedestrian',   # Pedestrian
    8: 'red_light',    # Red traffic light
    9: 'truck',        # Truck
}

# Defining colors corresponding to each detected class for bounding box visualizations.
colors = {
    0: (255, 0, 0),    # Red for '30'
    1: (0, 255, 0),    # Green for '60'
    2: (0, 0, 255),    # Blue for '90'
    3: (255, 255, 0),  # Cyan for 'bike'
    4: (255, 0, 255),  # Magenta for 'bus'
    5: (0, 255, 255),  # Yellow for 'car'
    6: (128, 0, 0),    # Dark Red for 'green_light'
    7: (0, 128, 0),    # Dark Green for 'pedestrian'
    8: (0, 0, 128),    # Dark Blue for 'red_light'
    9: (128, 128, 0),  # Olive for 'truck'
}

#===============================
# Draw-bounding-boxes-----------
#===============================

# Function to draw bounding boxes on detected objects within the image using Pygame.
def draw_bounding_boxes(image, detections):
    for detection in detections:
        # Extracting bounding box coordinates, confidence score, and class ID from the detection results.
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = int(detection.cls[0].cpu().numpy())

        # Applying the confidence threshold to filter out low-confidence detections.
        if conf >= CONFIDENCE_THRESHOLD:
            # Choosing the color and label for the bounding box based on the detected class.
            color = colors.get(cls, (255, 255, 255))
            class_name = custom_classes.get(cls, model.names[cls])
            label = f"{class_name}: {conf:.2f}"
            # Drawing the bounding box and the label on the image.
            pygame.draw.rect(image, color, pygame.Rect(int(x1), int(y1), int(x2-x1), int(y2-y1)), BOUNDING_BOX_THICKNESS)
            font = pygame.font.SysFont(None, 24)
            text_surface = font.render(label, True, color)
            image.blit(text_surface, (int(x1), int(y1) - 20))

#===============================
# Process-rgb-image-------------
#===============================

# Function to process the raw RGB image obtained from the Carla simulator and record it.
def process_rgb_img(image, video_writer):
    # Convert image format from Carla's default to raw format for processing.
    image.convert(carla.ColorConverter.Raw)
    # Convert the raw image data to a numpy array.
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    # Reshape the array into the desired image dimensions (IM_HEIGHT x IM_WIDTH x 4 channels).
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))
    array = array[:, :, :3].copy()  # Ensure it's writable and drop the alpha channel.
    
    # Convert the image from BGR to RGB by swapping the first and third channels.
    array = array[:, :, ::-1]  # Converts BGR to RGB

    # Perform inference on the image using the loaded YOLO model.
    results = model(array, verbose=False)

    # Extract detections (bounding boxes) from the results.
    detections = results[0].boxes if results and results[0].boxes is not None else []

    # Convert the numpy array back to a Pygame surface for display.
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    # If any detections exist, draw bounding boxes on the image.
    if detections:
        draw_bounding_boxes(surface, detections)

    # Display the processed image using Pygame.
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # Write the processed image to the video file.
    if video_writer is not None:
        frame_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)  # Convert back to BGR for recording.
        video_writer.write(frame_bgr)  # Record the frame to the video file.

# Function to clean up actors in the Carla simulator and release resources.
def cleanup(actor_list, video_writer):
    for actor in actor_list:
        actor.destroy()  # Destroy each actor (e.g., vehicle, sensor) to free resources.
    if video_writer:
        video_writer.release()  # Release the video writer if it was used.
    print("All actors cleaned up!")
    pygame.quit()  # Quit the Pygame display.

#===============================
# Main--------------------------
#===============================

def main():
    
    #===============================
    # global variables--------------
    #===============================
    
    global screen, video_writer  # Define global variables for Pygame display and video writer.

    # Setting up argument parsing to allow for customizable runtime options (e.g., host, port, resolution).
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    args = argparser.parse_args()

    # Parsing the resolution argument and splitting it into width and height.
    args.width, args.height = [int(x) for x in args.res.split('x')]

    # Setting up the logging level based on the verbosity flag.
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # Initializing Pygame and setting up the display window with the defined resolution.
    pygame.init()
    screen = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT))
    pygame.display.set_caption("CARLA Front Camera")

    # Initialize the video writer for recording the simulation.
    video_writer = initialize_video_writer(video_save_path, IM_WIDTH, IM_HEIGHT)
    
    #===============================
    # Initialize-CARLA-in-a-loop----
    #===============================

    try:
        # Connecting to the Carla simulator server.
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()  # Accessing the simulation world.
        blueprint_library = world.get_blueprint_library()  # Retrieving the blueprint library for actors.

        # Selecting a vehicle blueprint (e.g., Tesla Model 3) for spawning in the simulation.
        bp = blueprint_library.filter("model3")[0]

        # Choosing a random spawn point in the simulation map and spawning the vehicle.
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)  # Enabling autopilot mode for the vehicle.
        actor_list = [vehicle]  # Keeping track of spawned actors for cleanup.

        # Setting up an RGB camera sensor attached to the vehicle.
        rgb_cam_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        rgb_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        rgb_cam_bp.set_attribute("fov", "90")
        
        # Positioning the camera at the front of the vehicle.
        transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0))
        rgb_camera = world.spawn_actor(rgb_cam_bp, transform, attach_to=vehicle)
        actor_list.append(rgb_camera)  # Adding the camera to the list of actors.

        # Setting up the camera to call the process_rgb_img function for each captured frame.
        rgb_camera.listen(lambda image: process_rgb_img(image, video_writer))

        running = True
        while running:
            # Event loop to handle quitting the application via Pygame's event system.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            time.sleep(0.05)  # Introducing a small delay between frames.

    except KeyboardInterrupt:
        pass
    
    #===============================
    # Cleanup-----------------------
    #===============================

    finally:
        cleanup(actor_list, video_writer)  # Cleaning up all actors and resources when exiting.

#===============================
# Start Main Process------------
#===============================

if __name__ == "__main__":
    main()  # Entry point for the script execution.
