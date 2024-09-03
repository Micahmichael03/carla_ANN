#=============================================================================================================
# RGB_SEG_collecttion-FILE-for-RGB-Detection-from-simulation-by-Michael_Micah,ENGR.DR.MARTINS_OBASEKI (2024)-----
#=============================================================================================================

import glob
import os
import sys
import random
import time
import numpy as np
import cv2  # Saves images to drive
from datetime import datetime  # For timestamp in filenames
import logging
import argparse

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Adjustable image dimensions
IM_WIDTH = 640
IM_HEIGHT = 480

# Path to save images (modify as needed)
SAVE_PATH = "c:\Carla-data" # make sure you put it on the hard drive.

# Directories for different image types
def create_directories(base_path):
    for category in ["train", "val", "test"]:
        for cam in range(8):
            os.makedirs(os.path.join(base_path, category, "rgb", f"camera_{cam}"), exist_ok=True)
            os.makedirs(os.path.join(base_path, category, "segmentation", f"camera_{cam}"), exist_ok=True)

create_directories(SAVE_PATH)

# Directories for different image types
RGB_SAVE_PATH = os.path.join(SAVE_PATH, "rgb")
SEG_SAVE_PATH = os.path.join(SAVE_PATH, "segmentation")

# Create directories if they don't exist
os.makedirs(RGB_SAVE_PATH, exist_ok=True)
os.makedirs(SEG_SAVE_PATH, exist_ok=True)

# Example COLOR_MAPPING (replace with actual class colors)
COLOR_MAPPING = {
    0: [0, 0, 0],        # Background
    1: [70, 70, 70],     # Buildings
    2: [190, 153, 153],  # Fences
    3: [220, 20, 60],    # Pedestrians
    4: [153, 153, 153],  # Poles
    5: [157, 234, 50],   # RoadLines
    6: [128, 64, 128],   # Roads
    7: [244, 35, 232],   # Sidewalks
    8: [107, 142, 35],   # Vegetation
    9: [0, 0, 142],      # Vehicles
    3: [250, 170, 160],  # Other
    10: [220, 220, 0],   # TrafficSigns
    16: [102, 102, 156], # Walls
    11: [0, 255, 0],     # Traffic Light - Green
    12: [255, 0, 0],     # Traffic Light - Red
    13: [255, 255, 0],   # Traffic Light - Yellow
    14: [70, 130, 180],  # Sky
    15: [0, 128, 0],     # Tree
    17:[192, 192, 192],  # Street Light
}

CAMERA_POS_X = 1.4 #(X-axis offset)
CAMERA_POS_Z = 1.3 #(Z-axis offset)

def save_image(image, image_type, timestamp, idx, category):
    """
    Saves an image with a given timestamp and type (RGB or segmentation).

    Args:
        image: The image to save.
        image_type: The type of the image (either 'rgb' or 'segmentation').
        timestamp: The timestamp to use in the filename.
        idx: The index of the camera.
        category: The dataset category (train, val, test).
    """
    image_path = os.path.join(SAVE_PATH, category, image_type, f"camera_{idx}", f"{image_type}_{timestamp}.jpeg")

    try:
        cv2.imwrite(image_path, image)
        print(f"Saved {image_type} image: {image_path}")
    except Exception as e:
        print(f"Error saving {image_type} image {image_path}: {str(e)}")

def process_rgb_img(image, timestamp, idx, category):
    """
    Processes and displays the captured camera image.

    Args:
        image: A CARLA sensor.camera.rgb data object.
        timestamp: The timestamp to use for saving the image.
        idx: The index of the camera.
        category: The dataset category (train, val, test).
    """
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # Reshape with channel awareness
    i3 = i2[:, :, :3]  # Extract RGB channels (BGR order)

    # Display the image with reduced waiting time (consider using threading for further optimization)
    cv2.imshow(f"RGB Camera {idx}", i3)
    cv2.waitKey(1) & 0xFF  # Check for a pressed key (prevents indefinite waiting)

    # Save the RGB image
    save_image(i3, 'rgb', timestamp, idx, category)

    # Return the normalized image (BGR format)
    return i3 / 255.0

def process_segmentation_image(image, timestamp, idx, category):
    """
    Processes and displays the captured segmentation camera image.

    Args:
        image: A CARLA sensor.camera.semantic_segmentation data object.
        timestamp: The timestamp to use for saving the image.
        idx: The index of the camera.
        category: The dataset category (train, val, test).
    """
    # Convert raw data to NumPy array
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # Reshape with channel awareness
    i3 = i2[:, :, 2]  # Extract the semantic segmentation mask (G channel holds the class index)

    # Convert class indices to color mapping for visualization
    segmentation_colormap = np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAPPING.items():
        segmentation_colormap[i3 == class_id] = color

    # Display the segmentation image
    cv2.imshow(f"Segmentation Camera {idx}", segmentation_colormap)
    cv2.waitKey(1) & 0xFF  # Check for a pressed key (prevents indefinite waiting)

    # Save the segmentation image
    save_image(segmentation_colormap, 'segmentation', timestamp, idx, category)

    # Return the processed segmentation image
    return segmentation_colormap

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    actor_list = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Choose a suitable model (consider using a model with segmentation or depth capabilities)
        bp = blueprint_library.filter("model3")[0]

        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)
        actor_list.append(vehicle)

        # Configure RGB camera blueprint
        rgb_cam_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        rgb_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        rgb_cam_bp.set_attribute("fov", "90")

        # Configure Segmentation camera blueprint
        seg_cam_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        seg_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        seg_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        seg_cam_bp.set_attribute("fov", "90")

        # Define camera transformations for eight cameras
        camera_transformations = [
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0)),       # Front center cam 0
            carla.Transform(carla.Location(x=2.0, y=-0.5, z=1.4), carla.Rotation(yaw=-45)),  # Front left cam 1
            carla.Transform(carla.Location(x=2.0, y=0.5, z=1.4), carla.Rotation(yaw=45)),   # Front right cam 2
            carla.Transform(carla.Location(x=-1.0, z=1.4), carla.Rotation(yaw=180)),    # Rear center cam 3
            carla.Transform(carla.Location(x=-1.0, y=0.75, z=1.4), carla.Rotation(yaw=135)),  # Rear left cam 4
            carla.Transform(carla.Location(x=-1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-135)),  # Rear right cam 5
            carla.Transform(carla.Location(x=1.0, y=0.75, z=1.4), carla.Rotation(yaw=90)),  # Right B-pillar cam 6
            carla.Transform(carla.Location(x=1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-90)),  # Left side cam 7
        ]

        # Determine the dataset category (train, val, test) for each image
        def get_category():
            rand = random.random()
            if rand < 0.7:
                return "train"
            elif rand < 0.85:
                return "val"
            else:
                return "test"

        # Spawn and attach cameras
        for idx, transform in enumerate(camera_transformations):
            rgb_camera = world.spawn_actor(rgb_cam_bp, transform, attach_to=vehicle)
            seg_camera = world.spawn_actor(seg_cam_bp, transform, attach_to=vehicle)
            actor_list.append(rgb_camera)
            actor_list.append(seg_camera)

            rgb_camera.listen(lambda image, idx=idx: process_rgb_img(image, datetime.now().strftime('%Y%m%d%H%M%S%f'), idx, get_category()))
            seg_camera.listen(lambda image, idx=idx: process_segmentation_image(image, datetime.now().strftime('%Y%m%d%H%M%S%f'), idx, get_category()))

        time.sleep(30)

    finally:
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

if __name__ == '__main__':
    main()