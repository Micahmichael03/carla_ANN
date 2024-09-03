#=============================================================================================================
#RGB_img-FILE-for-RGB-Detection-from-simulation-by-Michael_Micah,ENGR.DR.MARTINS_OBASEKI (2024)-----
#=============================================================================================================

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
from datetime import datetime
import logging
import argparse
 
#===============================================================================================
#RGB_img-FILE- for-RGB-from-simulation-by-Michael-Micah----------------------------------------
#===============================================================================================

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

#===============================================================================================
#RGB_img-FILE- for-RGB-from-simulation-by-Michael-Micah----------------------------------------
#===============================================================================================

# Adjustable image dimensions
IM_WIDTH = 255
IM_HEIGHT = 255

# Path to save images (modify as needed)
SAVE_PATH = "c:\mydataset"

# Define camera transformations for eight cameras
camera_transformations =  [
    carla.Transform(carla.Location(x=2.0, y=-0.5, z=1.4), carla.Rotation(yaw=-45)),  # Front left cam 1
    carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0)),       # Front center cam 0
    carla.Transform(carla.Location(x=2.0, y=0.5, z=1.4), carla.Rotation(yaw=45)),   # Front right cam 2
    carla.Transform(carla.Location(x=1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-90)),  # Left side cam 7
    carla.Transform(carla.Location(x=-6.0, z=5.0), carla.Rotation(pitch=-30)),   # Top back cam 8
    carla.Transform(carla.Location(x=1.0, y=0.75, z=1.4), carla.Rotation(yaw=90)),  # Right B-pillar cam 6
    carla.Transform(carla.Location(x=-1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-135)),  # Rear right cam 5
    carla.Transform(carla.Location(x=-1.0, z=1.4), carla.Rotation(yaw=180)),   # Rear center cam 3
    carla.Transform(carla.Location(x=-1.0, y=0.75, z=1.4), carla.Rotation(yaw=135)) # Rear left cam 4
]
 
# Create directories for each camera
camera_directories = [os.path.join(SAVE_PATH, f"camera_{idx}") for idx in range(len(camera_transformations))]
for directory in camera_directories:
    os.makedirs(directory, exist_ok=True)
    
# Create a directory for saving combined images
combined_image_dir = os.path.join(SAVE_PATH, "combined_images")
os.makedirs(combined_image_dir, exist_ok=True)

video_save_path = os.path.join(SAVE_PATH, "combined_video.mp4")
os.makedirs(video_save_path, exist_ok=True)
video_writer = True

def initialize_video_writer(width, height):
    global video_writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_save_path, fourcc, 20.0, (width, height))


def save_image(image, image_type, timestamp, idx):
    image_path = os.path.join(camera_directories[idx], f"{image_type}_{timestamp}.jpeg")

    try:
        cv2.imwrite(image_path, image)
        print(f"Saved {image_type} image: {image_path}")
    except Exception as e:
        print(f"Error saving {image_type} image {image_path}: {str(e)}")

def process_rgb_img(image, timestamp, idx):
    global camera_images
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))
    array = array[:, :, :3]  # Extract RGB channels

    # Store the processed image in the global list
    camera_images[idx] = array

    # Save the individual image
    save_image(array, "rgb", timestamp, idx)

def combine_images(images):
    # Ensure all images are the same size
    h, w, _ = images[0].shape

    # Calculate grid dimensions based on the number of images (3x3 grid for 9 cameras)
    grid_h = 3
    grid_w = 3

    combined_image = np.zeros((grid_h * h, grid_w * w, 3), dtype=np.uint8)

    for i in range(grid_h):
        for j in range(grid_w):
            img_idx = i * grid_w + j
            if img_idx < len(images):
                combined_image[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[img_idx]

    return combined_image

def cleanup():
    global actor_list, video_writer
    for actor in actor_list:
        actor.destroy()
    if video_writer:
        video_writer.release()
    print("All actors cleaned up!")
    cv2.destroyAllWindows()

#================================================================================================
#RGB_img-FILE- for-RGB-from-simulation-by-Michael-Micah------------------------------------------
#================================================================================================

def main():
    
    #=============================================================================================
    #RGB_img-FILE- for-RGB-from-simulation-by-Michael-Micah---------------------------------------
    #=============================================================================================

    global camera_images, running, actor_list, video_writer
    camera_images = [None] * len(camera_transformations)
    running = True
    actor_list = []
    
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

    #===============================================================================================
    #TEST3-FILE- for-RGB-and-Segmentation-from-simulation-by-Michael-Micah--------------------------
    #===============================================================================================

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Choose a suitable model
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
        
        # Initialize an array to store images from all cameras
        camera_images = [np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8) for _ in range(len(camera_transformations))]


        # Spawn and attach cameras
        for idx, transform in enumerate(camera_transformations):
            rgb_camera = world.spawn_actor(rgb_cam_bp, transform, attach_to=vehicle)
            actor_list.append(rgb_camera)

            rgb_camera.listen(lambda image, idx=idx: process_rgb_img(image, datetime.now().strftime('%Y%m%d%H%M%S%f'), idx))

        # Initialize video writer
        initialize_video_writer(IM_WIDTH * 3, IM_HEIGHT * 3)
        
        #===============================================================================================
        #RGB_img-FILE- for-RGB-from-simulation-by-Michael-Micah----------------------------------------
        #===============================================================================================

            
        # Display the combined view
        while running:
            if all(img is not None for img in camera_images):
                combined_image = combine_images(camera_images)

                # Save the combined image
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
                combined_image_path = os.path.join(combined_image_dir, f"combined_{timestamp}.jpeg")
                cv2.imwrite(combined_image_path, combined_image)
                print(f"Saved combined image: {combined_image_path}")
                
                # Write the combined image to video
                video_writer.write(combined_image)

                # Display the combined image
                cv2.imshow('Combined Camera Feed', combined_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False

            time.sleep(0.1)
    finally:
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

if __name__ == '__main__':
    main()
