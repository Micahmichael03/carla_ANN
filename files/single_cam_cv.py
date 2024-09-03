#===============================================================================================
# single_cam-FILE- for-RGB-Detection-from-simulation-by-Michael-Micah
#===============================================================================================
from ultralytics import YOLO
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import logging
import argparse
import torch

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 640
IM_HEIGHT = 480
 
SAVE_PATH = "c:/mydataset"
video_save_path = os.path.join(SAVE_PATH, "front_camera_video.mp4")
video_writer = None

CONFIDENCE_THRESHOLD = 0.1  # Set your desired confidence score threshold here
BOUNDING_BOX_THICKNESS = 1  # Set your desired bounding box thickness here

def initialize_video_writer(path, width, height, fps=20.0, codec='mp4v'):
    global video_writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

def write_video_frames(frames, path, fps, frame_width, frame_height, codec='mp4v'):
    # Initialize the video writer
    initialize_video_writer(path, frame_width, frame_height, fps, codec)
    
    # Write each frame to the video
    for f in frames:
        video_writer.write(f)
    
    # Release the video writer once done
    video_writer.release()
    print(f"Video saved to {path}")

# Load YOLOv8 model
model = YOLO(r"C:\CARLA_0.9.5\PythonAPI\examples\models\train2\weights\best.pt")
print("Model successfully loaded")

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

def draw_bounding_boxes(image, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = int(detection.cls[0].cpu().numpy())

        # Apply confidence score threshold
        if conf >= CONFIDENCE_THRESHOLD:
            color = colors.get(cls, (255, 255, 255))
            class_name = custom_classes.get(cls, model.names[cls])
            label = f"{class_name}: {conf:.2f}"
            # Draw the bounding box with specified thickness
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, BOUNDING_BOX_THICKNESS)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_rgb_img(image):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))
    array = array[:, :, :3].copy()  # Ensure it's writable

    # Inference using YOLOv8
    results = model(array, verbose=False)

    # Ensure results is a list of detections
    detections = results[0].boxes if results and results[0].boxes is not None else []

    # Draw bounding boxes on the image if detections exist
    if detections:
        draw_bounding_boxes(array, detections)

    # Display the processed image
    cv2.imshow('Front Camera', array)

    # Write the image to the video file
    if video_writer is not None:
        try:
            video_writer.write(array)
        except cv2.error as e:
            print(f"Error writing frame to video: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False  # Stop the loop if 'q' is pressed
    return True

def cleanup(actor_list):
    for actor in actor_list:
        actor.destroy()
    if video_writer:
        video_writer.release()
    print("All actors cleaned up!")
    cv2.destroyAllWindows()

def main():
    global video_writer

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

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        bp = blueprint_library.filter("model3")[0]

        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)
        actor_list = [vehicle]

        rgb_cam_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        rgb_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        rgb_cam_bp.set_attribute("fov", "90")
        
        transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0))
        rgb_camera = world.spawn_actor(rgb_cam_bp, transform, attach_to=vehicle)
        actor_list.append(rgb_camera)

        initialize_video_writer(video_save_path, IM_WIDTH, IM_HEIGHT)

        rgb_camera.listen(lambda image: process_rgb_img(image))

        while True:
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        cleanup(actor_list)

if __name__ == "__main__":
    main()
