#===============================================================================================
# three_cam-FILE- for-RGB-Detection-from-simulation-by-Michael-Micah
#===============================================================================================
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
import cv2
 
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 350
IM_HEIGHT = 350
COMBINED_WIDTH = IM_WIDTH * 3  # Combined width for 3 cameras side by side
COMBINED_HEIGHT = IM_HEIGHT

SAVE_PATH = "c:/mydataset"
video_save_path = os.path.join(SAVE_PATH, "front_cameras_video.mp4")

CONFIDENCE_THRESHOLD = 0.3  # Set your desired confidence score threshold here
BOUNDING_BOX_THICKNESS = 1  # Set your desired bounding box thickness here

# Initialize the video writer
def initialize_video_writer(path, width, height, fps=20.0, codec='mp4v'):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    return video_writer

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

        if conf >= CONFIDENCE_THRESHOLD:
            color = colors.get(cls, (255, 255, 255))
            class_name = custom_classes.get(cls, model.names[cls])
            label = f"{class_name}: {conf:.2f}"
            pygame.draw.rect(image, color, pygame.Rect(int(x1), int(y1), int(x2-x1), int(y2-y1)), BOUNDING_BOX_THICKNESS)
            font = pygame.font.SysFont(None, 24)
            text_surface = font.render(label, True, color)
            image.blit(text_surface, (int(x1), int(y1) - 20))

def process_rgb_img(image):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))
    array = array[:, :, :3].copy()
    array = array[:, :, ::-1]  # Converts BGR to RGB

    # Inference using YOLOv8
    results = model(array, verbose=False)
    detections = results[0].boxes if results and results[0].boxes is not None else []

    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    if detections:
        draw_bounding_boxes(surface, detections)

    return surface, array  # Return both the surface for pygame and the raw image for recording

def create_combined_view(camera_surfaces):
    combined_surface = pygame.Surface((COMBINED_WIDTH, COMBINED_HEIGHT))

    for i, surface in enumerate(camera_surfaces):
        combined_surface.blit(surface, (i * IM_WIDTH, 0))

    return combined_surface

def cleanup(actor_list, video_writer):
    for actor in actor_list:
        actor.destroy()
    if video_writer:
        video_writer.release()
    print("All actors cleaned up!")
    pygame.quit()

def main():
    global screen

    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1920x1080', help='window resolution (default: 1920x1080)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    pygame.init()
    screen = pygame.display.set_mode((COMBINED_WIDTH, COMBINED_HEIGHT))
    pygame.display.set_caption("CARLA Three-Camera View")

    # Initialize the video writer
    video_writer = initialize_video_writer(video_save_path, COMBINED_WIDTH, COMBINED_HEIGHT)

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

        # Define camera transforms for the three front-facing cameras
        camera_transforms = [
            carla.Transform(carla.Location(x=2.0, y=-0.5, z=1.4), carla.Rotation(yaw=-45)),  # Front left cam
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0)),           # Front center cam
            carla.Transform(carla.Location(x=2.0, y=0.5, z=1.4), carla.Rotation(yaw=45)),    # Front right cam
        ]

        cameras = []
        for transform in camera_transforms:
            rgb_cam_bp = blueprint_library.find("sensor.camera.rgb")
            rgb_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
            rgb_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
            rgb_cam_bp.set_attribute("fov", "90")
            camera = world.spawn_actor(rgb_cam_bp, transform, attach_to=vehicle)
            actor_list.append(camera)
            cameras.append(camera)

        # Create surfaces and arrays for each camera view
        camera_surfaces = [None] * 3
        camera_arrays = [None] * 3

        def process_all_images(index, image):
            surface, array = process_rgb_img(image)
            camera_surfaces[index] = surface
            camera_arrays[index] = array

        for i, camera in enumerate(cameras):
            camera.listen(lambda image, i=i: process_all_images(i, image))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if all(camera_surfaces):
                combined_view = create_combined_view(camera_surfaces)
                screen.blit(combined_view, (0, 0))
                pygame.display.flip()

                # Combine the arrays horizontally for video recording
                combined_array = np.hstack(camera_arrays)
                frame_bgr = cv2.cvtColor(combined_array, cv2.COLOR_RGB2BGR)  # Convert back to BGR for recording
                video_writer.write(frame_bgr)

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        cleanup(actor_list, video_writer)

if __name__ == "__main__":
    main()
