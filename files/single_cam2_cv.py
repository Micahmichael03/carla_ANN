#===============================================================================================
# single_cam-FILE- for-RGB-Detection-from-simulation-by-Michael-Micah did not work, its hanging
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
import math
 
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

def get_specific_spawn_point(world, x=110, y=8.2, z=0.5, yaw=0.0, pitch=0.0, roll=0.0):
    return carla.Transform(location=carla.Location(x=x, y=y, z=z),
                        rotation=carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))

def get_final_destination(x=200, y=10, z=0.5):
    return carla.Location(x=x, y=y, z=z)

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

    # Write the image to the video file
    if video_writer is not None:
        # try:
            video_writer.write(array)
        # except cv2.error as e:
        #     print(f"Error writing frame to video: {e}")
        
    # Display the processed image
    cv2.imshow('Front Camera', array)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False  # Stop the loop if 'q' is pressed
    return True
print("Carla processing started...")

def cleanup():
    # global actor_list, video_writer
    # for actor in actor_list:
    #     actor.destroy()
    if video_writer:
        video_writer.release()
    print("All actors cleaned up!")
    cv2.destroyAllWindows()

def main():
    global video_writer, running

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

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        # start_points = random.choice(world.get_map().get_spawn_points()) #uncomments this line for random spawn point
        start_points = get_specific_spawn_point(world)
        print(start_points)
        vehicle = world.spawn_actor(bp, start_points)
        actor_list = [vehicle]
        
        vehicle.set_autopilot(True)
        
        # Get the default final destination
        # destination = get_final_destination()
        # draw_route(world, vehicle.get_location(), destination)
        # print(destination)
        
        

        rgb_cam_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        rgb_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        rgb_cam_bp.set_attribute("fov", "90")
        
        camera_sensor = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0))
        rgb_camera = world.spawn_actor(rgb_cam_bp, camera_sensor, attach_to=vehicle)
        actor_list.append(rgb_camera)

        initialize_video_writer(video_save_path, IM_WIDTH, IM_HEIGHT)

        rgb_camera.listen(lambda image: process_rgb_img(image))

        while True:
            # # Check if the vehicle has reached the final destination
            # if has_reached_destination(vehicle, destination):
                
            #     # Stop the vehicle
            #     vehicle.set_autopilot(False)
            #     print("Final destination reached!")
            #     running = False
            #     break
            # else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        cleanup(actor_list)

if __name__ == "__main__":
    main()
