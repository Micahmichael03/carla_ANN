#===============================================================================================
# Test1-FILE- for-RGB-Detection-from-simulation-by-Michael-Micah with improvements
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
import signal
 
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 720
IM_HEIGHT = 580

SAVE_PATH = "c:/mydataset"
video_save_path = os.path.join(SAVE_PATH, "front_camera_video.mp4")
video_writer = None

CONFIDENCE_THRESHOLD = 0.1
BOUNDING_BOX_THICKNESS = 1

def initialize_video_writer(path, width, height, fps=10.0, codec='mp4v'):
    global video_writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

def write_video_frames(frames, path, fps, frame_width, frame_height, codec='mp4v'):
    initialize_video_writer(path, frame_width, frame_height, fps, codec)
    for f in frames:
        video_writer.write(f)
    video_writer.release()
    print(f"Video saved to {path}")

model = YOLO(r"C:\CARLA_0.9.5\PythonAPI\examples\models\train2\weights\best.pt")
print("Model successfully loaded")

def get_specific_spawn_point(world, x=110, y=8.2, z=0.5, yaw=0.0, pitch=0.0, roll=0.0):
    return carla.Transform(location=carla.Location(x=x, y=y, z=z),
                           rotation=carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))

def spawn_stationary_vehicle(world, blueprint_library, transform):
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
    stationary_vehicle = world.spawn_actor(vehicle_bp, transform)
    return stationary_vehicle

custom_classes = {
    0: '30', 1: '60', 2: '90', 3: 'bike', 4: 'bus', 5: 'car', 
    6: 'green_light', 7: 'pedestrian',
    8: 'red_light', 9: 'truck',
}

colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 0),
    7: (0, 128, 0),
    8: (0, 0, 128),
    9: (128, 128, 0),
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
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, BOUNDING_BOX_THICKNESS)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def process_rgb_img(image, model):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))
    array = array[:, :, :3].copy()

    results = model(array, verbose=False)
    detections = results[0].boxes if results and results[0].boxes is not None else []

    if detections:
        draw_bounding_boxes(array, detections)

    if video_writer is not None:
        try:
            video_writer.write(array)
        except cv2.error as e:
            print(f"Error writing frame to video: {e}")
        
    cv2.imshow('Front Camera', array)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return detections  # Return detections for decision-making

# Define a function to control the vehicle with decision-making
def car_control(vehicle, model, camera_sensor):
    current_throttle = 0.52
    current_steer = 0.0  # Initialize steering angle
    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, gear=0))  # Set initial throttle and gear
    
    previous_positions = {}  # Initialize outside the loop to persist across frames

    def process_frame(image):
        nonlocal previous_positions, current_throttle, current_steer  # Ensure access to these variables
        
        # Process the image to get detections
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))
        array = array[:, :, :3].copy()

        results = model(array, verbose=False)
        detections = results[0].boxes if results and results[0].boxes is not None else []

        # Define the ROI for the front view (e.g., middle of the image)
        roi_x_min = int(IM_WIDTH * 0.3)  # 30% from the left
        roi_x_max = int(IM_WIDTH * 0.7)  # 70% from the left
        roi_y_min = int(IM_HEIGHT * 0.5)  # 50% from the top
        roi_y_max = IM_HEIGHT  # bottom of the image

        # Variables to track dynamic objects
        dynamic_objects_in_roi = False
        
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            cls = int(detection.cls[0].cpu().numpy())
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_position = (center_x, center_y)
            
            # Calculate speed of the object (simplified, assume it's proportional to the change in position)
            object_speed = 0
            if cls in previous_positions:
                prev_x, prev_y = previous_positions[cls]
                object_speed = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5

            # Update the previous position
            previous_positions[cls] = current_position

            if x1 >= roi_x_min and x2 <= roi_x_max and y1 >= roi_y_min and y2 <= roi_y_max:
                if cls in [3, 4, 5, 7, 9]:  # Bike, Bus, Car, Pedestrian, Truck
                    dynamic_objects_in_roi = True
                    print(f"Dynamic object ({cls}) detected in front, adjusting speed and steer.")

                    # Example steering logic: steer away from the object
                    if center_x < (IM_WIDTH / 2):  # Object is on the left side
                        current_steer = min(current_steer + 0.1, 1.0)  # Steer right
                    else:  # Object is on the right side
                        current_steer = max(current_steer - 0.1, -1.0)  # Steer left
                    
                    # Adjust throttle based on object speed
                    if object_speed < 2:
                        current_throttle = max(0.2, current_throttle - 0.15)
                    else:
                        current_throttle = min(0.6, current_throttle + 0.1)
                    
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))
                else:
                    current_steer = 0.0  # Reset steering if no obstacles are detected

                # Speed limit logic
                if cls == 0:  # Speed limit 30
                    if dynamic_objects_in_roi:
                        print("Speed limit 30 detected but dynamic objects present, reducing speed.")
                        current_throttle = max(0.2, current_throttle - 0.15)  # Reduce speed safely
                    else:
                        print("Speed limit 30 detected, adjusting speed.")
                        current_throttle = 0.3  # Adjust to match the speed limit
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

                elif cls == 1:  # Speed limit 60
                    if dynamic_objects_in_roi:
                        print("Speed limit 60 detected but dynamic objects present, reducing speed.")
                        current_throttle = max(0.4, current_throttle - 0.15)  # Adjust to a safer speed, but not too low
                    else:
                        print("Speed limit 60 detected, adjusting speed.")
                        current_throttle = 0.6  # Adjust to match the speed limit
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

                elif cls == 2:  # Speed limit 90
                    if dynamic_objects_in_roi:
                        print("Speed limit 90 detected but dynamic objects present, reducing speed.")
                        current_throttle = max(0.5, current_throttle - 0.15)  # Adjust to a safer speed, but maintain a higher speed
                    else:
                        print("Speed limit 90 detected, adjusting speed.")
                        current_throttle = 0.9  # Adjust to match the speed limit
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

                # Example dynamic object speed logic
                elif cls == 3:  # Bike
                    if object_speed > 5:
                        print("Fast-moving bike detected, maintaining or increasing speed.")
                        current_throttle = min(0.6, current_throttle + 0.1)
                    elif object_speed < 2:
                        print("Slow-moving or stationary bike detected, reducing speed.")
                        current_throttle = max(0.2, current_throttle - 0.2)
                    else:
                        print("Moderate speed bike detected, adjusting speed.")
                        current_throttle = max(0.3, current_throttle - 0.1)
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

                elif cls == 4:  # Bus
                    if object_speed > 3:
                        print("Fast-moving bus detected, maintaining or increasing speed.")
                        current_throttle = min(0.6, current_throttle + 0.1)
                    elif object_speed < 1:
                        print("Slow-moving or stationary bus detected, reducing speed.")
                        current_throttle = max(0.25, current_throttle - 0.15)
                    else:
                        print("Moderate speed bus detected, adjusting speed.")
                        current_throttle = max(0.3, current_throttle - 0.1)
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

                elif cls == 5:  # Car
                    if object_speed > 4:
                        print("Fast-moving car detected, maintaining or increasing speed.")
                        current_throttle = min(0.7, current_throttle + 0.1)
                    elif object_speed < 2:
                        print("Slow-moving or stationary car detected, reducing speed.")
                        current_throttle = max(0.2, current_throttle - 0.15)
                    else:
                        print("Moderate speed car detected, adjusting speed.")
                        current_throttle = max(0.3, current_throttle - 0.1)
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

                elif cls == 9:  # Truck
                    if object_speed > 2:
                        print("Fast-moving truck detected, maintaining or increasing speed.")
                        current_throttle = min(0.6, current_throttle + 0.1)
                    elif object_speed < 1:
                        print("Slow-moving or stationary truck detected, reducing speed.")
                        current_throttle = max(0.25, current_throttle - 0.15)
                    else:
                        print("Moderate speed truck detected, adjusting speed.")
                        current_throttle = max(0.3, current_throttle - 0.1)
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))
                    
                elif cls == 7:  # Pedestrian
                    print("Pedestrian detected in front, stopping vehicle.")
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    time.sleep(3)
                    current_throttle = 0.35
                    current_steer = 0.0  # Reset steer after stopping
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, brake=0.0))

                elif cls == 6:  # Green light
                    print("Green light detected in front, increasing speed.")
                    current_throttle = min(0.8, current_throttle + 0.15)
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

                elif cls == 8:  # Red light
                    print("Red light detected in front, stopping vehicle.")
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    time.sleep(5)
                    current_throttle = 0.35
                    current_steer = 0.0  # Reset steer after stopping
                    vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, brake=0.0))

        # If no detections, continue normal driving
        vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

    # Listen for camera images and process frames
    camera_sensor.listen(lambda image: process_frame(image))
    
def cleanup(actor_list):
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    if video_writer:
        video_writer.release()
    print("All actors cleaned up!")
    cv2.destroyAllWindows()

def signal_handler(sig, frame):
    print('Exiting gracefully')
    cleanup(actor_list)
    sys.exit(0)

def main():
    global video_writer, actor_list

    signal.signal(signal.SIGINT, signal_handler)

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

    actor_list = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        bp = blueprint_library.filter("model3")[0]

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        start_points = get_specific_spawn_point(world)
        vehicle = world.spawn_actor(bp, start_points)
        actor_list.append(vehicle)
        
        vehicle.set_autopilot(False) # disable autopilot or set at True for autonomous driving
    
        rgb_cam_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        rgb_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        rgb_cam_bp.set_attribute("fov", "90")
        
        camera_sensor = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0))
        rgb_camera = world.spawn_actor(rgb_cam_bp, camera_sensor, attach_to=vehicle)
        actor_list.append(rgb_camera)

        initialize_video_writer(video_save_path, IM_WIDTH, IM_HEIGHT)

        rgb_camera.listen(lambda image: process_rgb_img(image, model))
        car_control(vehicle, model, rgb_camera)
        
        stationary_transform = get_specific_spawn_point(world, x=195, y=8.2, z=0.5)
        stationary_vehicle = spawn_stationary_vehicle(world, blueprint_library, stationary_transform)
        actor_list.append(stationary_vehicle)

        while True:
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        cleanup(actor_list)

if __name__ == "__main__":
    main()
