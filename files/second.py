#============================================================================
# Second.py - Images from Simulation by Michael Micah
#============================================================================

import sys
import math
import glob
import os
import numpy as np
import cv2
import queue
import time

# Attempt to append the CARLA Python API path to the system path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass  # No CARLA egg file found; continue without it

import carla

# Function to calculate the speed of the vehicle in km/h
def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

# Class to manage vehicle control using PID controllers
class VehiclePIDController:
    
    def __init__(self, vehicle, args_Lateral, args_Longitudinal, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """
        Initialize the VehiclePIDController with PID parameters and limits.
        """
        self.max_brake = max_brake
        self.max_steering = max_steering
        self.max_throttle = max_throttle
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.past_steering = self.vehicle.get_control().steer
        self.long_controller = PIDLongitudinalControl(self.vehicle, **args_Longitudinal)
        self.lat_controller = PIDLateralControl(self.vehicle, **args_Lateral)
    
    def run_step(self, target_speed, waypoint):
        """
        Calculate control commands based on PID output and apply them to the vehicle.
        """
        acceleration = self.long_controller.run_step(target_speed)
        current_steering = self.lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        
        # Throttle and brake control
        if acceleration >= 0.0:
            control.throttle = min(abs(acceleration), self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)
            
        # Steering control with limits to avoid sudden changes
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1
        
        if current_steering >= 0:
            steering = min(self.max_steering, current_steering)
        else:
            steering = max(-self.max_steering, current_steering)
            
        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering
        
        return control
        
# Class for longitudinal control using PID
class PIDLongitudinalControl:
    
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Initialize PIDLongitudinalControl with PID parameters.
        """
        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen=10)
    
    def pid_controller(self, target_speed, current_speed):
        """
        Compute PID control signal based on speed error.
        """
        error = target_speed - current_speed
        self.errorBuffer.append(error)
        
        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0
        
        return np.clip(self.K_P * error + self.K_D * de + self.K_I * ie, -1.0, 1.0)
    
    def run_step(self, target_speed):
        """
        Calculate and return the control signal for achieving the target speed.
        """
        current_speed = get_speed(self.vehicle)
        return self.pid_controller(target_speed, current_speed)
    
# Class for lateral control using PID
class PIDLateralControl:
    
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Initialize PIDLateralControl with PID parameters.
        """
        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen=10)
        
    def run_step(self, waypoint):
        """
        Calculate and return the control signal for steering towards the waypoint.
        """
        return self.pid_controller(waypoint, self.vehicle.get_transform())
    
    def pid_controller(self, waypoint, vehicle_transform):
        """
        Compute PID control signal for steering based on waypoint error.
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(
            x=math.cos(math.radians(vehicle_transform.rotation.yaw)), 
            y=math.sin(math.radians(vehicle_transform.rotation.yaw))
        )
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])
        
        # Calculate the angle between vehicle direction and waypoint direction
        dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        cross = np.cross(v_vec, w_vec)
        if cross[2] < 0:
            dot *= -1
            
        self.errorBuffer.append(dot)
        
        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0
            
        return np.clip(self.K_P * dot + self.K_I * ie + self.K_D * de, -1.0, 1.0)

def main():
    """
    Main function to set up the simulation, spawn the vehicle and sensors, and control the vehicle.
    """
    actor_list = []  # List to keep track of spawned actors

    try:
        client = carla.Client('127.0.0.1', 2000)  # Connect to CARLA server
        client.set_timeout(10.0)  # Set connection timeout

        world = client.get_world()  # Retrieve the CARLA world
        map = world.get_map()  # Retrieve the map of the world

        # Get blueprint library and select vehicle blueprint
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]

        # Define spawn point for the vehicle
        spawn_point = carla.Transform(carla.Location(x=-75.4, y=-1.0, z=15), carla.Rotation(pitch=0, yaw=180, roll=0))

        # Spawn the vehicle and add to the actor list
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # Initialize the PID controller for the vehicle
        control_vehicle = VehiclePIDController(
            vehicle, 
            args_Lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},  # Lateral control PID parameters
            args_Longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0}  # Longitudinal control PID parameters
        )

        while True:
            # Retrieve waypoints from the map near the vehicle
            waypoints = world.get_map().get_waypoint(vehicle.get_location())
            
            # Select a random waypoint within 0.3 meters of the vehicle
            waypoint = np.random.choice(waypoints.next(0.3))

            # Compute and apply control signals to the vehicle
            control_signal = control_vehicle.run_step(5, waypoint)
            vehicle.apply_control(control_signal)

            # Uncomment these lines to add cameras
            camera_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
            camera_bp.set_attribute("image_size_x", '640')
            camera_bp.set_attribute("image_size_y", '480')
            camera_bp.set_attribute("fov", '90')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
             
            # Spawn a camera and save images to disk
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            camera.listen(lambda image: image.save_to_disk(f'output/dataset{time.time()}.png', carla.ColorConverter.CityScapesPalette))
            
            # Similar setup for depth camera (if needed)
            depth_camera_bp = blueprint_library.find("sensor.camera.depth")
            depth_camera_transform = carla.Transform(carla.Location(x=1.5, y=2.4))
            depth_camera = world.spawn_actor(depth_camera_bp, depth_camera_transform, attach_to=vehicle)
            depth_camera.listen(lambda image: image.save_to_disk(fr'C:\CARLA_0.9.5\PythonAPI\examples\output\dataset{time.time()}.png', carla.ColorConverter.LogarithmicDepth))
            
    finally:
        # Cleanup: destroy all spawned actors
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

if __name__ == "__main__":
    main()
