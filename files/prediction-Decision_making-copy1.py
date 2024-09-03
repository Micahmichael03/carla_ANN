# Importing necessary libraries for various functionalities
from distutils import dir_util
import time  # Provides time-related functions
import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
import math  # Provides mathematical functions
import sys  # Provides access to system-specific parameters and functions
import os  # Provides functions to interact with the operating system
import pathlib  # Object-oriented filesystem paths
# import yaml  # Library to parse YAML files
# import torch  # PyTorch library for deep learning
import logging  # Provides a way to log messages from code
import pygame  # Library for creating video games and multimedia applications
import glob  # Provides a way to find files and directories using patterns
import random  # Provides functions for generating random numbers
import networkx as nx  # Library for creating, manipulating, and studying complex networks
import queue  # Provides a FIFO queue implementation
# import six.moves.urllib as urllib  # Provides compatibility functions for URL handling
import tarfile  # Provides functions for working with tar archives
import tensorflow as tf  # TensorFlow library for deep learning
# sys.path.append('C:\Carla\CARLA_0.9.8\WindowsNoEditor\PythonAPI\models-master\models-master\research\slim')
import zipfile  # Provides functions for working with ZIP files

# Importing specific classes and functions from various libraries
from enum import Enum  # Provides support for creating enumerations
from collections import deque, defaultdict, OrderedDict  # Provides specialized container datatypes
from io import StringIO  # Provides an in-memory file-like object
from matplotlib import pyplot as plt  # Plotting library
from PIL import Image, ImageDraw  # Python Imaging Library for opening, manipulating, and saving images
from IPython.display import display  # Functions for displaying rich media in Jupyter notebooks
from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Importing specific functions and classes from custom modules
# from tool.geometry import update_intrinsics  # Custom function to update camera intrinsics
# from tool.config import Configuration, get_cfg  # Custom functions for configuration management
# from dataset.carla_dataset import ProcessImage, convert_slot_coord, ProcessSemantic, detokenize  # Custom functions for processing CARLA dataset
# from data_generation.network_evaluator import NetworkEvaluator  # Custom class for evaluating network performance
# from data_generation.tools import encode_npy_to_pil  # Custom function to encode NumPy arrays to PIL images
# from model.parking_model import ParkingModel  # Custom class for parking model
from agents.navigation.controller import VehiclePIDController  # Custom class for vehicle PID control

# Attempting to append the CARLA Python API to the system path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,  # Major version of Python
        sys.version_info.minor,  # Minor version of Python
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'  # OS-specific path
    ))[0])
except IndexError:
    pass  # If no matching path is found, do nothing

import carla  # Import the CARLA simulator API

# Functions to load models
def load_model(model_name):
    """
    Load a pre-trained TensorFlow model from the TensorFlow model zoo.
    Args:
    model_name (str): The name of the model to load from the TensorFlow model zoo.
    Returns:
    tf.SavedModel: The loaded TensorFlow model.
    """
    print('Loading model...')
    base_url = 'http://download.tensorflow.org/models/object_detection/'  # Base URL for TensorFlow models
    model_file = model_name + '.tar.gz'  # Model file name
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)  # Download and extract model
    model_dir = pathlib.Path(model_dir) / "saved_model"  # Path to saved model
    model = tf.saved_model.load(str(model_dir))  # Load the TensorFlow model
    print('Model loaded.')
    return model

def load_custom_model(custom_model_path):
    """
    Load a custom-trained TensorFlow model.
    Args:
    custom_model_path (str): The file path to the custom-trained model directory.
    Returns:
    tf.SavedModel: The loaded custom-trained TensorFlow model.
    """
    print('Loading custom trained model...')
    custom_model = tf.saved_model.load(custom_model_path)  # Load the custom TensorFlow model
    print('Custom model loaded.')
    return custom_model

# # Define the path to the label map
# PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'  # Path to the label map file
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)  # Create category index from label map

# Load pre-trained and custom models
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'  # Name of the pre-trained model
detection_model = load_model(model_name)  # Load the pre-trained model
custom_model_path = 'C:\Carla\saved_model\saved_model.keras'  # Path to the custom-trained model
custom_model = load_custom_model(custom_model_path)  # Load the custom-trained model
print('Models loaded.')

# Initializing an empty list to keep track of actors
actor_list = []  # List to store CARLA actors

# Function to handle camera data and update the display
def camera_callback(image, camera_data):
    """
    Callback function to process camera images and update the display.
    :param image: Raw image data from the camera.
    :param camera_data: Dictionary to store the camera image.
    """
    image.convert(carla.ColorConverter.Raw)  # Convert image to raw format
    array = np.frombuffer(image.raw_data, dtype=np.uint8)  # Convert raw image data to NumPy array
    array = array.reshape((image.height, image.width, 4))  # Reshape array to image dimensions
    camera_data['image'] = array  # Store image data in the dictionary

def draw_route(world, route):
    """
    Draw the route on the simulation window.
    :param world: The CARLA world object.
    :param route: List of waypoints representing the route.
    """
    for waypoint in route:
        world.debug.draw_string(
            waypoint[0].transform.location, '^', 
            draw_shadow=False,
            color=carla.Color(r=0, g=0, b=255), 
            life_time=30.0,
            persistent_lines=True
        )  # Draws the route waypoints in blue color

# Speed threshold and maximum steering angle
SPEED_THRESHOLD = 2  # Speed difference threshold for adjusting throttle
MAX_STEER_DEGREES = 40  # Maximum steering angle in degrees
STEERING_CONVERSION = 75  # Conversion factor for steering

# Text display parameters
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type for text display
org = (30, 30)  # Origin point for text display
fontScale = 0.5  # Font scale factor
color = (255, 255, 255)  # Color of the text (white)
thickness = 1  # Thickness of the text


class Normal(object):
    # Define the maximum speed (km/h) the vehicle can travel
    max_speed = 50
    # Define the distance (meters) to check for speed limit changes
    speed_lim_dist = 3
    # Define the amount (km/h) to decrease speed when necessary
    speed_decrease = 10
    # Define the safety time (seconds) to maintain distance from other vehicles
    safety_time = 3
    # Define the minimum proximity threshold (meters) to consider for other vehicles
    min_proximity_threshold = 10
    # Define the distance (meters) needed for the vehicle to brake safely
    braking_distance = 6
    # Counter for how long to maintain overtaking behavior, -1 means no overtaking
    overtake_counter = -1
    # Counter for how long to maintain tailgating behavior, initially 0 means no tailgating
    tailgate_counter = 0

class RoadOption(Enum):
    # Represents an undefined or invalid road option
    VOID = -1
    # Represents a left turn option
    LEFT = 1
    # Represents a right turn option
    RIGHT = 2
    # Represents a straight road option
    STRAIGHT = 3
    # Represents an option to follow the current lane
    LANEFOLLOW = 4
    # Represents an option to change to the left lane
    CHANGELANELEFT = 5
    # Represents an option to change to the right lane
    CHANGELANERIGHT = 6

# Import necessary modules
from collections import deque

class LocalPlanner(object):
    # Frames per second for the planner
    FPS = 20

    # Initialization method
    def __init__(self, agent):
        # Store the vehicle object from the agent
        self._vehicle = agent.vehicle
        # Get the map of the current world from the vehicle
        self._map = agent.vehicle.get_world().get_map()
        # Target speed for the vehicle
        self._target_speed = None
        # Radius for sampling waypoints
        self.sampling_radius = None
        # Minimum distance to the target waypoint
        self._min_distance = 3
        # Current distance to the target waypoint
        self._current_distance = None
        # Target road option (e.g., left, right, straight)
        self.target_road_option = None
        # Controller for the vehicle
        self._vehicle_controller = None
        # Global plan for the vehicle's route
        self._global_plan = None
        # PID controller for the vehicle
        self._pid_controller = None
        # Queue for storing waypoints
        self.waypoints_queue = deque(maxlen=20000)
        # Buffer size for waypoints
        self._buffer_size = 5
        # Buffer for storing the most recent waypoints
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # PID controller parameters for lateral control on highways
        self.args_lat_hw_dict = {
            'K_P': 0.75,  # Proportional gain
            'K_D': 0.02,  # Derivative gain
            'K_I': 0.4,   # Integral gain
            'dt': 1.0 / self.FPS  # Time step
        }

        # PID controller parameters for lateral control in city driving
        self.args_lat_city_dict = {
            'K_P': 0.58,  # Proportional gain
            'K_D': 0.02,  # Derivative gain
            'K_I': 0.5,   # Integral gain
            'dt': 1.0 / self.FPS  # Time step
        }

        # PID controller parameters for longitudinal control on highways
        self.args_long_hw_dict = {
            'K_P': 0.37,  # Proportional gain
            'K_D': 0.024, # Derivative gain
            'K_I': 0.032, # Integral gain
            'dt': 1.0 / self.FPS  # Time step
        }

        # PID controller parameters for longitudinal control in city driving
        self.args_long_city_dict = {
            'K_P': 0.15,  # Proportional gain
            'K_D': 0.05,  # Derivative gain
            'K_I': 0.07,  # Integral gain
            'dt': 1.0 / self.FPS  # Time step
        }


def get_speed(vehicle):
    vel = vehicle.get_velocity()  # Get the velocity vector of the vehicle
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # Calculate speed in km/h

def maintain_speed(s):
    ''' 
    Adjust the throttle based on the current speed to maintain a desired speed.
    :param s: The current speed of the vehicle in kilometers per hour.
    :return: A throttle value to adjust the vehicle's speed.
    '''
    THROTTLE_FULL = 0.9  # Full throttle
    THROTTLE_MODERATE = 0.4  # Moderate throttle
    THROTTLE_LOW = 0.1  # Low throttle
    THROTTLE_MIN = 0.0  # Minimum throttle
    PREFERRED_SPEED = 0.5

    if s >= PREFERRED_SPEED:  # If speed is above or equal to preferred speed
        return THROTTLE_MIN
    elif s < PREFERRED_SPEED - SPEED_THRESHOLD:  # If speed is below preferred speed minus threshold
        return THROTTLE_FULL
    else:  # For intermediate speed
        return THROTTLE_MODERATE

def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location  # Get the vehicle's location
    x = waypoint.transform.location.x - loc.x  # Calculate x distance to waypoint
    y = waypoint.transform.location.y - loc.y  # Calculate y distance to waypoint
    return math.sqrt(x**2 + y**2)  # Return Euclidean distance

def vector(location_1, location_2):
    x = location_2.x - location_1.x  # Calculate x component of the vector
    y = location_2.y - location_1.y  # Calculate y component of the vector
    z = location_2.z - location_1.z  # Calculate z component of the vector
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps  # Calculate vector norm
    return [x / norm, y / norm, z / norm]  # Return normalized vector

def angle_between(v1, v2):
    return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))  # Calculate angle between vectors in degrees

def get_angle(car, wp):
    vehicle_pos = car.get_transform()  # Get vehicle's transformation (position and orientation)
    car_x = vehicle_pos.location.x  # Vehicle's x coordinate
    car_y = vehicle_pos.location.y  # Vehicle's y coordinate

    wp_x = wp.transform.location.x  # Waypoint's x coordinate
    wp_y = wp.transform.location.y  # Waypoint's y coordinate

    x = (wp_x - car_x) / ((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5  # Normalize x component
    y = (wp_y - car_y) / ((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5  # Normalize y component

    car_vector = vehicle_pos.get_forward_vector()  # Get vehicle's forward direction vector
    degrees = angle_between((x, y), (car_vector.x, car_vector.y))  # Calculate angle between car direction and waypoint direction

    return degrees

class GlobalRoutePlannerDAO(object):
    # Constructor method to initialize the class
    def __init__(self, wmap, sampling_resolution):
        # Initialize the sampling resolution attribute
        self._sampling_resolution = sampling_resolution
        # Initialize the world map attribute
        self._wmap = wmap

    # Method to get the topology of the road network
    def get_topology(self):
        """
        Accessor for topology.
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects.

        :return topology: list of dictionary objects with the following attributes
            entry   -   waypoint of entry point of road segment
            entryxyz-   (x,y,z) of entry point of road segment
            exit    -   waypoint of exit point of road segment
            exitxyz -   (x,y,z) of exit point of road segment
            path    -   list of waypoints separated by 1m from entry
                        to exit
        """
        # Initialize an empty list to store the topology
        topology = []
        # Retrieve waypoints from the map to construct a detailed topology
        for segment in self._wmap.get_topology():
            # Get the entry and exit waypoints of the segment
            wp1, wp2 = segment[0], segment[1]
            # Get the locations (x, y, z) of the entry and exit waypoints
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Round off the locations to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            # Update the locations of the waypoints
            wp1.transform.location, wp2.transform.location = l1, l2
            # Initialize a dictionary to store the segment information
            seg_dict = dict()
            # Store the entry and exit waypoints in the dictionary
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            # Store the (x, y, z) coordinates of the entry and exit points
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            # Initialize an empty list to store the path waypoints
            seg_dict['path'] = []
            # Get the location of the exit waypoint
            endloc = wp2.transform.location
            # If the distance between the entry and exit waypoints is greater than the sampling resolution
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                # Get the next waypoint at the sampling resolution distance from the entry waypoint
                w = wp1.next(self._sampling_resolution)[0]
                # While the distance to the exit waypoint is greater than the sampling resolution
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    # Append the waypoint to the path
                    seg_dict['path'].append(w)
                    # Get the next waypoint at the sampling resolution distance
                    w = w.next(self._sampling_resolution)[0]
            else:
                # If the distance is less than the sampling resolution, append the next waypoint directly
                seg_dict['path'].append(wp1.next(self._sampling_resolution)[0])
            # Append the segment dictionary to the topology list
            topology.append(seg_dict)
        # Return the topology list
        return topology

    # Method to get the sampling resolution
    def get_resolution(self):
        # Return the sampling resolution attribute
        return self._sampling_resolution

    # Method to get the waypoint at a specific location
    def get_waypoint(self, location):
        # Get the waypoint from the map at the specified location
        waypoint = self._wmap.get_waypoint(location)
        # Return the waypoint
        return waypoint

class GlobalRoutePlanner(object):

    def __init__(self, dao):
        # Initialize the GlobalRoutePlanner with the Data Access Object (DAO)
        self._dao = dao
        # Initialize topology, graph, id_map, and road_id_to_edge to None
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None
        
        # Initialize intersection_end_node and previous_decision with default values
        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID

    def setup(self):
        # Print a message indicating that setup has been called
        print('setup called')
        # Get the topology from the DAO
        self._topology = self._dao.get_topology()
        # Build the graph, id_map, and road_id_to_edge from the topology
        self._graph, self._id_map, self._road_id_to_edge = self._build_graph()
        # Find loose ends in the graph
        self._find_loose_ends()
        # Link lane changes in the graph
        self._lane_change_link()

    def _build_graph(self):
        # Print a message indicating that _build_graph has been called
        print('_build_graph called')
        # Initialize a directed graph using NetworkX
        graph = nx.DiGraph()
        # Initialize id_map and road_id_to_edge as empty dictionaries
        id_map = dict()
        road_id_to_edge = dict()

        # Iterate over each segment in the topology
        for segment in self._topology:
            # Get entry and exit coordinates and waypoints of the segment
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']

            # Check if the entry waypoint is in a junction
            intersection = entry_wp.is_junction
            # Get road ID, section ID, and lane ID from the entry waypoint
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            # Iterate over entry and exit coordinates
            for vertex in entry_xyz, exit_xyz:
                # If the vertex is not in the id_map, add it with a new ID
                if vertex not in id_map:
                    new_id = len(id_map)
                    id_map[vertex] = new_id
                    graph.add_node(new_id, vertex=vertex)

            # Get the node IDs for the entry and exit coordinates
            n1 = id_map[entry_xyz]
            n2 = id_map[exit_xyz]

            # Initialize the road_id_to_edge dictionary for the current road, section, and lane
            if road_id not in road_id_to_edge:
                road_id_to_edge[road_id] = dict()
            if section_id not in road_id_to_edge[road_id]:
                road_id_to_edge[road_id][section_id] = dict()
            road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            # Get the forward vectors of the entry and exit waypoints
            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

            # Add an edge to the graph with relevant information about the segment
            graph.add_edge(
                n1, n2, length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array([entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array([exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW
            )
        # Return the graph, id_map, and road_id_to_edge
        return graph, id_map, road_id_to_edge

def _find_loose_ends(self):
    # Print a message indicating that _find_loose_ends has been called
    print('_find_loose_ends called')

    # Initialize a counter for loose ends
    count_loose_ends = 0

    # Get the hop resolution from the DAO
    hop_resolution = self._dao.get_resolution()

    # Iterate over each segment in the topology
    for segment in self._topology:
        # Get the exit waypoint and its coordinates
        end_wp = segment['exit']
        exit_xyz = segment['exitxyz']
        # Get road ID, section ID, and lane ID from the exit waypoint
        road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id

        # Check if the road, section, and lane exist in road_id_to_edge
        if road_id in self._road_id_to_edge and section_id in self._road_id_to_edge[road_id] and lane_id in self._road_id_to_edge[road_id][section_id]:
            pass
        else:
            # Increment the loose ends counter if not found
            count_loose_ends += 1
            # Initialize the road_id_to_edge dictionary for the current road, section, and lane
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()

            # Get the node ID for the exit coordinates
            n1 = self._id_map[exit_xyz]
            # Assign a unique negative ID for loose ends
            n2 = -1 * count_loose_ends
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            # Get the next waypoint and initialize the path
            next_wp = end_wp.next(hop_resolution)
            path = []
            # Iterate over waypoints in the same road, section, and lane
            while next_wp is not None and next_wp[0].road_id == road_id and next_wp[0].section_id == section_id and next_wp[0].lane_id == lane_id:
                path.append(next_wp[0])
                next_wp = next_wp[0].next(hop_resolution)

            if path:
                # Get the coordinates of the last waypoint in the path
                n2_xyz = (path[-1].transform.location.x, path[-1].transform.location.y, path[-1].location.z)
                # Add a node to the graph for the last waypoint
                self._graph.add_node(n2, vertex=n2_xyz)
                # Add an edge to the graph with relevant information about the segment
                self._graph.add_edge(
                    n1, n2, length=len(path) + 1, path=path,
                    entry_waypoint=end_wp, exit_waypoint=path[-1],
                    entry_vector=None, exit_vector=None,
                    net_vector=None, intersection=end_wp.is_junction,
                    type=RoadOption.LANEFOLLOW
                )

def _localize(self, location):
    # Print a message indicating that _localize has been called
    print('_localize called')

    # Get the waypoint for the given location
    waypoint = self._dao.get_waypoint(location)
    edge = None
    try:
        # Try to get the edge from road_id_to_edge using the waypoint's road ID, section ID, and lane ID
        edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
    except KeyError:
        # Print a message if localization fails
        print('failed to localize')

    return edge

def _lane_change_link(self):
    # Print a message indicating that _lane_change_link has been called
    print('_lane_change_link called')

    # Iterate over each segment in the topology
    for segment in self._topology:
        left_found, right_found = False, False

        # Iterate over waypoints in the segment path
        for waypoint in segment['path']:
            # Skip if the entry waypoint is in a junction
            if not segment['entry'].is_junction:
                next_waypoint, next_road_option, next_segment = None, None, None

                # Check for right lane change
                if waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and not right_found:
                    next_waypoint = waypoint.get_right_lane()
                    if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and waypoint.road_id == next_waypoint.road_id:
                        next_road_option = RoadOption.CHANGELANERIGHT
                        next_segment = self._localize(next_waypoint.transform.location)
                        if next_segment is not None:
                            self._graph.add_edge(
                                self._id_map[segment['entryxyz']], next_segment[0],
                                entry_waypoint=waypoint, exit_waypoint=next_waypoint,
                                intersection=False, exit_vector=None, path=[], length=0,
                                type=next_road_option, change_waypoint=next_waypoint
                            )
                            right_found = True

                # Check for left lane change
                if waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:
                    next_waypoint = waypoint.get_left_lane()
                    if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and waypoint.road_id == next_waypoint.road_id:
                        next_road_option = RoadOption.CHANGELANELEFT
                        next_segment = self._localize(next_waypoint.transform.location)
                        if next_segment is not None:
                            self._graph.add_edge(
                                self._id_map[segment['entryxyz']], next_segment[0],
                                entry_waypoint=waypoint, exit_waypoint=next_waypoint,
                                intersection=False, exit_vector=None, path=[], length=0,
                                type=next_road_option, change_waypoint=next_waypoint
                            )
                            left_found = True

                # Break if both left and right lane changes are found
                if left_found and right_found:
                    break

def _distance_heuristic(self, n1, n2):
    # Get the coordinates of the two nodes
    l1 = np.array(self._graph.nodes[n1]['vertex'])
    l2 = np.array(self._graph.nodes[n2]['vertex'])
    # Return the Euclidean distance between the nodes
    return np.linalg.norm(l1 - l2)

def _path_search(self, origin, destination):
    # Print a message indicating that _path_search has been called
    print('_path_search called')

    # Get the start and end nodes by localizing the origin and destination
    start, end = self._localize(origin), self._localize(destination)

    # Use A* search to find the shortest path from start to end nodes
    route = nx.astar_path(self._graph, source=start[0], target=end[0], heuristic=self._distance_heuristic, weight='length')
    # Append the end node's edge to the route
    route.append(end[1])
    return route

def _successive_last_intersection_edge(self, index, route):
    # Print a message indicating that _successive_last_intersection_edge has been called
    print('_successive_last_intersection_edge called')

    last_intersection_edge = None
    last_node = None

    # Iterate over node pairs in the route starting from the given index
    for node1, node2 in [(route[i], route[i + 1]) for i in range(index, len(route) - 1)]:
        candidate_edge = self._graph.edges[node1, node2]
        if node1 == route[index]:
            last_intersection_edge = candidate_edge

        # Check if the edge is a LANEFOLLOW and is at an intersection
        if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:
            last_intersection_edge = candidate_edge
            last_node = node2
        else:
            break

    return last_node, last_intersection_edge

def _turn_decision(self, index, route, threshold=math.radians(35)):
    # Print a message indicating that _turn_decision has been called
    print('_turn_decision called')

    decision = None
    previous_node = route[index - 1]
    current_node = route[index]
    next_node = route[index + 1]
    next_edge = self._graph.edges[current_node, next_node]

    if index > 0:
        if self._previous_decision != RoadOption.VOID and self._intersection_end_node > 0 and self._intersection_end_node != previous_node and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']:
            decision = self._previous_decision
        else:
            self._intersection_end_node = -1
            current_edge = self._graph.edges[previous_node, current_node]
            calculate_turn = current_edge['type'] == RoadOption.LANEFOLLOW and not current_edge['intersection'] and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']

            if calculate_turn:
                last_node, tail_edge = self._successive_last_intersection_edge(index, route)
                self._intersection_end_node = last_node
                if tail_edge is not None:
                    next_edge = tail_edge

                cv, nv = current_edge['exit_vector'], next_edge['exit_vector']
                if cv is None or nv is None:
                    return next_edge['type']

                cross_list = []
                for neighbor in self._graph.successors(current_node):
                    select_edge = self._graph.edges[current_node, neighbor]
                    if select_edge['type'] == RoadOption.LANEFOLLOW:
                        if neighbor != route[index + 1]:
                            sv = select_edge['net_vector']
                            cross_list.append(np.cross(cv, sv)[2])

                next_cross = np.cross(cv, nv)[2]
                deviation = math.acos(np.clip(np.dot(cv, nv) / (np.linalg.norm(nv)), -1.0, 1.0))

                if not cross_list:
                    cross_list.append(0)

                if deviation < threshold:
                    decision = RoadOption.STRAIGHT
                elif cross_list and next_cross < min(cross_list):
                    decision = RoadOption.LEFT
                elif cross_list and next_cross > max(cross_list):
                    decision = RoadOption.RIGHT
                elif next_cross < 0:
                    decision = RoadOption.LEFT
                elif next_cross > 0:
                    decision = RoadOption.RIGHT
            else:
                decision = next_edge['type']

            self._previous_decision = decision
            return decision

def _find_closest_in_list(self, current_waypoint, waypoint_list):
    # Print a message indicating that _find_closest_in_list has been called
    print('_find_closest_in_list called')

    min_distance = float('inf')
    closest_index = -1

    # Iterate over the waypoints in the list to find the closest one
    for i, waypoint in enumerate(waypoint_list):
        distance = waypoint.transform.location.distance(current_waypoint.transform.location)
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index

def trace_route(self, origin, destination):
    # Print a message indicating that trace_route has been called
    print('trace_route of GlobalRoutePlanner called')

    route_trace = []
    route = self._path_search(origin, destination)
    current_waypoint = self._dao.get_waypoint(origin)
    destination_waypoint = self._dao.get_waypoint(destination)
    resolution = self._dao.get_resolution()

    for i in range(len(route) - 1):
        road_option = self._turn_decision(i, route)
        edge = self._graph.edges[route[i], route[i + 1]]
        path = []

        if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:
            route_trace.append((current_waypoint, road_option))
            exit_wp = edge['exit_waypoint']
            n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
            next_edge = self._graph.edges[n1, n2]

            if next_edge['path']:
                closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                closest_index = min(len(next_edge['path']) - 1, closest_index + 5)
                current_waypoint = next_edge['path'][closest_index]
            else:
                current_waypoint = next_edge['exit_waypoint']

            route_trace.append((current_waypoint, road_option))
        else:
            path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
            closest_index = self._find_closest_in_list(current_waypoint, path)
            for waypoint in path[closest_index:]:
                current_waypoint = waypoint
                route_trace.append((current_waypoint, road_option))

                if len(route) - i <= 2 and waypoint.transform.location.distance(destination) < 2 * resolution:
                    break
                elif len(route) - i <= 2 and current_waypoint.road_id == destination_waypoint.road_id and current_waypoint.section_id == destination_waypoint.section_id and current_waypoint.lane_id == destination_waypoint.lane_id:
                    destination_index = self._find_closest_in_list(destination_waypoint, path)
                    if closest_index > destination_index:
                        break

    return route_trace
    
class Behavior:
    def __init__(self):
        self.tailgate_counter = 0
        self.overtake_counter = 0
        self.braking_distance = 10.0  # Braking distance in meters
        self.safety_time = 2.0  # Safety time in seconds
        self.speed_decrease = 10.0  # Speed decrease value in km/h
        self.max_speed = 30.0  # Maximum speed in km/h
        self.speed_lim_dist = 5.0  # Speed limit distance in meters
        self.min_proximity_threshold = 10.0  # Minimum proximity threshold in meters

class Normal(Behavior):
    pass  # Normal behavior inherits from Behavior

class Agent(object):
    def __init__(self, vehicle):
        self._vehicle = vehicle  # Vehicle controlled by the agent
        self._proximity_tlight_threshold = 5.0  # Proximity threshold for traffic lights
        self._proximity_vehicle_threshold = 10.0  # Proximity threshold for vehicles
        self._local_planner = None  # Local planner instance
        self._world = self._vehicle.get_world()  # World instance
        try:
            self._map = self._world.get_map()  # Map instance
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
        self._last_traffic_light = None  # Last traffic light encountered

    def _bh_is_vehicle_hazard(self, ego_wpt, ego_loc, vehicle_list, proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):
        print('_bh_is_vehicle_hazard called')
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1  # Adjust lane offset for negative lane IDs
        for target_vehicle in vehicle_list:
            target_vehicle_loc = target_vehicle.get_location()  # Get the location of the target vehicle
            target_wpt = self._map.get_waypoint(target_vehicle_loc)  # Get the waypoint of the target vehicle
            if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=5)[0]
                if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                    continue
            if carla.is_within_distance(target_vehicle_loc, ego_loc, self._vehicle.get_transform().rotation.yaw, proximity_th, up_angle_th, low_angle_th):
                return (True, target_vehicle, carla.compute_distance(target_vehicle_loc, ego_loc))  # Return if the target vehicle is a hazard
        return (False, None, -1)

    @staticmethod
    def emergency_stop():
        print('emergency_stop called')
        control = carla.VehicleControl()  # Create a new vehicle control object
        control.steer = 0.0  # Set steering to 0
        control.throttle = 0.0  # Set throttle to 0
        control.brake = 1.0  # Set brake to maximum
        control.hand_brake = False  # Disable hand brake
        return control  # Return the control object

# Define the BehavoirAgent class that inherits from Agent
class BehavoirAgent(Agent):

    # Initialization method for the BehavoirAgent class
    def __init__(self, vehicle, ignore_traffic_light=False, behavior='normal'):
        # Call the parent class (Agent) constructor
        super(BehavoirAgent, self).__init__(vehicle)
        # Assign the vehicle to an instance variable
        self.vehicle = vehicle
        # Whether to ignore traffic lights or not
        self.ignore_traffic_light = ignore_traffic_light
        # Initialize the local planner with the current agent
        self._local_planner = LocalPlanner(self)
        # Placeholder for the global route planner
        self._grp = None
        # Look ahead steps for the planner
        self.look_ahead_steps = 0

        # Initialize vehicle information
        self.speed = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.start_waypoint = None
        self.end_waypoint = None
        self.is_at_traffic_light = 0
        self.light_state = "Green"
        self.light_id_to_ignore = -1
        self.min_speed = 5
        self.behavior = None
        self._sampling_resolution = 4.5

        # Set behavior to Normal if specified
        if behavior == 'normal':
            self.behavior = Normal()

    # Method to update vehicle information
    def update_information(self):
        print('update_information called')

        # Get the current speed of the vehicle
        self.speed = get_speed(self.vehicle)
        # Get the speed limit of the current road
        self.speed_limit = self.vehicle.get_speed_limit()
        # Set the speed limit for the local planner
        self._local_planner.set_speed(self.speed_limit)
        # Get the current direction from the local planner
        self.direction = self._local_planner.target_road_option

        # Default to LANEFOLLOW if no direction is specified
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW

        # Determine look ahead steps based on speed limit
        self.look_ahead_steps = int((self.speed_limit) / 10)

        # Get the incoming waypoint and direction
        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(steps=self.look_ahead_steps)

        # Default to LANEFOLLOW if no incoming direction is specified
        if self.incoming_direction is None:
            self.direction = RoadOption.LANEFOLLOW

        # Check if the vehicle is at a traffic light
        self.is_at_traffic_light = self.vehicle.is_at_traffic_light()

        # Set light state to Green if ignoring traffic lights
        if self.ignore_traffic_light:
            self.light_state = "Green"
        else:
            # Get the current traffic light state
            self.light_state = str(self.vehicle.get_traffic_light_state())

    # Method to set the destination for the agent
    def set_destination(self, start_location, end_location, clean=False):
        print('set_destination called')

        # Clear waypoints queue if clean is specified
        if clean:
            self._local_planner.waypoints_queue.clear()

        # Get the start and end waypoints from the map
        self.start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)

        # Trace the route from start to end waypoint
        route_trace = self._trace_route(self.start_waypoint, self.end_waypoint)

        # Set the global plan for the local planner
        self._local_planner.set_global_plan(route_trace, clean)

    # Method to trace the route from start to end waypoint
    def _trace_route(self, start_waypoint, end_waypoint):
        print('_trace_route BehavoirAgent called')

        # Initialize global route planner if not already initialized
        if self._grp is None:
            wld = self.vehicle.get_world()
            dao = GlobalRoutePlannerDAO(wld.get_map(), sampling_resolution=self._sampling_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Trace the route from start to end waypoint
        route = self._grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        return route

    # Method to reroute the agent to a new destination
    def reroute(self, spawn_points):
        print('reroute called')

        # Shuffle the spawn points to get a random new destination
        random.shuffle(spawn_points)
        # Get the last waypoint in the queue as the new start location
        new_start = self._local_planner.waypoints_queue[-1][0].transform.location
        # Choose a new destination that is not the same as the start location
        destination = spawn_points[0].location if spawn_points[0].location != new_start else spawn_points[1].location

        # Set the new destination
        self.set_destination(new_start, destination)

    # Method to manage traffic lights
    def traffic_light_manager(self, waypoint):
        print('traffic_light_manager called')

        # Get the ID of the current traffic light
        light_id = self.vehicle.get_traffic_light().id if self.vehicle.get_traffic_light() is not None else -1

        # Check if the light is red
        if self.light_state == "Red":
            # If not at a junction and light is not to be ignored
            if not waypoint.is_junction and (self.light_id_to_ignore != light_id or light_id == -1):
                return 1
            # If at a junction, set the light ID to ignore
            elif waypoint.is_junction and light_id != -1:
                self.light_id_to_ignore = light_id

        # Reset light ID to ignore if necessary
        if self.light_id_to_ignore != light_id:
            light_id_to_ignore = -1

        return 0

    # Method to handle overtaking
    def _overtake(self, location, waypoint, vehicle_list):
        print('overtake called')

        left_turn = None
        right_turn = None

        # Get the left and right waypoints
        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        # Check if overtaking to the left is possible
        if (left_turn == carla.LaneChange.Left or left_turn == carla.LaneChange.Both) and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
            new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=-1)
            if not new_vehicle_state:
                self.behavior.overtake_counter = 200
                self.set_destination(left_wpt.transform.location, self.end_waypoint, clean=True)

        # Check if overtaking to the right is possible
        elif right_turn == carla.LaneChange.Right and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
            new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=1)
            if not new_vehicle_state:
                self.behavior.overtake_counter = 200
                self.set_destination(right_wpt.transform.location, self.end_waypoint.transform.location, clean=True)

    # Method to handle tailgating
    def _tailgating(self, location, waypoint, vehicle_list):
        print('tailgating called')

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        # Get the left and right waypoints
        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        # Check if there is a vehicle hazard behind
        behind_vehicle_state, behind_vehicle, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self.speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle:
                    print('tailgating moving towards right')
                    self.behavior.tailgate_counter = 200
                    self.set_destination(right_wpt.transform.location, self.end_waypoint.transform.location, clean=True)

            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not carla.new_vehicle_state:
                    print('tailgating, moving to the left!')
                    self.behavior.tailgate_counter = 200
                    self.set_destination(left_wpt.transform.location, self.end_waypoint.transform.location, clean=True)

    # Method to manage collision and car avoidance
    def collision_and_car_avoid_manager(self, location, waypoint):
        print('collision_and_car_avoid_manager called')

        # Get the list of nearby vehicles
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        # Function to calculate distance between vehicle and waypoint
        def dist(v):
            return v.get_location().distance(waypoint.transform.location)

        # Filter vehicles within a 45-meter radius
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self.vehicle.id]

        # Check for vehicle hazard based on direction
        if self.direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=30)

            # If a vehicle hazard is detected, perform overtaking or tailgating
            if vehicle_state and self.direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and self.speed > 10 and self.behavior.overtake_counter == 0 and self.speed > get_speed(vehicle):
                self._overtake(location, waypoint, vehicle_list)
            elif not vehicle_state and self.direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and self.speed > 10 and self.behavior.tailgate_counter == 0:
                self._tailgating(location, waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    # Method to manage car following behavior
    def car_following_manager(self, vehicle, distance, debug=False):
        print('car_following_manager called')

        # Get the speed of the vehicle being followed
        vehicle_speed = get_speed(vehicle)

        # Calculate the speed difference in m/s
        delta_v = max(1, (self.speed - vehicle_speed) / 3.6)

        # Calculate time to collision (TTC)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Determine the control based on TTC
        if self.behavior.safety_time > ttc > 0.0:
            control = self._local_planner.run_step(target_speed=min(np.positive(vehicle_speed - self.behavior.speed_decrease), min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)), debug=debug)
        elif 2 * self.behavior.safety_time > ttc >= self.behavior.safety_time:
            control = self._local_planner.run_step(target_speed=min(max(self.min_speed, vehicle_speed), min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)), debug=debug)
        else:
            control = self._local_planner.run_step(target_speed=min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)

        return control

    # Main method to execute a step in the agent's behavior
    def run_step(self, debug=False):
        print('run_step function called')

        # Update vehicle information
        self.update_information()

        control = None
        # Decrement counters for tailgating and overtaking
        if self.behavior.tailgate_counter > 0:
            self.behavior.tailgate_counter -= 1

        if self.behavior.overtake_counter > 0:
            self.behavior.overtake_counter -= 1

        # Get the current location and waypoint of the ego vehicle
        ego_vehicle_loc = self.vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # Check traffic light status
        if self.traffic_light_manager(ego_vehicle_wp) != 0:
            return self.emergency_stop()

        # Manage collision and car avoidance
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_loc, ego_vehicle_wp)

        if vehicle_state:
            # Adjust distance for bounding boxes
            distance = distance - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(self.vehicle.bounding_box.extent.y, self.vehicle.bounding_box.extent.x)
            if distance < self.behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)
        elif self.incoming_waypoint and self.incoming_waypoint.is_junction and (self.incoming_direction == RoadOption.LEFT or self.incoming_direction == RoadOption.RIGHT):
            control = self._local_planner.run_step(target_speed=min(self.behavior.max_speed, self.speed_limit - 5), debug=debug)
        else:
            control = self._local_planner.run_step(target_speed=min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)

        return control


# Main block to run the simulation
try:
    print('Started')

    client = carla.Client('localhost', 2000)  # Connect to CARLA server
    client.set_timeout(10.0)  # Set timeout for server communication
    world = client.get_world()  # Get the CARLA world
    print('Town map loaded')

    blueprint_library = world.get_blueprint_library()  # Get the blueprint library for creating actors
    vehicle_bp = blueprint_library.filter('cybertruck')[0]  # Get blueprint for Cybertruck vehicle or any vehicle like model3

    spawnpoint = carla.Transform(carla.Location(x=130, y=195, z=40), carla.Rotation(yaw=180))  # Define spawn point for the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawnpoint)  # Spawn the vehicle at the spawn point
    actor_list.append(vehicle)  # Add vehicle to the list of actors

    agent = BehavoirAgent(vehicle)  # Create a behavior agent to control the vehicle
    print('Agent created')

    spawn_points = world.get_map().get_spawn_points()  # Get all possible spawn points
    random.shuffle(spawn_points)  # Shuffle spawn points to choose a random destination
    destination = spawn_points[0].location if spawn_points[0].location != vehicle.get_location() else spawn_points[1].location  # Set destination
    agent.set_destination(vehicle.get_location(), destination, clean=True)  # Set the agent's destination

    point_a = spawnpoint.location  # Start point for route planning
    sampling_resolution = 1  # Resolution for route planning
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)  # Create global route planner

    distance = 0  # Initialize distance variable
    for loc in spawn_points:  # Iterate over spawn points
        cur_route = grp.trace_route(point_a, loc.location)  # Get route from start point to current spawn point
        if len(cur_route) > distance:  # Update longest route
            distance = len(cur_route)
            route = cur_route

    draw_route(world, route)  # Draw the route on the carla simulation window

    camera_positions = [
        carla.Transform(carla.Location(x=1.5, z=2.4)),  # Front camera
        carla.Transform(carla.Location(x=-1.5, z=2.4)), # Rear camera
        carla.Transform(carla.Location(x=0, y=1.0, z=2.4)),  # Left camera
        carla.Transform(carla.Location(x=0, y=-1.0, z=2.4)), # Right camera
        carla.Transform(carla.Location(x=1.5, y=1.0, z=2.4)),  # Front Left camera
        carla.Transform(carla.Location(x=1.5, y=-1.0, z=2.4)), # Front Right camera
        carla.Transform(carla.Location(x=-1.5, y=1.0, z=2.4)), # Rear Left camera
        carla.Transform(carla.Location(x=-1.5, y=-1.0, z=2.4)) # Rear Right camera
    ]

    camera_data = [{} for _ in camera_positions]  # Initialize list to store camera data
    for idx, cam_transform in enumerate(camera_positions):  # Iterate over camera positions
        cam_bp = blueprint_library.find('sensor.camera.rgb')  # Get blueprint for RGB camera
        cam_bp.set_attribute('image_size_x', '800')  # Set image width
        cam_bp.set_attribute('image_size_y', '600')  # Set image height
        cam_bp.set_attribute('fov', '110')  # Set field of view

        sensor = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)  # Spawn camera sensor
        actor_list.append(sensor)  # Add camera to the list of actors
        sensor.listen(lambda image, idx=idx: camera_callback(image, camera_data[idx]))  # Set up callback for camera data

    # Draw bounding boxes around detected objects
    def draw_bounding_boxes(image, boxes, color):
        """
        Draw bounding boxes on an image.
        :param image: The image on which to draw the bounding boxes.
        :param boxes: List of bounding box coordinates.
        :param color: The color to use for the bounding boxes.
        :return: The image with bounding boxes drawn.
        """
        for box in boxes:  # Iterate over bounding boxes
            ymin, xmin, ymax, xmax = box  # Extract bounding box coordinates
            start_point = (int(xmin * image.shape[1]), int(ymin * image.shape[0]))  # Top-left corner
            end_point = (int(xmax * image.shape[1]), int(ymax * image.shape[0]))  # Bottom-right corner
            image = cv2.rectangle(image, start_point, end_point, color, 2)  # Draw rectangle on image
        return image

    while True:
        world_snapshot = world.get_snapshot()  # Get a snapshot of the world
        vehicle_snapshot = world_snapshot.find(vehicle.id)  # Find the vehicle's snapshot
        control = vehicle.get_control()  # Get current vehicle control settings
        throttle = control.throttle  # Current throttle setting
        steer = control.steer  # Current steering setting

        vehicle_location = vehicle.get_location()  # Get vehicle location
        waypoint = world.get_map().get_waypoint(vehicle_location)  # Get waypoint for current location
        speed = get_speed(vehicle)  # Get current vehicle speed
        degrees = get_angle(vehicle, waypoint)  # Get angle to waypoint

        steer -= degrees / 180.0  # Adjust steering based on angle
        steer = np.clip(steer, -1, 1)  # Clip steering to valid range
        PREFERRED_SPEED = 0.9
        
        if speed < PREFERRED_SPEED - SPEED_THRESHOLD:  # If speed is below the preferred speed threshold
            throttle = 0.9  # Set throttle to full
        elif speed < PREFERRED_SPEED:  # If speed is below the preferred speed
            throttle = 0.5  # Set moderate throttle
        else:  # If speed is at or above the preferred speed
            throttle = 0.0  # Set throttle to minimum

        control.throttle = throttle  # Update vehicle control with new throttle
        control.steer = steer  # Update vehicle control with new steering
        vehicle.apply_control(control)  # Apply control settings to the vehicle

        display_image = np.zeros((600, 800, 3), dtype=np.uint8)  # Create an empty image for display
        for cam_data in camera_data:  # Iterate over camera data
            if 'image' in cam_data:  # If image data is available
                display_image = np.hstack((display_image, cam_data['image'][:, :, :3]))  # Concatenate images horizontally

        for model in [detection_model, custom_model]:  # Iterate over models
            input_tensor = tf.convert_to_tensor(display_image)  # Convert image to TensorFlow tensor
            input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

            detections = model(input_tensor)  # Get detections from the model

            # Convert boxes to numpy array
            boxes = np.squeeze(detections['detection_boxes'].numpy())  # Get detection boxes
            classes = np.squeeze(detections['detection_classes'].numpy()).astype(np.int32)  # Get detection classes
            scores = np.squeeze(detections['detection_scores'].numpy())  # Get detection scores

            dir_util.visualize_boxes_and_labels_on_image_array(
                display_image,
                boxes,
                classes,
                scores,
                # category_index,
                use_normalized_coordinates=True,
                line_thickness=8
            )  # Visualize detections on the image

             # Check if the number of waypoints in the local planner's queue is less than 21
        if len(agent._local_planner.waypoints_queue) < 21:
            # If it is, call the reroute method to get new waypoints
            agent.reroute(spawnpoint)
        
        # Check if the number of waypoints in the local planner's queue is exactly 0
        elif len(agent._local_planner.waypoints_queue) == 0:
            # If it is, print that the target has been reached
            print('target reached')
            # Break out of the loop since the destination has been reached
            break
    
        # Get the speed limit of the road where the vehicle is currently
        speed_limit = vehicle.get_speed_limit()
        # Set the speed limit in the local planner to match the road's speed limit
        agent._local_planner.set_speed(speed_limit)
    
        # Call the agent's run_step method to decide the next control actions
        control = agent.run_step()
        # Apply the control actions to the vehicle
        vehicle.apply_control(control)

        cv2.putText(display_image, f'Speed: {speed:.2f} km/h', org, font, fontScale, color, thickness, cv2.LINE_AA)  # Display speed on image
        cv2.imshow('Autonomous Driving', display_image)  # Show image in OpenCV window
        cv2.waitKey(1)  # Wait for a key press

finally:
    for actor in actor_list:  # Iterate over actors
        actor.destroy()  # Destroy each actor
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print('Actors destroyed.')
