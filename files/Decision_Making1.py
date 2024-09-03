#===============================================================================
#Decision_Making-for-simulation-by-Michael-Micah--------------------------------
#===============================================================================



# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import glob
import os
import sys
import math
import numpy as np
from enum import Enum
from collections import deque
import random

import time
import cv2
from datetime import datetime
import logging
import argparse
import re
import weakref
# import pygame 

# from carla import ColorConverter as cc
from agents.navigation.basic_agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.basic_agent import Agent
from agents.navigation.local_planner import LocalPlanner
from agents.tools.misc import get_speed
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import vector
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle
from manual_control import World

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass 

import carla

import networkx as nx

# ==============================================================================
# -- Local_planner --------------------------------------------------------
# ==============================================================================
class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                    opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                       args_lateral=args_lateral_dict,
                                                       args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW
        self._global_plan = True

    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)

        return control


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
 
# ==============================================================================
# -- Global_Route_planner --------------------------------------------------------
# ==============================================================================
class GlobalRoutePlanner(object):
    """
    This class provides a very high level route plan.
    Instantiate the class by passing a reference to
    A GlobalRoutePlannerDAO object.
    """

    def __init__(self, dao):
        """
        Constructor
        """
        self._dao = dao
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None

    def setup(self):
        """
        Performs initial server data lookup for detailed topology
        and builds graph representation of the world map.
        """
        self._topology = self._dao.get_topology()
        self._graph, self._id_map, self._road_id_to_edge = self._build_graph()
        self._lane_change_link()

    def _build_graph(self):
        """
        This function builds a networkx  graph representation of topology.
        The topology is read from self._topology.
        graph node properties:
            vertex   -   (x,y,z) position in world map
        graph edge properties:
            entry_vector    -   unit vector along tangent at entry point
            exit_vector     -   unit vector along tangent at exit point
            net_vector      -   unit vector of the chord from entry to exit
            intersection    -   boolean indicating if the edge belongs to an
                                intersection
        return      :   graph -> networkx graph representing the world map,
                        id_map-> mapping from (x,y,z) to node id
                        road_id_to_edge-> map from road id to edge in the graph
        """
        graph = nx.DiGraph()
        id_map = dict() # Map with structure {(x,y,z): id, ... }
        road_id_to_edge = dict() # Map with structure {road_id: {lane_id: edge, ... }, ... }

        for segment in self._topology:

            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_intersection
            road_id, lane_id = entry_wp.road_id, entry_wp.lane_id

            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in id_map:
                    new_id = len(id_map)
                    id_map[vertex] = new_id
                    graph.add_node(new_id, vertex=vertex)
            n1 = id_map[entry_xyz]
            n2 = id_map[exit_xyz]
            if road_id not in road_id_to_edge:
                road_id_to_edge[road_id] = dict()
            road_id_to_edge[road_id][lane_id] = (n1, n2)

            # Adding edge with attributes
            graph.add_edge(
                n1, n2,
                length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=vector(
                    entry_wp.transform.location,
                    path[0].transform.location if len(path) > 0 else exit_wp.transform.location),
                exit_vector=vector(
                    path[-1].transform.location if len(path) > 0 else entry_wp.transform.location,
                    exit_wp.transform.location),
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW)

        return graph, id_map, road_id_to_edge

    def _localize(self, location):
        """
        This function finds the road segment closest to given location
        location        :   carla.Location to be localized in the graph
        return          :   pair node ids representing an edge in the graph
        """
        waypoint = self._dao.get_waypoint(location)
        return self._road_id_to_edge[waypoint.road_id][waypoint.lane_id]

    def _lane_change_link(self):
        """
        This method places zero cost links in the topology graph
        representing availability of lane changes.
        """

        for segment in self._topology:
            left_found, right_found = False, False

            for waypoint in segment['path']:
                if not segment['entry'].is_intersection:
                    next_waypoint, next_road_option, next_segment = None, None, None

                    if bool(waypoint.lane_change & carla.LaneChange.Right) and not right_found:
                        next_waypoint = waypoint.get_right_lane()
                        if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and \
                            waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANERIGHT
                            try:
                                next_segment = self._localize(next_waypoint.transform.location)
                            except KeyError:
                                print("Failed to localize! : ", next_waypoint.road_id, next_waypoint.lane_id)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=segment['entry'],
                                    exit_waypoint=self._graph.edges[next_segment[0], next_segment[1]]['entry_waypoint'],
                                    path=[], length=0, type=next_road_option, change_waypoint = waypoint)
                                right_found = True

                    if bool(waypoint.lane_change & carla.LaneChange.Left) and not left_found:
                        next_waypoint = waypoint.get_left_lane()
                        if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and \
                            waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANELEFT
                            try:
                                next_segment = self._localize(next_waypoint.transform.location)
                            except KeyError:
                                print("Failed to localize! : ", next_waypoint.road_id, next_waypoint.lane_id)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=segment['entry'],
                                    exit_waypoint=self._graph.edges[next_segment[0], next_segment[1]]['entry_waypoint'],
                                    path=[], length=0, type=next_road_option, change_waypoint = waypoint)
                                left_found = True

                if left_found and right_found:
                    break

    def _distance_heuristic(self, n1, n2):
        """
        Distance heuristic calculator for path searching
        in self._graph
        """
        l1 = np.array(self._graph.nodes[n1]['vertex'])
        l2 = np.array(self._graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1-l2)

    def _path_search(self, origin, destination):
        """
        This function finds the shortest path connecting origin and destination
        using A* search with distance heuristic.
        origin      :   carla.Location object of start position
        destination :   carla.Location object of of end position
        return      :   path as list of node ids (as int) of the graph self._graph
        connecting origin and destination
        """

        start, end = self._localize(origin), self._localize(destination)

        route = nx.astar_path(
            self._graph, source=start[0], target=end[0],
            heuristic=self._distance_heuristic, weight='length')
        route.append(end[1])
        return route

    def _turn_decision(self, index, route, threshold=math.radians(5)):
        """
        This method returns the turn decision (RoadOption) for pair of edges
        around current index of route list
        """

        decision = None
        previous_node = route[index-1]
        current_node = route[index]
        next_node = route[index+1]
        next_edge = self._graph.edges[current_node, next_node]
        if index > 0:
            current_edge = self._graph.edges[previous_node, current_node]
            calculate_turn = current_edge['type'].value == RoadOption.LANEFOLLOW.value and \
                not current_edge['intersection'] and \
                    next_edge['type'].value == RoadOption.LANEFOLLOW.value and \
                        next_edge['intersection']
            if calculate_turn:
                cv, nv = current_edge['exit_vector'], next_edge['net_vector']
                cross_list = []
                for neighbor in self._graph.successors(current_node):
                    select_edge = self._graph.edges[current_node, neighbor]
                    if select_edge['type'].value == RoadOption.LANEFOLLOW.value:
                        if neighbor != route[index+1]:
                            sv = select_edge['net_vector']
                            cross_list.append(np.cross(cv, sv)[2])
                next_cross = np.cross(cv, nv)[2]
                deviation = math.acos(np.dot(cv, nv) /\
                    (np.linalg.norm(cv)*np.linalg.norm(nv)))
                if not cross_list:
                    cross_list.append(0)
                if deviation < threshold:
                    decision = RoadOption.STRAIGHT
                elif cross_list and next_cross < min(cross_list):
                    decision = RoadOption.LEFT
                elif cross_list and next_cross > max(cross_list):
                    decision = RoadOption.RIGHT
            else:
                decision = next_edge['type']
        else:
            decision = next_edge['type']

        return decision

    def abstract_route_plan(self, origin, destination):
        """
        The following function generates the route plan based on
        origin      : carla.Location object of the route's start position
        destination : carla.Location object of the route's end position
        return      : list of turn by turn navigation decisions as
        agents.navigation.local_planner.RoadOption elements
        Possible values are STRAIGHT, LEFT, RIGHT, LANEFOLLOW, VOID
        CHANGELANELEFT, CHANGELANERIGHT
        """

        route = self._path_search(origin, destination)
        plan = []

        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)
            plan.append(road_option)

        return plan

    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        min_distance = float('inf')
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index

    def trace_route(self, origin, destination):
        """
        This method returns list of (carla.Waypoint, RoadOption) from origin to destination
        """

        route_trace = []
        route = self._path_search(origin, destination)
        current_waypoint = self._dao.get_waypoint(origin)
        resolution = self._dao.get_resolution()

        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)
            edge = self._graph.edges[route[i], route[i+1]]
            path = []

            if edge['type'].value != RoadOption.LANEFOLLOW.value and edge['type'].value != RoadOption.VOID.value:
                n1, n2 = self._road_id_to_edge[edge['exit_waypoint'].road_id][edge['exit_waypoint'].lane_id]
                next_edge = self._graph.edges[n1, n2]
                if next_edge['path']:
                    closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    closest_index = min(len(next_edge['path'])-1, closest_index+5)
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
                    if len(route)-i <= 2 and waypoint.transform.location.distance(destination) < 2*resolution:
                        break

        return route_trace

# ==============================================================================
# -- Global_Route_planner_Dao --------------------------------------------------------
# ==============================================================================
class GlobalRoutePlannerDAO(object):
    """
    This class is the data access layer for fetching data
    from the carla server instance for GlobalRoutePlanner
    """

    def __init__(self, wmap, sampling_resolution=1):
        """get_topology
        Constructor

        wmap    :   carl world map object
        """
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap

    def get_topology(self):
        """
        Accessor for topology.
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects.

        return: list of dictionary objects with the following attributes
                entry   -   waypoint of entry point of road segment
                entryxyz-   (x,y,z) of entry point of road segment
                exit    -   waypoint of exit point of road segment
                exitxyz -   (x,y,z) of exit point of road segment
                path    -   list of waypoints separated by 1m from entry
                            to exit
        """
        topology = []
        # Retrieving waypoints to construct a detailed topology
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                seg_dict['path'].append(wp1.next(self._sampling_resolution/2.0)[0])
            topology.append(seg_dict)
        return topology

    def get_waypoint(self, location):
        """
        The method returns waypoint at given location
        """
        waypoint = self._wmap.get_waypoint(location)
        return waypoint

    def get_resolution(self):
        """ Accessor for self._sampling_resolution """
        return self._sampling_resolution

# ==============================================================================
# -- Misc --------------------------------------------------------
# ==============================================================================
def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2
    location_1, location_2:   carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

# ==============================================================================
# -- Agents --------------------------------------------------------
# ==============================================================================

class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3


class Agent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = vehicle
        self._proximity_threshold = 10.0  # meters
        self._local_planner = None
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self._last_traffic_light = None
        self._world = World
        self._local_planner = LocalPlanner(self)
        self._map = self._world.get_map()

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()

        if debug:
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

        return control

    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        if self._map.name == 'Town01' or self._map.name == 'Town02':
            return self._is_light_red_europe_style(lights_list)
        else:
            return self._is_light_red_us_style(lights_list)

    def _is_light_red_europe_style(self, lights_list):
        """
        This method is specialized to check European style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                  affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_waypoint = self._map.get_waypoint(traffic_light.get_location())
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = traffic_light.get_location()
            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)

        return (False, None)

    def _is_light_red_us_style(self, lights_list, debug=False):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_intersection:
            # It is too late. Do not block the intersection! Keep going!
            return (False, None)

        if self._local_planner.target_waypoint is not None:
            if self._local_planner.target_waypoint.is_intersection:
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 60.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if debug:
                        print('=== Magnitude = {} | Angle = {} | ID = {}'.format(
                            sel_magnitude, min_angle, sel_traffic_light.id))

                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.TrafficLightState.Red:
                        return (True, self._last_traffic_light)
                else:
                    self._last_traffic_light = None

        return (False, None)

    def _is_vehicle_hazard(self, vehicle_list):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = target_vehicle.get_location()
            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold):
                return (True, target_vehicle)

        return (False, None)

    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle
        :return:
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control

# ==============================================================================
# -- Basic_Agents --------------------------------------------------------
# ==============================================================================
class BasicAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(BasicAgent, self).__init__(vehicle)

        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed' : target_speed,
            'lateral_control_dict':args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None

    def set_destination(self, location):
        """
        This method creates a list of waypoints from the agent's position to the destination location
        based on the route returned by the global router.
        """

        # Get the starting waypoint from the current location of the vehicle
        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        
        # Creating a CARLA Location object for the destination and get the waypoint for the destination location
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2])
        )

        # Generate the route trace from the start waypoint to the end waypoint
        route_trace = self._trace_route(start_waypoint, end_waypoint)
        
        # Ensure that the route trace is valid
        assert route_trace

        # Set the global plan in the local planner to the generated route trace
        self._local_planner.set_global_plan(route_trace)


    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint.
        """
        # Setting up global router
        # Check if the global route planner (_grp) has been initialized
        if self._grp is None:
            # Create a data access object (DAO) for the global route planner
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            
            # Initialize the global route planner with the DAO
            grp = GlobalRoutePlanner(dao)
            
            # Setup the global route planner
            grp.setup()
            
            # Store the global route planner instance in the class variable _grp
            self._grp = grp
        # Obtain route plan
        # Use the global route planner to trace the route from the start location to the end location
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location
        )
        # Return the calculated route
        return route


    # def run_step(self, debug=False):
    #     """
    #     Execute one step of navigation.
    #     :return: carla.VehicleControl
    #     """

    #     # is there an obstacle in front of us?
    #     hazard_detected = False

    #     # retrieve relevant elements for safe navigation, i.e.: traffic lights
    #     # and other vehicles
    #     actor_list = self._world.get_actors()
    #     vehicle_list = actor_list.filter("*vehicle*")
    #     lights_list = actor_list.filter("*traffic_light*")

    #     # check possible obstacles
    #     vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
    #     if vehicle_state:
    #         if debug:
    #             print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

    #         self._state = AgentState.BLOCKED_BY_VEHICLE
    #         hazard_detected = True

    #     # check for the state of the traffic lights
    #     light_state, traffic_light = self._is_light_red(lights_list)
    #     if light_state:
    #         if debug:
    #             print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

    #         self._state = AgentState.BLOCKED_RED_LIGHT
    #         hazard_detected = True

    #     if hazard_detected:
    #         control = self.emergency_stop()
    #     else:
    #         self._state = AgentState.NAVIGATING
    #         # standard local planner behavior
    #         control = self._local_planner.run_step()

    #     return control

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        # Initialize a flag to indicate if a hazard is detected
        hazard_detected = False

        # Retrieve relevant elements for safe navigation, i.e., traffic lights, vehicles, pedestrians, traffic signboards, and other objects
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        pedestrian_list = actor_list.filter("*walker.pedestrian*")
        traffic_sign_list = actor_list.filter("*traffic.sign*")
        other_objects_list = actor_list.filter("*static.prop*")

        # Check for possible obstacles: other vehicles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            # If in debug mode, print a message indicating a vehicle is blocking the way
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            # Update the state to indicate being blocked by a vehicle
            self._state = AgentState.BLOCKED_BY_VEHICLE
            
            # Set the hazard detected flag to True
            hazard_detected = True

        # Check for pedestrians in the path
        pedestrian_state, pedestrian = self._is_pedestrian_hazard(pedestrian_list)
        if pedestrian_state:
            # If in debug mode, print a message indicating a pedestrian is blocking the way
            if debug:
                print('!!! PEDESTRIAN BLOCKING AHEAD [{}])'.format(pedestrian.id))

            # Update the state to indicate being blocked by a pedestrian
            self._state = AgentState.BLOCKED_BY_PEDESTRIAN
            
            # Set the hazard detected flag to True
            hazard_detected = True

        # Check the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            # If in debug mode, print a message indicating a red light ahead
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            # Update the state to indicate being blocked by a red light
            self._state = AgentState.BLOCKED_RED_LIGHT
            
            # Set the hazard detected flag to True
            hazard_detected = True

        # Check for traffic signboards
        sign_state, traffic_sign = self._is_traffic_sign_hazard(traffic_sign_list)
        if sign_state:
            # If in debug mode, print a message indicating a traffic sign hazard
            if debug:
                print('*** TRAFFIC SIGN HAZARD [{}])'.format(traffic_sign.id))

            # Update the state to indicate being affected by a traffic sign
            self._state = AgentState.BLOCKED_TRAFFIC_SIGN
            
            # Set the hazard detected flag to True
            hazard_detected = True

        # Check for other static objects on the road
        object_state, road_object = self._is_object_hazard(other_objects_list)
        if object_state:
            # If in debug mode, print a message indicating a road object hazard
            if debug:
                print('@@@ OBJECT BLOCKING AHEAD [{}])'.format(road_object.id))

            # Update the state to indicate being blocked by a road object
            self._state = AgentState.BLOCKED_OBJECT
            
            # Set the hazard detected flag to True
            hazard_detected = True

        # Check for lane markings and lane detection
        lane_state = self._is_lane_hazard()
        if lane_state:
            # If in debug mode, print a message indicating a lane marking hazard
            if debug:
                print('### LANE MARKING HAZARD')

            # Update the state to indicate being affected by lane markings
            self._state = AgentState.BLOCKED_LANE_MARKING
            
            # Set the hazard detected flag to True
            hazard_detected = True

        # If any hazard is detected, perform an emergency stop
        if hazard_detected:
            control = self.emergency_stop()
        else:
            # If no hazard is detected, update the state to navigating
            self._state = AgentState.NAVIGATING
            
            # Execute the standard local planner behavior
            control = self._local_planner.run_step()

        # Return the control commands for the vehicle
        return control

    
# ==============================================================================
# -- Controller --------------------------------------------------------
# ==============================================================================
class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral=None, args_longitudinal=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        if not args_lateral:
            args_lateral = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}
        if not args_longitudinal:
            args_longitudinal = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        throttle = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step(waypoint)

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

        :param target_speed: target speed in Km/h
        :return: throttle control in the range [0, 1]
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), 0.0, 1.0)


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x -
                          v_begin.x, waypoint.transform.location.y -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
        
    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# Adjustable image dimensions
IM_WIDTH = 640 
IM_HEIGHT = 480

# # Define camera transformations for eight cameras on Tesla model3
# camera_transformations = [
#     carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0)),       # Front center cam 0
#     carla.Transform(carla.Location(x=2.0, y=-0.5, z=1.4), carla.Rotation(yaw=-45)),  # Front left cam 1
#     carla.Transform(carla.Location(x=2.0, y=0.5, z=1.4), carla.Rotation(yaw=45)),   # Front right cam 2
#     carla.Transform(carla.Location(x=-1.0, z=1.4), carla.Rotation(yaw=180)),    # Rear center cam 3
#     carla.Transform(carla.Location(x=-1.0, y=0.75, z=1.4), carla.Rotation(yaw=135)),  # Rear left cam 4
#     carla.Transform(carla.Location(x=-1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-135)),  # Rear right cam 5
#     carla.Transform(carla.Location(x=1.0, y=0.75, z=1.4), carla.Rotation(yaw=90)),  # Right B-pillar cam 6
#     carla.Transform(carla.Location(x=1.0, y=-0.75, z=1.4), carla.Rotation(yaw=-90))  # Left side cam 7
# ]

# Define camera transformations for eight cameras on a Cybertruck
camera_transformations = [
    carla.Transform(carla.Location(x=2.5, z=2.0), carla.Rotation(yaw=0)),       # Front center cam 0
    carla.Transform(carla.Location(x=2.5, y=-0.75, z=2.0), carla.Rotation(yaw=-45)),  # Front left cam 1
    carla.Transform(carla.Location(x=2.5, y=0.75, z=2.0), carla.Rotation(yaw=45)),   # Front right cam 2
    carla.Transform(carla.Location(x=-2.5, z=2.0), carla.Rotation(yaw=180)),    # Rear center cam 3
    carla.Transform(carla.Location(x=-2.5, y=1.0, z=2.0), carla.Rotation(yaw=135)),  # Rear left cam 4
    carla.Transform(carla.Location(x=-2.5, y=-1.0, z=2.0), carla.Rotation(yaw=-135)),  # Rear right cam 5
    carla.Transform(carla.Location(x=0.5, y=1.0, z=2.0), carla.Rotation(yaw=90)),  # Right B-pillar cam 6
    carla.Transform(carla.Location(x=0.5, y=-1.0, z=2.0), carla.Rotation(yaw=-90))  # Left side cam 7
]


def process_rgb_img(image, timestamp, idx):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3] 

    cv2.imshow(f"RGB Camera {idx}", i3)
    cv2.waitKey(1) & 0xFF
    return i3 / 255.0

def draw_route(wp, route,seconds=3.0):
    #draw the next few points route in sim window - Note it does not
    # get into the camera of the car
    if len(route)-wp <25: # route within 25 points from end is red
        draw_colour = carla.Color(r=255, g=0, b=0)
    else:
        draw_colour = carla.Color(r=0, g=0, b=255)
    for i in range(10):
        if wp+i<len(route)-2:
            World.debug.draw_string(route[wp+i][0].transform.location, '^', draw_shadow=False,
                color=draw_colour, life_time=seconds,
                persistent_lines=True)
    return None

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

        # Choose a suitable model
        bp = blueprint_library.filter("cybertruck")[0]

        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        
        # # Oh wait, I don't like the location we gave to the vehicle, I'm going
        # # to move it a bit forward.
        # location = vehicle.get_location()
        # location.x += 70
        # vehicle.set_location(location)
        # print('moved vehicle to %s' % location)
        
        vehicle.set_autopilot(True)
        actor_list.append(vehicle)

        # Configure RGB camera blueprint
        rgb_cam_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        rgb_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        rgb_cam_bp.set_attribute("fov", "90")

        # Spawn and attach cameras
        for idx, transform in enumerate(camera_transformations):
            rgb_camera = world.spawn_actor(rgb_cam_bp, transform, attach_to=vehicle)
            actor_list.append(rgb_camera)

            rgb_camera.listen(lambda image, idx=idx: process_rgb_img(image, datetime.now().strftime('%Y%m%d%H%M%S%f'), idx))

        time.sleep(3000)

    finally:
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

if __name__ == '__main__':
    main()
