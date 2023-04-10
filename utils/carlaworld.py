from __future__ import print_function
import numpy as np
import os
import time
import math
import pygame
from utils.tools import get_folder_name
import signal
import subprocess
import config as cfg
import carla
import logging

class TrafficLight:
    COLOR_BACKGROUND = pygame.Color(46, 52, 54)
    COLOR_OFF = pygame.Color(85, 87, 83)
    COLOR_RED = pygame.Color(239, 41, 41)
    COLOR_YELLOW = pygame.Color(252, 233, 79)
    COLOR_GREEN = pygame.Color(138, 226, 52)

    def __init__(self, width):
        self.width = width
        self.height = 3 * self.width
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.surface.fill(TrafficLight.COLOR_BACKGROUND)
        self.state = 'off'

    def change_state(self, state):
        if state != self.state:
            self.state = state

            hw = int(self.width / 2)
            radius = int(0.4 * self.width)

            pygame.draw.circle(self.surface, TrafficLight.COLOR_RED if state == 'Red' else TrafficLight.COLOR_OFF,
                               (hw, hw), radius)

            pygame.draw.circle(self.surface, TrafficLight.COLOR_YELLOW if state == 'Yellow' else TrafficLight.COLOR_OFF,
                               (hw, self.width + hw), radius)

            pygame.draw.circle(self.surface, TrafficLight.COLOR_GREEN if state == 'Green' else TrafficLight.COLOR_OFF,
                               (hw, 2 * self.width + hw), radius)


class TrafficLightsDisplay:
    def __init__(self, carla_world):
        self.world = carla_world
        self.open_drive_map = carla_world.get_map()
        self.font = pygame.font.SysFont(None, 24) # Textsize can be changed here
        self.traffic_light = TrafficLight(width=40)
        self.landmarks = None
        self.start = None
        self.max_tl_dist = 25

    def fetch_tl_landmarks(self, location, distance=50):
        waypoint = self.open_drive_map.get_waypoint(location, lane_type=carla.LaneType.Driving)
        self.landmarks = None

        # Do not check inside a junction
        if not waypoint.is_junction:
            # Traffic Light: '1000001'
            self.landmarks = waypoint.get_landmarks_of_type(distance=distance, type='1000001', stop_at_junction=False)

            # Is this needed? Are landmarks sorted already?
            if self.landmarks is not None:
                self.landmarks.sort(key=lambda x: x.distance)

    def render(self, display, pos_xy):
        if self.landmarks is not None:
            for landmark in self.landmarks:
                traffic_light = self.world.get_traffic_light(landmark)

                if isinstance(traffic_light, carla.TrafficLight):
                    self.traffic_light.change_state(str(traffic_light.get_state()))
                    display.blit(self.traffic_light.surface, pos_xy)

                    #Dropping shadow for readability
                    speed_text = self.font.render(f'{round(landmark.distance)} m', True, (80, 80, 80))
                    display.blit(speed_text, (pos_xy[0] + 7, pos_xy[1] + self.traffic_light.height + 12))

                    dist_text = self.font.render(f'{round(landmark.distance)} m', True, (200, 200, 200))
                    display.blit(dist_text, (pos_xy[0] + 5, pos_xy[1] + self.traffic_light.height + 10))
                    return  # Draw only the first traffic light

    def change_tl_state(self):
        if self.landmarks is not None:
            for landmark in self.landmarks:
                traffic_light = self.world.get_traffic_light(landmark)

                if isinstance(traffic_light, carla.TrafficLight) and landmark.distance < self.max_tl_dist: # only change lights max_tl_dist meters in front of us
                    if traffic_light.get_state()==carla.TrafficLightState.Red: # if we are waiting in front of a red light
                        if self.start is None: # set the timer
                            self.start = time.time()
                        elif time.time() - self.start > 10.0:
                            for tl in traffic_light.get_group_traffic_lights(): # switch all lights in group to red
                                tl.set_state(carla.TrafficLightState.Red)
                            traffic_light.set_state(carla.TrafficLightState.Green) # switch light in front of us to green
                            self.start = None # reset timer after changing light state
                        
                else: self.start = None # reset timer if not in front of traffic light
                return # change only first traffic light in list

class SpeedDisplay:
    def __init__(self):
        # Textsize can be changed here
        self.font = pygame.font.SysFont(None, 24)

    def render(self, actor, display, pos_xy):
        if isinstance(actor, carla.Actor):
            v = actor.get_velocity()
            speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

            # Dropping shadow for readability
            speed_text = self.font.render(f'{round(speed)} km/h', True, pygame.Color(200, 200, 200))
            display.blit(speed_text, (pos_xy[0]-1, pos_xy[1]-1))

            speed_text = self.font.render(f'{round(speed)} km/h', True, pygame.Color(80, 80, 80))
            display.blit(speed_text, pos_xy)


class TravelDistance():
    def __init__(self, current_loc):
        self.prev_loc = current_loc
        self.travelled_distance = 0
        
    def next(self, current_loc):        
        self.travelled_distance += np.sqrt((current_loc.x - self.prev_loc.x)**2 + (current_loc.y - self.prev_loc.y)**2 + (current_loc.z - self.prev_loc.z)**2)
        self.prev_loc = current_loc
        #return self.travelled_distance

    def reset(self):
        self.travelled_distance = 0
        return self.travelled_distance

def count_vehicles_and_walkers(world):
    world_snapshot = world.get_snapshot()
    cnter_vehicles = 0
    cnter_walkers = 0
    for actor_snapshot in world_snapshot:
        actor = world.get_actor(actor_snapshot.id)
        if str(actor.type_id).startswith('vehicle'):
            cnter_vehicles +=1
        if str(actor.type_id).startswith('walker'):
            cnter_walkers +=1
    return cnter_vehicles, cnter_walkers

def get_ego_car(actors):
    """
    get the ego car
    :param actors: list of actors
    """
    for actor in actors:
        if str(actor.type_id).startswith('vehicle'):
            if not actor.attributes:
                pass
            else:
                if actor.attributes['role_name'] == 'hero':
                    ego_car = actor
                    break
    return ego_car


def get_ego_position(player_pos):
    """
    get the ego vehicle position
    output: array (x,y,z)
    """
    ### get the position
    pos_tmp = player_pos.get_transform()
    ego_pos = np.zeros(3)
    for i in range(3):
        ego_pos[i] = pos_tmp.get_matrix()[i][3]
    
    print('ego position: ', ego_pos)

def get_ego_id():
    """
    reads the ego vehicle id from a text file
    """
    with open(os.path.join(get_folder_name(), '00_log', 'ego_id.txt')) as f:
            lines = f.readlines()
            ego_id = int(lines[0])
    return ego_id


def ignore_blueprints(blueprints):
    """
    ignore specific blueprints like 2 wheeled vehicles, teslas cybertruck etc.
    :param blueprints: list of vehicle blueprints
    """
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    return blueprints


def kill_carla_world(carla_proc, processes=None, sensor_list=None):
    """
    kills the CARLA world
    needs a second kill, because CARLA was still alive
    processes is a list of different processes, which has started and need to be killed
    """
    if sensor_list is not None:
        for sensor in sensor_list:
            sensor.destroy()
    if processes is not None:
        for process in processes:
            os.killpg(process.pid, signal.SIGTERM)        
    pid = os.getpgid(carla_proc.pid) # get the PID
    os.killpg(pid, signal.SIGTERM)
    # outs, errs = carla_proc.communicate()
    time.sleep(2)
    os.killpg(pid, signal.SIGTERM)
    time.sleep(2)


def restart_carla_world(carla_proc, processes=None, sensor_list=None, town=None):
    """
    kills the actual CARLA world and starts a new one
    :param carla_proc:  the actual CARLA world subprocess
    :param sensor_list: list of all active sensors
    :return carla_proc: new CARLA world subprocess
    :return client:     client
    :return world:      world
    """
    kill_carla_world(carla_proc, processes, sensor_list)
    ### start carla world
    carla_proc = subprocess.Popen(['./CarlaUE4.sh', '-RenderOffScreen'], cwd=cfg.path_carla, preexec_fn=os.setsid) 
    time.sleep(7)

    client = carla.Client('localhost', 2000, worker_threads=1)
    client.set_timeout(60.0)
    world = client.get_world()
    return carla_proc, client, world

def spawning_area(world, radius):
    ego_pos = world.get_ego_location()
    x_ego, y_ego = ego_pos.x, ego_pos.y
    x_min = x_ego - radius; x_max = x_ego + radius
    y_min = y_ego - radius; y_max = y_ego + radius
    return [x_min, x_max, y_min, y_max]

def remove_fences(world):
    if cfg.carla_version > 10: 
        logging.info("Removing fences...")
        fences = world.world.get_environment_objects(carla.CityObjectLabel.Fences)
        world.world.enable_environment_objects([*map(lambda obj: obj.id, fences)], False)