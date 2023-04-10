#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# This script builds on the manual_control_steeringwheel.py of 
# the official CARLA 0.9.10 repository: 
# https://github.com/carla-simulator/carla/blob/0.9.10/PythonAPI/examples/manual_control_steeringwheel.py

# Allows controlling a vehicle with one or two steering wheels 
# and with a keyboard. 
# This setup allows two modes:
# 1) generating corner cases with two steering wheels and two 
# human drivers 
# 2) demonstrator for visualisation of an output of a semantic
# segmentation network
"""
Welcome to A-Eye: Driving with the eyes of AI. 
"""

from __future__ import print_function

import config as cfg
import carla

from carla import ColorConverter as cc
import os
import sys
import argparse

import logging
import math
import random
import re
import weakref
# ==============================================================================
# -- additional imports --------------------------------------------------------
# ==============================================================================
import time
import torch

from utils.cc import CheckCornerCase
from utils.weather import Weather
from utils.inference import Inference
from utils.tools import TimeMeasurement, get_folder_name, get_model_name 
from utils.tools import output_folders_data_generator as output_folders
from utils.tracking import pedal_tracking
from utils.rec import QRecording
from utils.save import save_csv
from utils.carlaworld import TravelDistance, SpeedDisplay, TrafficLightsDisplay
from utils.carlaworld import get_ego_car, remove_fences
# from utils.spawn import spawning_radius
# from utils.carla_classes import find_weather_presets, GnssSensor, HUD
# ==============================================================================
# -- GPU settings  --------------------------------------------------------
# ==============================================================================
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
if len(available_gpus) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ==============================================================================
# -- parser and pygame  --------------------------------------------------------
# ==============================================================================

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL, KMOD_SHIFT, K_BACKQUOTE, K_BACKSPACE
    from pygame.locals import K_COMMA, K_ESCAPE, K_SPACE, K_TAB, K_PERIOD
    from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN
    from pygame.locals import K_0, K_9
    from pygame.locals import K_a, K_c, K_d, K_m, K_p, K_q, K_r, K_s, K_w
    from pygame.locals import K_F1, K_F2, K_F5, K_F6, K_F7, K_F8
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        height_fadingtext = height
        self._notifications = FadingText(font, (width-900, 40), (0, height - height_fadingtext))
        # self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = False # change here when HUD should start at the beginning
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        #####
        if cfg.live_plotter:
            self.size_diag = 100
            self.x_time = np.linspace(0,1,self.size_diag+1)[0:-1]
            self.y_throttle = np.zeros(len(self.x_time))
            self.y_brake = np.zeros(len(self.x_time))
            self.y_steer = np.zeros(len(self.x_time))
            self.line1 = []
            self.line2 = []
            self.line3 = []

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        
        # ------------------------------------------------------------------------
        # throttle, brake and steeringwheel tracking added
        # ------------------------------------------------------------------------
        if cfg.live_plotter: #TODO: create class
            self.y_throttle[-1] = c.throttle
            self.y_brake[-1] = c.brake
            self.y_steer[-1] = c.steer
            
            self.line1, self.line2, self.line3 = live_plotter(self.x_time,
                                self.y_throttle,self.line1,
                                self.y_brake,self.line2,
                                (self.y_steer)*229.241,self.line3) 
                                # output of steeringwheel = [-1.963455, 1.963129] 
                                # and steering range = 900°
                                # --> factor ca. 229.241

            self.y_throttle = np.append(self.y_throttle[1:],0.0)
            self.y_brake= np.append(self.y_brake[1:],0.0)
            self.y_steer = np.append(self.y_steer[1:],0.0)
        # ------------------------------------------------------------------------
        # 
        # ------------------------------------------------------------------------

        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        vehicles = world.world.get_actors().filter('vehicle.*')

        # for vehicle in vehicles:
        #     if not vehicle.attributes:
        #         pass
        #     else:
        #         if vehicle.attributes['role_name'] == 'hero':
        #             ego_car = vehicle
        #             break
        
        # print('Actor ID: ', ego_car.id)
        # print('Location ', ego_car.get_location())
        # print('velo [m/s]: ', ego_car.get_velocity())
        # print('transform: ', ego_car.get_transform())

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
#            'Map:     % 20s' % world.world.map_name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            # 'Collision:',
            # collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        # self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=5.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


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


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter, path, model_name, qrecording, args):
        self.world = carla_world
        self.hud = hud
        self.player = None
        # self.collision_sensor = False #None
        # self.lane_invasion_sensor = False #None
        self.gnss_sensor = False #None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.args = args
        self.qrecording = qrecording # QueueRecording object
        self.path = path 
        self.model_name = model_name
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'): blueprint.set_attribute('color', '186,0,0') #set red
            
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_points[0]) # same place spawning
            time.sleep(5.0)     # wait to get the correct location
            print('PLAYER: ', self.player)
            if not self.args.demonstration:
                with open(os.path.join(self.path, '00_log', 'ego_id.txt'), 'a') as file:
                    file.write('{}\n'.format(self.player.id))
                    file.write('{}\n'.format(self.player.type_id))
                    file.write('Location: {}\n'.format(self.player.get_location()))
                    file.write('transform: {}\n'.format(self.player.get_transform()))
                # file.write('velo [m/s]: {}\n'.format(self.player.get_velocity()))
                # file.write('acc [m/s^2]: {}\n'.format(self.player.get_acceleration()))
            print('ID: {}\n'.format(self.player.id))
            print('Location: {}\n'.format(self.player.get_location()))
        # Set up the sensors.
        # self.collision_sensor = CollisionSensor(self.player, self.hud)
        # self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self.model_name, self.qrecording)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])
        

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        # get_ego_position(self.player)               # get ego position of the vehicle
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            # self.collision_sensor.sensor,
            # self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
    
    def get_ego_location(self):
        return self.player.get_location()
    
# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    """
    first:      plug in steering wheel for safety driver
    second:     plug in steering wheel for semantic driver
    """
    def __init__(self, world, start_in_autopilot, client, ccc, start, weather):
        self._previous_steer_safety_driver = 0 
        self._autopilot_enabled = start_in_autopilot
        self.i_rec = 1 # start with 1
        self.client = client
        self.use_keyboard = False
        self.counter_cc = 0  # counter for corner cases, if not set, the situation would stop directly (needed for steer)
        self.timer=start
        self.weather = weather

        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0

        #check corner case
        self.ccc = ccc

        # initialize steering wheel
        pygame.joystick.init()
        self.joystick_count = pygame.joystick.get_count()
        if self.joystick_count < 1:
            self.use_steeringwheel = False
        else:
            self.use_steeringwheel = True
        
        if self.use_steeringwheel:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            if self.joystick_count > 1:
                self._joystick_safety_driver = pygame.joystick.Joystick(1)
                self._joystick_safety_driver.init()

            self._parser = ConfigParser()
            if cfg.os_system == 'Linux':
                self._parser.read('./wheel_config.ini')
            else:
                self._parser.read('C:\\UnrealEngine\\carla\\wheel_config.ini')
            self._steer_idx = int(
                self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            self._throttle_idx = int(
                self._parser.get('G29 Racing Wheel', 'throttle'))
            self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            self._reverse_idx_left = int(self._parser.get('G29 Racing Wheel', 'reverse_left'))
            self._reverse_idx_right = int(self._parser.get('G29 Racing Wheel', 'reverse_right'))

            
    def parse_events(self, world, clock, travel_distance):
        self.get_inf_name = world.camera_manager.get_inf_name
        self.ego_car_loc = travel_distance.travelled_distance

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            ############ event buttons for G29 steering wheel ###############
            elif event.type == pygame.JOYBUTTONDOWN:
                # if event.button == 0:
                #     world.restart()
                
                if event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx_left:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == self._reverse_idx_right:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()
                elif event.button == 6:                 # R2 --> ground truth
                    world.camera_manager.set_sensor(6)
                elif event.button == 10:                # R3 --> inference 2
                    world.camera_manager.set_sensor(2)
                elif event.button == 7:                 # L2 --> original
                    world.camera_manager.set_sensor(0)
                elif event.button == 11:                # L3 --> inference 1
                    world.camera_manager.set_sensor(1)

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_F2:
                    self.timer = self.ccc.gui('brake', self.timer, self.get_inf_name, self.ego_car_loc)
                # elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                #     world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_F5:
                    self.weather.set_weather('clear')
                    world.hud.notification('Weather: clear')
                elif event.key == K_F6:
                    self.weather.set_weather('rain')
                    world.hud.notification('Weather: rain')
                elif event.key == K_F7:
                    self.weather.set_weather('fog')
                    world.hud.notification('Weather: fog')
                elif event.key == K_F8:
                    self.weather.set_weather('night')
                    world.hud.notification('Weather: night')
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl): 
            # it is not possible to use both keyboard AND steering wheel at the same time
            # --> use a bool query !!!
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                if self.use_steeringwheel and not self.use_keyboard:
                    self._parse_vehicle_wheel(self._previous_steer_safety_driver)
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)


    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        # self._control_2.steer = round(self._steer_cache, 1)
        # self._control_2.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

        if keys[K_w] or keys[K_s] or keys[K_a] or keys[K_d]:
            self.use_keyboard = True
        else:
            self.use_keyboard = False

    def _parse_vehicle_wheel(self, previous_steer_safety_driver):
        # if previous_steer_safety_driver == 0:
        #     self._previous_steer_safety_driver = 0

        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]
        if self.joystick_count > 1:
            numAxes_safety_driver = self._joystick_safety_driver.get_numaxes()
            jsInputs_safety_driver = [float(self._joystick_safety_driver.get_axis(i)) for i in range(numAxes_safety_driver)]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 0.55  # standard: 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])
        if self.joystick_count > 1:
            steerCmd_safety_driver = K1 * math.tan(1.1 * jsInputs_safety_driver[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if self.joystick_count > 1:
            throttleCmd_safety_driver = K2 + (2.05 * math.log10(
            -0.7 * jsInputs_safety_driver[self._throttle_idx] + 1.4) - 1.2) / 0.92

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if self.joystick_count > 1:
            brakeCmd_safety_driver = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs_safety_driver[self._brake_idx] + 1.4) - 1.2) / 0.92
        
        ###############
        # retrieve throttle/brake for sem/safety driver
        pedal_tracking(throttleCmd, throttleCmd_safety_driver, brakeCmd, brakeCmd_safety_driver, time.time()-self.timer) 
        ###############
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1
        if self.joystick_count > 1:
            if throttleCmd_safety_driver <= 0:
                throttleCmd_safety_driver = 0
            elif throttleCmd_safety_driver > 1:
                throttleCmd_safety_driver = 1
        
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1
        if self.joystick_count > 1:
            if brakeCmd_safety_driver <= 0:
                brakeCmd_safety_driver = 0
            elif brakeCmd_safety_driver > 1:
                brakeCmd_safety_driver = 1
        
        if self.joystick_count > 1:
            #-------- steer --------
            if abs(self._previous_steer_safety_driver - steerCmd_safety_driver)>0:
                self._control.steer = steerCmd_safety_driver
                self._previous_steer_safety_driver = steerCmd_safety_driver
                # print('steer intervention by safety_driver')
                if self.counter_cc > 10: # prevent small movements from disturbing the ride
                     self.timer = self.ccc.gui('steer', self.timer, self.get_inf_name, self.ego_car_loc)
                else: 
                    self.counter_cc +=1
                
            else: 
                self._control.steer = steerCmd
                self.counter_cc = 0     # set counter to zero
            
            #-------- brake --------
            if brakeCmd_safety_driver > 0 and brakeCmd >0: 
                #when both step on the brake, the safety driver has priority
                self._control.brake = brakeCmd_safety_driver
                self.timer =self.ccc.gui('brake',  self.timer, self.get_inf_name, self.ego_car_loc)
                # print('brake intervention by safety_driver')
            elif brakeCmd_safety_driver > 0:
                self._control.brake = brakeCmd_safety_driver
                self.timer =self.ccc.gui('brake', self.timer, self.get_inf_name, self.ego_car_loc)
                # print('brake intervention by safety_driver')
            else:
                self._control.brake = brakeCmd
               
            #-------- throttle --------
            if throttleCmd_safety_driver> 0 and brakeCmd >0: 
                #when sd steps on throttle and semseg_driver steps on brake, the safety driver has priority
                self._control.throttle = throttleCmd_safety_driver
                self._control.brake = brakeCmd_safety_driver
                # print('throttle safety_driver, brake semantic driver')
            elif throttleCmd_safety_driver > 0:
                self._control.throttle = throttleCmd_safety_driver
                # print('throttle intervention by safety_driver')
            else:
                self._control.throttle = throttleCmd
        else: # when just one steering wheel is plugged in
            self._control.steer = steerCmd 
            self._control.brake = brakeCmd
            self._control.throttle = throttleCmd
        
        #toggle = jsButtons[self._reverse_idx]
    ########################### SNES Gamepad ###############################
    def _parse_vehicle_snes(self, milliseconds):    
        numAxes = self._joystick.get_numaxes()
        jsInputs = [ float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButtons = [float(self._joystick.get_button(i)) for i in
                    range(self._joystick.get_numbuttons())]
                
        if jsButtons[1] > 0:
            self._control.brake = 1
        if jsButtons[2] > 0:
            self._control.throttle = 1
        
        # define the turning radius
        steer_increment = 3e-2 * milliseconds
        if jsInputs[0] < 0:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif jsInputs[0] > 0:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        
        ### get button outputs 
        #print('Js inputs [%s]' % ', '.join(map(str, jsInputs)))
        #print('Button inputs [%s]' % ', '.join(map(str, jsButtons)))

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)







# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, model_name, qrecording): # newly added qrecording
    # def __init__(self, parent_actor, hud, model_name):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.model_name = model_name
        self.recording = False
        self.get_inf_name = None
        self.qrecording = qrecording
        self.time_measurement = TimeMeasurement()

        self.inf_1 = Inference(cfg.ckpt_1)
        self.inf_2 = Inference(cfg.ckpt_2)
        self.inf_3 = Inference(cfg.ckpt_3)
        self.inf_4 = Inference(cfg.ckpt_4)
        self.model_name_1 = get_model_name(cfg.ckpt_1)
        self.model_name_2 = get_model_name(cfg.ckpt_2)
        self.model_name_3 = get_model_name(cfg.ckpt_3)
        self.model_name_4 = get_model_name(cfg.ckpt_4)
        
        if hud.dim[0] == 3840 or hud.dim[0] == 7680:              # move camera because of new resolution 
                self._camera_transforms = [
                    carla.Transform(carla.Location(x=-12.5, z=2.8), carla.Rotation(pitch=-5)),
                    carla.Transform(carla.Location(x=1.6, z=1.7))]
        else:	# original setup
            self._camera_transforms = [
                carla.Transform(carla.Location(x=1.6, z=1.7)),
                carla.Transform(carla.Location(x=0.1, z=1.4)),  # bonnet
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                carla.Transform(carla.Location(x=0.0, z=30.0), carla.Rotation(pitch=-90))
                ]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ############################ add inference #############################
            ['sensor.camera.rgb', cc.Raw, f'{self.model_name_1}'],
            ['sensor.camera.rgb', cc.Raw, f'{self.model_name_2}'],
            ['sensor.camera.rgb', cc.Raw, f'{self.model_name_3}'],
            ['sensor.camera.rgb', cc.Raw, f'{self.model_name_4}'],
            ########################################################################
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'ground truth'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
                bp.set_attribute('points_per_second', '1300000')
                bp.set_attribute('channels', '64')
                bp.set_attribute('rotation_frequency', '20')
                bp.set_attribute('upper_fov', '3.0')
                bp.set_attribute('lower_fov', '-25.0')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            if self.sensors[index][0].startswith('sensor.lidar'):
                self.sensor = self._parent.get_world().spawn_actor(
                    self.sensors[index][-1],
                    self._camera_transforms[self.transform_index+1],    # camera position changed for LIDAR
                    attach_to=self._parent)
            else:
                self.sensor = self._parent.get_world().spawn_actor(
                    self.sensors[index][-1],
                    self._camera_transforms[self.transform_index],
                    attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        self.time_measurement.start()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        ############################ added inference #############################
        elif self.sensors[self.index][2].startswith(f'{self.model_name_1}'):
            self.get_inf_name = f'{self.model_name_1}'            
            mask = self.inf_1.processing(image)
            if self.qrecording is not None: self.qrecording.add(mask, image)
            self.surface = pygame.surfarray.make_surface(mask.swapaxes(0, 1))
        elif self.sensors[self.index][2].startswith(f'{self.model_name_2}'):
            self.get_inf_name = f'{self.model_name_2}'
            mask = self.inf_2.processing(image)
            if self.qrecording is not None: self.qrecording.add(mask, image)
            self.surface = pygame.surfarray.make_surface(mask.swapaxes(0, 1))
        elif self.sensors[self.index][2].startswith(f'{self.model_name_3}'):
            self.get_inf_name = f'{self.model_name_3}'
            mask = self.inf_3.processing(image)
            if self.qrecording is not None: self.qrecording.add(mask, image)
            self.surface = pygame.surfarray.make_surface(mask.swapaxes(0, 1))
        elif self.sensors[self.index][2].startswith(f'{self.model_name_4}'):
            self.get_inf_name = f'{self.model_name_4}'
            mask = self.inf_4.processing(image)
            if self.qrecording is not None: self.qrecording.add(mask, image)
            self.surface = pygame.surfarray.make_surface(mask.swapaxes(0, 1))
        ########################################################################
        else:
            self.get_inf_name = None
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.time_measurement.end()
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    start = time.time()
    if args.cc_gen_mode:
        path = get_folder_name() # folders already created in utils/spawn_npc.py
    else:
        if args.demonstration:
            path =''
        else:
            path = output_folders()
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(7.0)
        
        display = pygame.display.set_mode(
            (cfg.resolution[0], cfg.resolution[1]),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(cfg.resolution[0], cfg.resolution[1])
        
        weather = Weather(client, args.weather)
        
        if args.save_inference_images: 
            qrecording = QRecording(cfg.fps_server, 
                                    cfg.save_seconds_before_cc, 
                                    cfg.record_every_x_frames)
        else:
            qrecording= None
        world = World(client.get_world(), hud, args.filter, path, cfg.model_name, qrecording, args)
        if not args.demonstration: weather.save_weather()
        #----------------- controller --------------------
        td = TravelDistance(world.get_ego_location())
        ccc = CheckCornerCase(world, args, client, td, qrecording, weather) 
        controller = DualControl(world, args.autopilot, client, ccc, start, weather)

        #----------------- visualisation ----------------- 
        # show traffic lights and speed
        #------------------------------------------------- 
        tld = TrafficLightsDisplay(world.world)
        speedmeter = SpeedDisplay()
        clock = pygame.time.Clock()
        
        #----------------- time step --------------------- 
        # *** fixed time step configured ***
        #------------------------------------------------- 
        settings = world.world.get_settings()
        settings.fixed_delta_seconds=1/cfg.fps_server
        world.world.apply_settings(settings)

        ego_car = get_ego_car(world.world.get_actors())
        frame = 1
        record_every_x_frames = 10
        remove_fences(world.world)
                
        while True:
            clock.tick_busy_loop(cfg.fps_server)
            if controller.parse_events(world, clock, td):
                return 
            world.tick(clock)
            world.render(display)

            if frame % record_every_x_frames == 0:
                save_csv(ego_car, world.world.get_actors(), cfg.radius_trajectory, frame, start)
            frame += 1

            tld.fetch_tl_landmarks(world.player.get_location(), distance=50)
            if cfg.available_displays == 3:
                tld.render(display, (cfg.resolution[0]-1350, 30)) # for 3 displays
            else:
                tld.render(display, (cfg.resolution[0]-100, 30)) 
            tld.change_tl_state()
            td.next(world.get_ego_location())
            speedmeter.render(world.player, display, (cfg.width//2, 30))
            pygame.display.flip()
    finally:
        if ccc is not None: 
            ccc.delete_recording()
            print('recording stopped')
        if not args.demonstration:
            with open(os.path.join(path, '00_log', 'recording_time_seconds.txt'), 'a') as file:
                    file.write('{}\n'.format(int(time.time()-start)))
        if world is not None:            
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')    #TODO: autopilot chrashes when stopping
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable car lights')
    argparser.add_argument(
        '-s','--semseg_name',
        type=str,
        help='name of the semantic driver')
    argparser.add_argument(
        '-f', '--safety_name',
        type=str,
        help='name of the safety driver')
    argparser.add_argument(
        '-ccgm', '--cc_gen_mode',
        action='store_true',
        default=False,
        help='corner case generation mode, needed when cc_gui.py started')
    argparser.add_argument(
        '-demo', '--demonstration',
        action='store_true',
        default=False,
        help='demonstration mode, no recording and drive on 3 monitors') #TODO: not implemented yet
    argparser.add_argument(
        '-images', '--save_inference_images',
        action='store_true',
        default=False,
        help='save inference images')
    argparser.add_argument(
        '-w', '--weather',
        type=str,
        default='clear',
        help='choose between clear, rain, fog, night')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=False)

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()