#!/usr/bin/env python
from decimal import localcontext
import os
import time
import sys
import glob
import copy
from cv2 import transform
import config as cfg 

try:
    sys.path.append(glob.glob(cfg.path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import signal
import subprocess
import random
import config as cfg
random.seed(10)
print(random.random()) 
client = carla.Client('localhost', 2000)
world =  client.get_world()
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[23]

# transform = carla.Transform(carla.Location(x=5,y =50, z=5), carla.Rotation(yaw=-90))
time_sleep = 60
actors = []
x = spawn_point.location.x
y = spawn_point.location.y
for n in range(150):
    try:
        i = random.randint(1, 25)
        rot = random.randint(0, 360)
        walker_bp = bp_lib.find(f'walker.pedestrian.00{i:02}')
        locx = random.uniform(x+6, x+12)
        locy = random.uniform(y-10, y+10)
        # locx=x+8
        # locy=y+2
        spawn_point.location.x=locx
        spawn_point.location.y=locy
        spawn_point.rotation.yaw=rot
        actor = world.spawn_actor(walker_bp, spawn_point)
        actors.append(actor)
    except:
        pass
        
time.sleep(time_sleep)
for actor in actors:
    error = actor.destroy()