#!/usr/bin/env python
import os
import time
import sys
import glob
sys.path.append(os.getcwd())
import config as cfg 

try:
    sys.path.append(glob.glob(cfg.path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


client = carla.Client('localhost', 2000)
world =  client.get_world()
bp_lib = world.get_blueprint_library()
cycle_bp = bp_lib.find('vehicle.bh.crossbike')
transform = carla.Transform(carla.Location(x=5,y =50, z=5), carla.Rotation(yaw=-90))
time_sleep = 10
actor_cycle = world.spawn_actor(cycle_bp, transform)
        
### apply physics control 
physics_control = actor_cycle.get_physics_control()
physics_control.use_sweep_wheel_collision = True
actor_cycle.apply_physics_control(physics_control)
actor_cycle.set_autopilot(True)
        
time.sleep(time_sleep)
error = actor_cycle.destroy()