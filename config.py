#!/usr/bin/env python
import getpass
import platform

name_out_folder = "scene"
user = str(getpass.getuser())
os_system = platform.system()

carla_version = 10

if os_system == 'Linux':
    path_egg_file = f'/home/{user}/Dokumente/CARLA_0.9.{carla_version}/PythonAPI'
    path_carla = f'/home/{user}/Dokumente/CARLA_0.9.{carla_version}' 
    path_buw = f'/home/{user}/Dokumente/AEye'
else:
    path_egg_file = r"C:\UnrealEngine\carla\PythonAPI"
    path_carla = r'C:\UnrealEngine\carla' 
    path_buw = f'C:\\Users\\{user}\\Documents\\Eye' 

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys

try:
    sys.path.append(glob.glob(path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================


fps_server = 20
save_seconds_before_cc = 7
record_every_x_frames = 3
radius_trajectory = 15

available_displays= 1 # [1,3]
resolution = [1280, 640] # [width, height]
width = resolution[0]
height = resolution[1]

model_name = 'fast_scnn' # [fast_scnn, bisenetv2]
weather = 'clear'         # available: clear, rain, fog, night


duration = 100

nr_runs = 10

town = 3     
nr_vehicles = [50, 100]           # [min, max]
nr_walkers  = [50, 100]           # [min, max]


live_plotter = False        # tracks the pedals of each driver


#---------------------------- checkpoints ----------------------------
ckpt_1 = 'clear_FastSCNN_impaired-perception'
ckpt_2 = 'clear_FastSCNN_cornercases' 
ckpt_3 = 'clear_FastSCNN_origin'
ckpt_4 = 'clear_FastSCNN_pedestrian'

