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
import subprocess

def test_scene():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    vehicle_bp = bp_lib.find('vehicle.audi.tt')
    vehicle_bp.set_attribute('color', '255,0,0')

    # time.sleep(10)
    for i in range(2000):
        spawn_bike_proc = subprocess.Popen('supplement/spawn_bike.py', cwd=cfg.path_buw, preexec_fn=os.setsid)
        spawn_bike_proc.wait()
      

def main():
    try:
        test_scene()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()