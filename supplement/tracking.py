import glob
import os
import sys
import config as cfg 

try:
    sys.path.append(glob.glob(cfg.path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time


def main():
    """
    script for printing the actual location, velocity and acceleration
    """

    try:
        
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        actors = world.get_actors()
        for actor in actors:
            if not actor.attributes:
                pass
            else:
                if actor.attributes['role_name'] == 'hero':
                    ego_car = actor
                    break
        while True:
            # #print('Actor ID: ', ego_car.id)
            
            print('Location ', ego_car.get_location())
            print('velo [m/s]: ', ego_car.get_velocity())
            print('acc: [m/s^2]', ego_car.get_acceleration())
            print('transform: ', ego_car.get_transform())
            print('vehicle: ', ego_car.type_id)
            # print('attributes: ', ego_car.attributes)
            # print('parent: ', ego_car.parent)
            # print('world: ', ego_car.get_world())
            

    finally:
        pass

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')