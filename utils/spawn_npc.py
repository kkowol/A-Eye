#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from random import sample

from numpy import random
# print(os.getcwd())
from tools import get_folder_name, output_folders_data_generator

# sys.path.append(os.getcwd())

import config as cfg

try:
    sys.path.append(glob.glob(cfg.path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


if cfg.carla_version > 10:
    from carla import VehicleLightState as vls

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
SetVehicleLightState = carla.command.SetVehicleLightState
FutureActor = carla.command.FutureActor


def ignore_blueprints(blueprints):
    """
    ignore specific blueprints like 2 wheeled vehicles, teslas cybertruck etc.
    :param blueprints: list of vehicle blueprints
    """
    # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    # blueprints = [x for x in blueprints if not x.id.endswith('omafiets')] # bicyle
    # blueprints = [x for x in blueprints if not x.id.endswith('crossbike')] # bicyle
    blueprints = [x for x in blueprints if not x.id.endswith('zx125')] # vespa

    return blueprints

def spawning_area(spawn_points, spawning_area):
    """
    get only points in a predefined area around the ego vehicle
    :param spawn_points: list
    """
    area_points = []
    for point in spawn_points:
        if point.get_matrix()[0][3] >= spawning_area[0] and point.get_matrix()[0][3] < spawning_area[1]:
            if point.get_matrix()[1][3] >= spawning_area[2] and point.get_matrix()[1][3] < spawning_area[3]:
                area_points.append(point)
    return area_points


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
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
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enable')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        default=5,
        type=int,
        help='the value itself is not relevant, but the same value will always result in the same output. Two simulations, with the same conditions, that use the same seed value, will be deterministic')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable car lights')
    argparser.add_argument(
        '--spawning-area',
        default=None,
        nargs='+',
        type = int,
        help='Use a spawning area around the ego car [x_min, x_max, y_min, y_max]')
    argparser.add_argument(
        '--spawning-values',
        default=None,
        nargs='+',
        type = float,
        help='Use percentage spawning values for vehicles/walkers [perc_v, perc_w] related to the maps size')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = True
    random.seed(args.seed if args.seed is not None else int(time.time()))
    if args.spawning_area is not None or args.spawning_values is not None:
        path = get_folder_name()
    else:
        path = output_folders_data_generator()

    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        # needed for deterministic mode (https://carla.readthedocs.io/en/latest/adv_traffic_manager/#deterministic-mode)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)
        # use hybrid physics mode
        # This feature removes the vehicle physics bottleneck from the simulator
        traffic_manager.set_hybrid_physics_radius(25)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05

        # --------------
        # Spawn vehicles
        # --------------
        if args.spawning_values is not None:
            perc_v = args.spawning_values[0]
            perc_w = args.spawning_values[1]
        # percent_vehicles = random.uniform(0.6, 0.9)
        # percent_walkers = 1.5

        spawn_points = world.get_map().get_spawn_points()
        if args.spawning_area is not None:
            spawn_points = spawning_area(spawn_points, args.spawning_area)
            number_of_vehicles = int(perc_v*len(spawn_points))
        elif args.spawning_values is not None: 
            number_of_vehicles = int(perc_v*len(spawn_points))
        else:
            number_of_vehicles = args.number_of_vehicles
        number_of_spawn_points = len(spawn_points)
        
        if number_of_vehicles <= number_of_spawn_points:
            random.shuffle(spawn_points)
        else:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        print(f'Trying to spawn {number_of_vehicles} vehicles...')

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprints = ignore_blueprints(blueprints)
        batch = []
        vehicles = []
        for spawn_point in spawn_points[:number_of_vehicles]:
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            if cfg.carla_version > 10:
                light_state = vls.NONE
                if args.car_lights_on:
                    light_state = vls.Position | vls.LowBeam | vls.LowBeam

            loc = spawn_point.location
            rot = spawn_point.rotation
            vehicles.append({
                'type_id': blueprint.id,
                'tags': blueprint.tags,
                'location': {'x': loc.x, 'y': loc.y, 'z': loc.z},
                'rotation': {'pitch': rot.pitch, 'yaw': rot.yaw, 'roll': rot.roll},
            })

            # spawn the cars and set their autopilot and light state all together
            if cfg.carla_version > 10:
                batch.append(SpawnActor(blueprint, spawn_point)
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                    .then(SetVehicleLightState(FutureActor, light_state)))
            else: 
                batch.append(SpawnActor(blueprint, spawn_point)
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
                
        iter = zip(vehicles, client.apply_batch_sync(batch, synchronous_master))
        vehicles = [dict(vehicle, id=response.actor_id) for vehicle, response in iter if not response.error]

        with open(os.path.join(path, '00_log',  'vehicles.json'), 'w') as file:
            json.dump(vehicles, file)

        print(f'Spawned {len(vehicles)} vehicles.', end='\n\n')

        # -------------
        # Spawn Walkers
        # -------------
        if args.spawning_area is not None:
            number_of_walkers = int(perc_w * number_of_vehicles)
        elif args.spawning_values is not None: 
            number_of_walkers = int(perc_w*number_of_vehicles)
        else:
            number_of_walkers = args.number_of_walkers
        # some settings
        percentagePedestriansRunning = 0.0 # how many pedestrians will run
        percentagePedestriansCrossing = 0.0 # how many pedestrians will walk through the road

        print(f'Trying to spawn {number_of_walkers} walkers...')

        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
        batch = []
        walkers = []
        for spawn_point in spawn_points:
            blueprint = random.choice(blueprints)

            # set as not invincible
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'false')

            # set the max speed
            if blueprint.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    speed = float(blueprint.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    speed = float(blueprint.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                speed = 0.0

            loc = spawn_point.location
            rot = spawn_point.rotation
            walkers.append({
                'type_id': blueprint.id,
                'tags': blueprint.tags,
                'speed': speed,
                'location': {'x': loc.x, 'y': loc.y, 'z': loc.z},
                'rotation': {'pitch': rot.pitch, 'yaw': rot.yaw, 'roll': rot.roll},
            })

            batch.append(SpawnActor(blueprint, spawn_point))

        iter = zip(walkers, client.apply_batch_sync(batch, True))
        walkers = [dict(walker, id=response.actor_id) for walker, response in iter if not response.error]

        # 3. we spawn the walker controller
        blueprints = world.get_blueprint_library().find('controller.ai.walker')
        batch = [SpawnActor(blueprints, carla.Transform(), walker['id']) for walker in walkers]

        iter = zip(walkers, client.apply_batch_sync(batch, True))
        walkers = [dict(walker, cid=response.actor_id) for walker, response in iter if not response.error]

        with open(os.path.join(path, '00_log', 'walkers.json'), 'w') as file:
            json.dump(walkers, file)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 4. initialize each controller and set target to walk to
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for walker in walkers:
            actor = world.get_actor(walker['cid'])
            actor.start()
            actor.go_to_location(world.get_random_location_from_navigation())
            actor.set_max_speed(walker['speed'])

        print(f'Spawned {len(walkers)} walkers (with controllers).', end='\n\n')

        print('Press CTRL+C to quit.')
        # with open(os.path.join(path, '00_log', 'scene_setup.txt'), 'a') as file:
        #     file.write('spawned %d vehicles and %d walkers\n\n' % (len(vehicles), len(walkers)))

        # example of how to use parameters

        ##### vehicle speeds #####
        # traffic_manager.global_percentage_speed_difference(30.0) # standard value
        traffic_manager.global_percentage_speed_difference(-50.0) # Vehicles now go 60% faster than the speed limit (<0 -> faster / >0 -> slower)
        # more info: https://carla.readthedocs.io/en/latest/adv_traffic_manager/#configuring-autopilot-behavior

        #######################################################################################
        # this allows to separate the vehicles into 3 groups with different speed limits
        # the remaining 60% are controlled by the global speed limit

        percent_speeding_vehicles = 30 # set the percentage of speeding vehicles
        percent_slower_vehicles = 10 # set the percentage of slower vehicles

        speeding_vehicles = sample(vehicles, int(percent_speeding_vehicles / 100 * len(vehicles))) # 30% speeding vehicles, randomly picked from all vehicles

        for v in speeding_vehicles:
            traffic_manager.vehicle_percentage_speed_difference(world.get_actor(v['id']), -70.0) # they go 60% faster than the regular speed limit

        slower_vehicles = sample([v for v in vehicles if v not in speeding_vehicles], int(percent_slower_vehicles / 100 * len(vehicles))) # 10% slower vehicles

        for v in slower_vehicles:
            traffic_manager.vehicle_percentage_speed_difference(world.get_actor(v['id']), -30.0) # they go 30% faster than the regular speed limit but still slower than everyone else
        #######################################################################################

        while True:
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    finally:
        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print(f'\nDestroying {len(vehicles)} vehicles...')
        client.apply_batch([carla.command.DestroyActor(vehicle['id']) for vehicle in vehicles])

        for walker in walkers:
            actor = world.get_actor(walker['cid'])
            actor.stop()

        print(f'Destroying {len(walkers)} walkers...')
        client.apply_batch([carla.command.DestroyActor(id) for id in (walker['id'], walker['cid']) for walker in walkers])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.')
