import random
import logging
import os
import config as cfg
import carla
from utils.carla import ignore_blueprints
from carla import VehicleLightState as vls

def spawning(args, client, world, path, nr_vehicles, nr_walkers): 
    """
    spawning of vehicles and pedestrians
    """
    vehicles_list = []
    walkers_list = []
    all_id = []
    synchronous_master = True

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    # use hybrid physics mode
    # This feature removes the vehicle physics bottleneck from the simulator
    traffic_manager.set_hybrid_physics_radius(50)                   

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    blueprints = sorted(blueprints, key=lambda bp: bp.id)
    blueprints = ignore_blueprints(blueprints)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    number_of_vehicles = nr_vehicles

    if number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif number_of_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, number_of_vehicles, number_of_spawn_points)
        number_of_vehicles = number_of_spawn_points
    
    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------
    batch2 = []
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        if args.car_lights_on:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam
              
        
        with open(os.path.join(path, '00_log',  'vehicles.txt'), 'a') as file:
            file.write('{}\n'.format(blueprint.id))
            file.write('{}\n'.format(blueprint.tags))
            file.write('{}\n\n'.format(transform))

        # spawn the cars and set their autopilot and light state all together
        batch2.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch2, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    
    with open(os.path.join(path,'00_log', 'vehicles.txt'), 'a') as file:
        file.write('---------- vehicles_list (IDs)----------\n')
        file.write('{}\n'.format(vehicles_list))
    
    # -------------
    # Spawn Walkers
    # -------------

    number_of_walkers = nr_walkers
    # some settings
    percentagePedestriansRunning = 0.0      # how many pedestrians will run
    percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(number_of_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        
        # save vehicle list
        with open(os.path.join(path, '00_log', 'walkers.txt'), 'a') as file:  
            file.write('{}\n'.format(walker_bp))
            file.write('{}\n\n'.format(spawn_point))

        batch.append(SpawnActor(walker_bp, spawn_point))
    
    results = client.apply_batch_sync(batch, True)
    
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            pass
            # logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    
    for i in range(len(results)):
        if results[i].error:
            pass
            # logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id

    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)
    
    with open(os.path.join(path, '00_log', 'walkers.txt'), 'a') as file:
            file.write('---------- walkers_list (IDs)----------\n')
            file.write('{}\n'.format(walkers_list))
    
    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    if not args.sync or not synchronous_master:
        world.wait_for_tick()
    else:
        world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))
    with open(os.path.join(path, '00_log', 'scene_setup.txt'), 'a') as file:
            file.write('spawned %d vehicles and %d walkers' % (len(vehicles_list), len(walkers_list)))    
    # example of how to use parameters
    traffic_manager.global_percentage_speed_difference(30.0)

    return batch2


def spawning_radius(args, client, world, path, spawning_area, percentage_of_points, percentage_walkers): 
    """
    spawning of vehicles and pedestrians
    spawning_area --> allowed spawning coordinates [spawn_x_min, spawn_x_max, spawn_y_min, spawn_y_max]
    percentage_of_points --> percentage usage of spawning points in the spawning radius
    percentage_walkers   -->    use of some more walkers than available vehicle points due to the fact, 
                                that not all walkers can be spawned
    """
    vehicles_list = []
    walkers_list = []
    all_id = []
    synchronous_master = True
    percent_v = percentage_of_points
    percent_w = percentage_walkers

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    # use hybrid physics mode
    # This feature removes the vehicle physics bottleneck from the simulator
    traffic_manager.set_hybrid_physics_radius(50)                   

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    blueprints = sorted(blueprints, key=lambda bp: bp.id)
    blueprints = ignore_blueprints(blueprints)

    spawn_points = world.get_map().get_spawn_points()
    list_radius = []
    for point in spawn_points:
        if point.get_matrix()[0][3] >= spawning_area[0] and point.get_matrix()[0][3] < spawning_area[1]:
            if point.get_matrix()[1][3] >= spawning_area[2] and point.get_matrix()[1][3] < spawning_area[3]:
                list_radius.append(point)
    spawn_points = list_radius

    number_of_spawn_points = len(spawn_points)
    number_of_vehicles = int(percent_v*number_of_spawn_points)

    if number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif number_of_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, number_of_vehicles, number_of_spawn_points)
        number_of_vehicles = number_of_spawn_points
    
    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------
    batch2 = []
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        if args.car_lights_on:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam
        
        
        with open(os.path.join(path, '00_log', 'vehicles.txt'), 'a') as file:  
            file.write('{}\n'.format(blueprint.id))
            file.write('{}\n\n'.format(transform))

        # spawn the cars and set their autopilot and light state all together
        batch2.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch2, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    
    with open(os.path.join(path, '00_log', 'vehicles.txt'), 'a') as file:
        file.write('---------- vehicles_list (IDs)----------\n')
        file.write('{}\n'.format(vehicles_list))
    
    # -------------
    # Spawn Walkers
    # -------------
    
    number_of_walkers = int(percent_w*number_of_vehicles)
    # some settings
    percentagePedestriansRunning = 0.0      # how many pedestrians will run
    percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn
    spawn_points = []
    while len(spawn_points) < number_of_walkers:        # get a list with points with the number of walkers
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            if loc.x >= spawning_area[0] and loc.x < spawning_area[1]:
                if loc.y >= spawning_area[2] and loc.y < spawning_area[3]:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
    
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        
        # save vehicle list
        with open(os.path.join(path, '00_log', 'walkers.txt'), 'a') as file:  
            file.write('{}\n'.format(walker_bp))
            file.write('{}\n\n'.format(spawn_point))

        batch.append(SpawnActor(walker_bp, spawn_point))
    
    results = client.apply_batch_sync(batch, True)
    
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            pass
            # logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    
    for i in range(len(results)):
        if results[i].error:
            pass
            # logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id

    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)
    
    with open(os.path.join(path, '00_log', 'walkers.txt'), 'a') as file:
            file.write('---------- walkers_list (IDs)----------\n')
            file.write('{}\n'.format(walkers_list))
    
    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    if not args.sync or not synchronous_master:
        world.wait_for_tick()
    else:
        world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))
    with open(os.path.join(path, '00_log', 'scene_setup.txt'), 'a') as file:
            file.write('spawned %d vehicles and %d walkers\n' % (len(vehicles_list), len(walkers_list)))
            file.write('spawned radius from ego vehicle %dm' % (args.radius))    
    # example of how to use parameters
    traffic_manager.global_percentage_speed_difference(30.0)

    return batch2