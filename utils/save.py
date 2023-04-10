import time
import json
import os
import csv
from utils.tools import get_folder_name

class SaveContext():
    def __init__(self, path):
        self.context={}
        # self.context['environment']={}
        self.path = path
        # needed for collect_trajectories:
        self.trajectories = {}
        self.age = {}
        self.timestamp = None
        self.last_loc = {}

    
    def map(self, world):
        self.context['map']= world.get_map().name
    
    def weather(self, world):
        self.context['weather'] = { 
            'cloudiness':           world.get_weather().cloudiness,
            'rain intensity':       world.get_weather().precipitation,
            'puddle coverage':      world.get_weather().precipitation_deposits,
            'wetness':              world.get_weather().wetness,
            'sun altitude':         world.get_weather().sun_altitude_angle,
            'sun azimuth':          world.get_weather().sun_azimuth_angle,
            'wind intensity':       world.get_weather().wind_intensity, 
            'fog concentration':    world.get_weather().fog_falloff,
            'fog density':          world.get_weather().fog_density,
            'fog distance':         world.get_weather().fog_distance,
            'fog light':            world.get_weather().scattering_intensity,
            'mie':                  world.get_weather().mie_scattering_scale,
            'rayleigh':             world.get_weather().rayleigh_scattering_scale,
        }

    def duration(self, start):
        """
        Length of the recorded scene in seconds
        """
        self.context['scene_recording_time']= int(time.time()-start)
    
    def duration_sensor_generation(self, start):
        """
        considered time period in seconds for saving of sensor data
        """
        self.context['sensor_recording_time']= int(time.time()-start)
    
    def ego_vehicle(self, ego_vehicle, ego_spawn_point):
        self.context['ego_vehicle']= {
            'id': ego_vehicle.id,
            'type_id': ego_vehicle.type_id,
            'starting_location': format(ego_spawn_point.location),
            'starting_rotation': format(ego_spawn_point.rotation)
    }

    def get_actors(self, world):
        world_snapshot = world.get_snapshot()
        vehicle_ids = {}
        walker_ids = {}

        for actor_snapshot in world_snapshot:
            actual_actor = world.get_actor(actor_snapshot.id)

            if str(actual_actor.type_id).startswith('vehicle'):
                vehicle_ids['{0:04d}'.format(actual_actor.id)] ={ 
                    'id':           actual_actor.id,
                    'actor_type':   str(actual_actor.type_id)
                }  
            if str(actual_actor.type_id).startswith('walker'):
                walker_ids['{0:04d}'.format(actual_actor.id)] ={ 
                    'id':           actual_actor.id,
                    'actor_type':   str(actual_actor.type_id)
                }
        self.context['number_vehicles'] = len(vehicle_ids)
        self.context['number_walkers'] = len(walker_ids)
        self.context['vehicles'] = vehicle_ids
        self.context['walkers'] = walker_ids
        


    def sensor_settings(self, sensor_list, sensor_pos_ego):
        loc = sensor_pos_ego.location
        rot = sensor_pos_ego.rotation
        self.context['sensor_settings'] = {
            'camera':       sensor_list[0].attributes,
            'semseg':       sensor_list[1].attributes,
            'lidar' :       sensor_list[2].attributes}
        self.context['sensor_pos_ego'] = {
            'x': loc.x, 'y': loc.y, 'z': loc.z, 
            'pitch': rot.pitch, 'yaw': rot.yaw, 'roll': rot.roll
            }

    def sensor_locations(self, world, sensor_list):
        actor_cam = world.get_actor(sensor_list[0].id)
        actor_lid = world.get_actor(sensor_list[2].id)
        loc_cam = actor_cam.get_transform().location
        rot_cam = actor_cam.get_transform().rotation
        loc_lid = actor_lid.get_transform().location
        rot_lid = actor_lid.get_transform().rotation
        self.context['transforms']= {'camera': { 
                'location' : {'x': loc_cam.x, 'y': loc_cam.y, 'z': loc_cam.z},
                'rotation' : {'pitch': rot_cam.pitch, 'yaw': rot_cam.yaw, 'roll': rot_cam.roll},
                'matrix'   : str(actor_cam.get_transform().get_matrix())
                }, 
                'lidar': { 
                'location' : {'x': loc_lid.x, 'y': loc_lid.y, 'z': loc_lid.z},
                'rotation' : {'pitch': rot_lid.pitch, 'yaw': rot_lid.yaw, 'roll': rot_lid.roll},
                'matrix'   : str(actor_lid.get_transform().get_matrix())
                }
        }

    def save_json_post(self, scene, num=None):
        if num is not None:
            with open(os.path.join(self.path, scene, '00_log', f'context_{num}.json'), "w") as f:
                json.dump(self.context, f)
        else:
            with open(os.path.join(self.path, scene, '00_log', f'context_sensor_generation.json'), "w") as f:
                json.dump(self.context, f)
    
    def save_json_pre(self):
        with open(os.path.join(self.path, '00_log', f'context.json'), "w") as f:
            json.dump(self.context, f)
    
    def collect_trajectories(self, ego_vehicle, snapshot, world, fps, bboxes = None, radius = 15):
        """
        trajectories method for json
        :param ego_vehicle:     actor object from carla referencing the ego vehicle
        :param snapshot:        carla world snapshot
        :param world:           carla world object # maybe we don't need the snapshot then
        :param fps:             int determining the fps, has to be correct for speed and acceleration calculation
        :param bboxes:          Python dictionary derived from Fabian's output json
        :param radius:          float determining the radius around the ego vehicle in which actors are recorded [m]
        """
        if self.timestamp is not None:
            self.timestamp = int(self.timestamp + (10**6)/fps)
            #self.timestamp = int(time.time() * 10**6)
        else:
            self.timestamp = int(time.time() * 10**6)

        ego_car_location = ego_vehicle.get_location()
        eclx = ego_car_location.x
        ecly = ego_car_location.y
        actors_in_scene = []
        data_list = [] # used to store entries for the current timestamp

        actors = [world.get_actor(actor_snapshot.id) for actor_snapshot in snapshot] # extract the actor objects from snapshot
        for actor in actors:
            if ((actor.get_location().x - eclx)**2 + (actor.get_location().y - ecly)**2)**0.5 < radius:
                actors_in_scene.append(actor)

        for actor in actors_in_scene:
            if str(actor.type_id).startswith('vehicle') or str(actor.type_id).startswith('walker'):
                transform = actor.get_transform()
                geolocation = world.get_map().transform_to_geolocation(transform.location)
                distance = ((actor.get_location().x - eclx)**2 + (actor.get_location().y - ecly)**2)**0.5

                # update age
                if str(actor.id) in self.age.keys():
                    self.age[str(actor.id)] += 1
                else:
                    self.age[str(actor.id)] = 1  

                # create actor entry
                if str(actor.type_id) in ["vehicle.mercedes.sprinter", "vehicle.volkswagen.t2", "vehicle.volkswagen.t2_2021", "vehicle.ford.ambulance"]:
                    class_name1 = "van"
                elif str(actor.type_id).startswith('walker'):
                    class_name1 = "pedestrian"
                elif str(actor.type_id) in ["vehicle.yamaha.yzf", "vehicle.vespa.zx125", "vehicle.kawasaki.ninja"]:
                    class_name1 = "motorcycle"
                elif str(actor.type_id) == "vehicle.bh.crossbike":
                    class_name1 = "cyclist"
                else:
                    class_name1 = "car"


                if bboxes is not None:
                    for i in range(len(bboxes[0]['3d'])):
                        if bboxes[0]["3d"][i]["id"] == actor.id:
                            index = i

                    if index is not None:
                        entry = {
                        "id": actor.id,
                        "age": self.age[str(actor.id)], 
                        "existence_prob": 1, 
                        "class_name1": class_name1, 
                        "class_prob1": 1, 
                        "latitude": geolocation.latitude,
                        "longitude": geolocation.longitude, 
                        "length": bboxes[0]["3d"][index]["extent"][0] * 2, # multiplied by two to get box extent
                        "width": bboxes[0]["3d"][index]["extent"][1] * 2,
                        "height": bboxes[0]["3d"][index]["extent"][2] * 2, 
                        "orientation": transform.rotation.yaw + 180 # added 180 degrees to get the scale in [0, 360]
                        }

                    else:
                        print("ERROR: could not find bounding box for actor {}.".format(actor.id))
                        #raise IndexError
                else:
                    entry = {
                        "id": actor.id,
                        "age": self.age[str(actor.id)], 
                        "existence_prob": 1, 
                        "class_name1": class_name1, 
                        "class_prob1": 1, 
                        "latitude": geolocation.latitude,
                        "longitude": geolocation.longitude, 
                        "length": "NA", 
                        "width": "NA",
                        "height": "NA", 
                        "orientation": transform.rotation.yaw
                        }

                #---- Berechnung der Geschwindigkeiten + Beschleunigungen 
                #if last_loc is None:
                if str(actor.id) in self.last_loc.keys():
                    timedelta = (self.timestamp - self.last_loc[str(actor.id)]['starttime']) / (10 ** 6)
                    #print(timedelta)
                    vx = (transform.location.x - self.last_loc[str(actor.id)]['locx'])/timedelta
                    vy = (transform.location.y - self.last_loc[str(actor.id)]['locy'])/timedelta
                    vz = (transform.location.z - self.last_loc[str(actor.id)]['locz'])/timedelta
                    ax = (vx - self.last_loc[str(actor.id)]['vx'])/timedelta
                    ay = (vy - self.last_loc[str(actor.id)]['vy'])/timedelta
                    az = (vz - self.last_loc[str(actor.id)]['vz'])/timedelta

                    # updating the entry
                    entry["speed"] = (vx ** 2 + vy ** 2 + vz ** 2)**0.5 # in m/s
                    entry["acceleration_longitudinal"] = (ax ** 2 + ay ** 2 + az ** 2)**0.5 # in m/s²
                        
                    # ----- last_loc  --> Liste erzeugen mit den aktuellen Positionen aller Teilnehmer (Auto+ Fußgänger)
                    self.last_loc[str(actor.id)] = {
                        'locx':      transform.location.x,
                        'locy':      transform.location.y,
                        'locz':      transform.location.z,
                        'vx':        vx,
                        'vy':        vy,
                        'vz':        vz,
                        'starttime': self.timestamp
                    }
                    
                else:  
                    # updating the entry
                    entry["speed"] = 0
                    entry["acceleration_longitudinal"] = 0
                        
                        # ----- last_loc  --> Liste erzeugen mit den aktuellen Positionen aller Teilnehmer (Auto+ Fußgänger)
                    self.last_loc[str(actor.id)] = {
                        'locx':      transform.location.x,
                        'locy':      transform.location.y,
                        'locz':      transform.location.z,
                        'vx':        0,
                        'vy':        0,
                        'vz':        0,
                        'starttime': self.timestamp
                    }           
                #print(actor.id, self.last_loc[str(actor.id)])        

                entry["kidt_car_company"] = "buw"
                entry["kidt_car_latitude"] = world.get_map().transform_to_geolocation(ego_car_location).latitude
                entry["kidt_car_longitude"] = world.get_map().transform_to_geolocation(ego_car_location).longitude
                        
                # add entry to the list of actor data for the current timestamp
                data_list.append(entry)

        self.trajectories[str(self.timestamp)] = data_list 

    def save_trajectories_json(self, scene, run=None):
        if run is None:
            with open(os.path.join(self.path, scene, '08_trajectory', 'trajectories.json'), 'w') as file:
                #path = get_folder_name()
                json.dump(self.trajectories, file, indent = 4)
        else:
            tmp = os.path.join(self.path, scene, '08_trajectory', run)
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            if os.path.exists(os.path.join(tmp, 'trajectories.json')):
                os.remove(os.path.join(tmp, 'trajectories.json')) 
            with open(os.path.join(tmp, 'trajectories.json'), 'w') as file:
                json.dump(self.trajectories, file, indent = 4)


def save_csv(ego_car, actors, radius, frame, start):
    """
    saves position, velocity, acceleration of each moving actor within a predefined radius in a csv file
    :param ego_car: actor object representing the driver
    :param actors: the world's actor list
    :param radius: Objects to be stored within a radius in [m] 
    :param frame: current frame
    :param start: starting time
    """

    ego_car_location = ego_car.get_location()
    eclx = ego_car_location.x
    ecly = ego_car_location.y
    actors_in_scene = []
    path = get_folder_name()
        
    for actor in actors:
        if ((actor.get_location().x - eclx)**2 + (actor.get_location().y - ecly)**2)**0.5 < radius:
            actors_in_scene.append(actor)
    
    fields = [  'id','frame','time [ms]', 'type', 'x [m]', 'y [m]', 'z [m]', 'distance to ego [m]',
                'vx [m/s]', 'vy [m/s]', 'vz [m/s]', 
                'ax [m/s2]', 'ay [m/s2]', 'az [m/s2]', 
                'pitch [deg]', 'yaw [deg]', 'roll [deg]'    ]
    rows = []

    for actor in actors_in_scene:
        if str(actor.type_id).startswith('vehicle') or str(actor.type_id).startswith('walker'):
            transform = actor.get_transform()
            distance = ((actor.get_location().x - eclx)**2 + (actor.get_location().y - ecly)**2)**0.5

            row = [     str(actor.id),
                        str(frame),
                        str(int(1000*(time.time()-start))),
                        str(actor.type_id), 
                        str(round(transform.location.x,2)), 
                        str(round(transform.location.y,2)), 
                        str(round(transform.location.z,2)),
                        str(round(distance, 2)), 
                        str(round(actor.get_velocity().x,2)), 
                        str(round(actor.get_velocity().y,2)), 
                        str(round(actor.get_velocity().z,2)), 
                        str(round(actor.get_acceleration().x,2)),
                        str(round(actor.get_acceleration().y,2)), 
                        str(round(actor.get_acceleration().z,2)), 
                        str(round(transform.rotation.pitch,2)), 
                        str(round(transform.rotation.yaw,2)), 
                        str(round(transform.rotation.roll,2))   ]
            rows.append(row)   

    with open(os.path.join(path, '08_trajectory', 'actor_data.csv'), 'a') as file:
        path = get_folder_name()
        csvwriter = csv.writer(file)
        if not os.path.exists(os.path.join(path, '08_trajectory', 'actor_data.csv')) or os.stat(os.path.join(path, '08_trajectory', 'actor_data.csv')).st_size == 0:
            csvwriter.writerow(fields)
        csvwriter.writerows(rows)