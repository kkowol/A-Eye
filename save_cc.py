#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import config as cfg 
from utils.inference import Inference
from utils.save import SaveContext
# from supplement.radar import radar as rd
import time
import subprocess
import signal
from PIL import Image
import json
import shutil

try:
    sys.path.append(glob.glob(cfg.path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import queue

width = 1920
height = 1080
fov = 90.



class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        fps = 10.0
        self.delta_seconds = 1.0/fps # needs to be <0.1!!!!   --> ändert die framerate !!! 
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def get_recordings(folder, path):
    """
    returns a list of all recorded files
    :param folder:  needs root folder of each recorded scene
    :param path:    working directory
    """
    recorded_files = []
    for i, recording in enumerate(sorted(os.listdir(os.path.join(path, folder, '00_log')))):
        if recording.startswith('scene_recording'):
            recorded_files.append(recording)
    return recorded_files


def load_weather(world, path, scene):
    """
    loading the orignal weather
    """
    with open(os.path.join(path, scene, '00_log', 'weather.json')) as f:
        json_data = json.load(f)
    
    weather = carla.WeatherParameters(
        cloudiness                  = json_data['weather']['cloudiness'],
        precipitation               = json_data['weather']['rain intensity'],
        precipitation_deposits      = json_data['weather']['puddle coverage'],
        wetness                     = json_data['weather']['wetness'],
        sun_altitude_angle          = json_data['weather']['sun altitude'],
        sun_azimuth_angle           = json_data['weather']['sun azimuth'],
        # sun_altitude_angle          = json_data['weather']['altitude sun'],
        wind_intensity              = json_data['weather']['wind intensity'],  
        fog_falloff                 = json_data['weather']['fog concentration'],
        fog_density                 = json_data['weather']['fog density'],
        fog_distance                = json_data['weather']['fog distance'],
        # scattering_intensity        = json_data['weather']['fog light'],
        # mie_scattering_scale        = json_data['weather']['mie'],
        # rayleigh_scattering_scale   = json_data['weather']['rayleigh']
    )
    world.set_weather(weather)

def load_weather_txt(world, path, scene):
    """
    loading the orignal weather
    """
    with open(os.path.join(path, scene, '00_log', 'scene_setup.txt')) as f:
        lines = f.readlines()
        cloud   = lines[2].strip().replace(' ', '').split(':')[1]
        rain    = lines[3].strip().replace(' ', '').split(':')[1]
        puddle  = lines[4].strip().replace(' ', '').split(':')[1]
        wet     = lines[5].strip().replace(' ', '').split(':')[1]
        sun_alt = lines[6].strip().replace(' ', '').split(':')[1]
        wind    = lines[7].strip().replace(' ', '').split(':')[1]
        fog_den = lines[8].strip().replace(' ', '').split(':')[1]
        fog_dis = lines[9].strip().replace(' ', '').split(':')[1]
    
    weather = carla.WeatherParameters(
        cloudiness              = np.float(cloud),
        precipitation           = np.float(rain),
        precipitation_deposits  = np.float(puddle),
        wetness                 = np.float(wet),
        sun_altitude_angle      = np.float(sun_alt),
        wind_intensity          = np.float(wind),  
        fog_density             = np.float(fog_den),
        fog_distance            = np.float(fog_dis)
    )
    world.set_weather(weather)


def kill_carla_world(carla_proc, sensor_list):
    """
    kills the CARLA world
    needs a second kill, because CARLA was still alive
    """
    for sensor in sensor_list:
            sensor.destroy()
    pid = os.getpgid(carla_proc.pid) # get the PID
    os.killpg(pid, signal.SIGTERM)
    outs, errs = carla_proc.communicate()
    time.sleep(2)
    # os.killpg(pid, signal.SIGTERM)
    # time.sleep(2)


def restart_carla_world(carla_proc, sensor_list):
    """
    kills the actual CARLA world and starts a new one
    :param carla_proc:  the actual CARLA world subprocess
    :param sensor_list: list of all active sensors
    :return carla_proc: new CARLA world subprocess
    :return client:     client
    :return world:      world
    """
    kill_carla_world(carla_proc, sensor_list)
    ### start carla world
    carla_proc = subprocess.Popen('./CarlaUE4.sh', cwd=cfg.path_carla, preexec_fn=os.setsid) 
    time.sleep(20)

    client = carla.Client('localhost', 2000, worker_threads=1)
    client.set_timeout(60.0)
    world = client.get_world()
    return carla_proc, client, world


def save_inference(image, path, scene, name_rec, frame, network_1, torch_transform, model_name):
    """
    save the networks output
    :param path:        working path
    :param scene:       actual scene
    :param name_rec:    name of the current recording folder
    :param frame:       actual frame
    """
    inf = Inference()
    tmp_path = os.path.join(path, scene, '04_inference', name_rec)
    ### create folder
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    mask = inf.inference(image, network_1, torch_transform, model_name)
    img = Image.fromarray(mask)
    img.save(os.path.join(tmp_path, 'inf_{0:05d}.png'.format(frame)))

def sensor_settings(world):
    """
    create sensors
    """
    
    #----------------------------------------------------------------------
    # Camera
    #----------------------------------------------------------------------
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f"{width}")
    camera_bp.set_attribute('image_size_y', f"{height}")
    # camera_bp.set_attribute('fov', f"{fov}")
    camera_bp.set_attribute('motion_blur_max_distortion', '0.0')
    camera_bp.set_attribute('motion_blur_intensity', '0.0')
    camera_bp.set_attribute('blur_amount', '0.0')
    camera_bp.set_attribute('motion_blur_min_object_screen_size', '0.0')
    # camera_bp.set_attribute('sensor_tick', '2.0')

    #----------------------------------------------------------------------
    # semantic Segmentation
    #----------------------------------------------------------------------
    segm_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    segm_bp.set_attribute('image_size_x', f"{width}")
    segm_bp.set_attribute('image_size_y', f"{height}")
    # segm_bp.set_attribute('fov', f"{fov}")
    # segm_bp.set_attribute('sensor_tick', '2.0')

    #----------------------------------------------------------------------
    # LiDAR
    #----------------------------------------------------------------------
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', '128')
    lidar_bp.set_attribute('range', '150.0')
    lidar_bp.set_attribute('points_per_second', '2500000')
    lidar_bp.set_attribute('rotation_frequency', '10.0')
    lidar_bp.set_attribute('upper_fov', '15.0')
    lidar_bp.set_attribute('lower_fov', '-25.0')
    # lidar_bp.set_attribute('horizontal_fov', '360.0')
    # lidar_bp.set_attribute('sensor_tick', '0.0')

    #----------------------------------------------------------------------
    # depth
    #----------------------------------------------------------------------
    depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', f"{width}")
    depth_bp.set_attribute('image_size_y', f"{height}")

    #----------------------------------------------------------------------
    # instance segmentation
    #----------------------------------------------------------------------
    # inst_bp = world.get_blueprint_library().find('sensor.camera.instance_segmentation')
    # inst_bp.set_attribute('image_size_x', f"{width}")
    # inst_bp.set_attribute('image_size_y', f"{height}")

    #----------------------------------------------------------------------
    # radar
    #----------------------------------------------------------------------    
    # radar_bp = world.get_blueprint_library().find('sensor.other.radar')
    # radar_bp.set_attribute('points_per_second', '5000')


    return camera_bp, segm_bp, lidar_bp, depth_bp#, inst_bp

# def lidar_save(path, depth, mask):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     ocfg = OmegaConf.load("../sensors/configs/carla.yaml") #
#     camera = buw_sensors.Camera(ocfg.camera) #
#     lidar = buw_sensors.Lidar(ocfg.lidar) #
#     precomputed_coords = camera.project_angles(*lidar.all_angles) #
#     depth_array = np.frombuffer(depth.raw_data, dtype=np.dtype("uint8"))
#     depth_array = np.reshape(depth_array, (depth.height, depth.width, 4))[..., :3]
#     convert = np.array([1, 256, 256 * 256], dtype=float) #
#     convert *= 1000 / (256 * 256 * 256 - 1) #
#     depth_array = depth_array @ convert
#     mask_array = np.frombuffer(mask.raw_data, dtype=np.dtype("uint8"))
#     mask_array = np.reshape(mask_array, (mask.height, mask.width, 4))[..., 0]
#     points, labels = lidar.from_depth(depth_array, mask_array, camera, image_coords=precomputed_coords)
#     points = points.astype(np.float32)
#     np.savez(path, points=points, labels=labels)


def main(folder_name):
    start_cc = time.time()
    
    try:
        ### start carla world
        # carla_proc = subprocess.Popen(['./CarlaUE4.sh', '-RenderOffScreen'], cwd=cfg.path_carla, preexec_fn=os.setsid) 
        # carla_proc = subprocess.Popen('./CarlaUE4.sh', cwd=cfg.path_carla, preexec_fn=os.setsid) 
        # time.sleep(10)
        
        client = carla.Client('localhost', 2000, worker_threads=1)
        client.set_timeout(60.0)
        world = client.get_world()
    
        cur_dir = os.path.dirname(__file__)
        path = os.path.join(cur_dir, 'output', folder_name)
        
        sc = SaveContext(path)

        camera_bp, segm_bp, lidar_bp, depth_bp = sensor_settings(world)

        for i, scene in enumerate(sorted(os.listdir(path))):
            recorded_files = get_recordings(scene, path)
            if not recorded_files:
                print('no recorded file available')
                exit(1)
            for file in recorded_files:
                sensor_list = []
                ### get ego id
                if not os.path.exists(os.path.join(path, scene, '00_log', 'context.json')):
                    with open(os.path.join(path, scene, '00_log', 'ego_id.txt')) as f:
                        lines = f.readlines()
                        ego_id = int(lines[0])
                else:
                    with open(os.path.join(path, scene, '00_log', 'context.json')) as f:
                        json_data = json.load(f)
                        ego_id = json_data['ego_vehicle']['id']
                
                path_rec_file = os.path.join(path, scene, '00_log', file)
                considered_time_period = 9
                client.replay_file(path_rec_file, -considered_time_period, 0, ego_id) # (file, start [s], duration [s], actor id)
                time.sleep(1)
                ### load the weather settings
                load_weather(world, path, scene)
                time.sleep(1)
                               

                vehicle = world.get_actor(ego_id)
                transform = carla.Transform(carla.Location(x=1.6, z=1.7))
                transform_lidar = transform
                
                transform_bird = carla.Transform(carla.Location(x=0.0, z=35.0), carla.Rotation(pitch=-90))

                camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
                sensor_list.append(camera)

                segm = world.spawn_actor(segm_bp, transform, attach_to=vehicle)
                sensor_list.append(segm)

                lidar = world.spawn_actor(lidar_bp, transform_lidar, attach_to=vehicle)
                sensor_list.append(lidar)

                depth = world.spawn_actor(depth_bp, transform, attach_to=vehicle)
                sensor_list.append(depth)

                ### inference - usable with CARLA 0.9.13+ ###
                # inst = world.spawn_actor(inst_bp, transform, attach_to=vehicle)
                # sensor_list.append(inst)

                ### radar ###
                # transform_radar = transform
                # radar = world.spawn_actor(radar_bp, transform_radar, attach_to=vehicle)
                # sensor_list.append(radar)

                
                ### birds-eye-view ###
                bird = world.spawn_actor(camera_bp, transform_bird, 
                                            attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
                sensor_list.append(bird)
                bird_sem = world.spawn_actor(segm_bp, transform_bird, 
                                            attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
                sensor_list.append(bird_sem)

                time.sleep(1)
                sc.sensor_settings(sensor_list, transform)
                sc.sensor_locations(world, sensor_list)
                sc.save_json_post(scene)
                
                #------- initializing bboxes and radar_data---------
                # bb = BoundingBoxes.BoundingBoxes(world, depth, width, height)
                # radar_data = rd.SensorData()
                #---------------------------------------------------

                fps_stat = 1 #--> changes not the framerate
                fps_factor = 1
                last_loc = {}
                radius = 15
                frame = 1  
                start = time.time()
                delete_first_10 = False # needed for inference

                ### some settings for recording ###
                client.set_replayer_time_factor(fps_factor) #--> changes the framerate !!!
                # 1 --> normal speed, 2 --> double speed, 0.5 --> 1/2 speed
                # 10 --> of 1fps is needed
                
                with CarlaSyncMode(world, camera, segm, lidar, depth, bird, bird_sem, fps=fps_stat) as sync_mode:
                    while True:
                        if frame == 5 and delete_first_10: # overrides the first 10 images, because the first ones are damaged
                            frame =1
                            delete_first_10 = False
                        snapshot, image_rgb, image_semseg, lidar_pc, image_depth, img_bird, img_bird_segm= sync_mode.tick(timeout=2.0) #--> timeout: Dauer die gewartet werden soll, dass der Sensor Daten sendet
                        image_rgb.save_to_disk(os.path.join(path, scene, '01_cam', file.split('.')[0], '{0:05d}.png'.format(frame)))
                        image_semseg.save_to_disk(os.path.join(path, scene, '02_semseg_cs', file.split('.')[0], '{0:05d}.png'.format(frame)), carla.ColorConverter.CityScapesPalette)
                        image_semseg.save_to_disk(os.path.join(path, scene, '02_semseg_raw', file.split('.')[0], '{0:05d}.png'.format(frame)))
                        # img_inst.save_to_disk(os.path.join(path, scene, '03_inseg_raw', file.split('.')[0], '{0:05d}.png'.format(frame)))
                        # img_inst.save_to_disk(os.path.join(path, scene, '03_inseg_cs', file.split('.')[0], '{0:05d}.png'.format(frame)), carla.ColorConverter.CityScapesPalette)
                        # lidar_save(os.path.join(path, scene, '04_lidar', file.split('.')[0], '{0:05d}.npz'.format(frame)), image_depth, image_semseg)
                        # lidar_pc.save_to_disk(os.path.join(path, scene, '04_lidar', '{0:05d}'.format(frame)))
                        image_depth.save_to_disk(os.path.join(path, scene, '06_depth_raw', file.split('.')[0], '{0:05d}.png'.format(frame)))
                        # image_depth.save_to_disk(os.path.join(path, scene, '06_depth_log', file.split('.')[0], '{0:05d}.png'.format(frame)), carla.ColorConverter.LogarithmicDepth)
                        img_bird.save_to_disk(os.path.join(path, scene, '12_bird', '01_cam', file.split('.')[0], '{0:05d}.png'.format(frame)))
                        img_bird_segm.save_to_disk(os.path.join(path, scene, '12_bird', '02_semseg_cs', file.split('.')[0], '{0:05d}.png'.format(frame)), carla.ColorConverter.CityScapesPalette)
                        #------------------ radar --------------------------------------------------
                        # radar_data.collect(radar_pc)
                        # Process each Frame
                        # for radar_measurement in radar_data:
                        #     points = rd.polar_to_cartesian(radar_measurement, transform_radar.location)
                        #     rd.save_to_disk(points, os.path.join(path, scene, '05_radar', '{0:05d}.ply'.format(frame))) 
                        #     radar_data.clear()       
                        #----------------------------------------------------------------------------

                        # bboxes = bb.on_tick(snapshot = snapshot, image_semseg = image_semseg, image_depth = image_depth)
                        # sc.collect_trajectories(vehicle, snapshot, world, fps = fps_factor/10, bboxes = bboxes, radius = radius) # maybe fps needs to be adjusted
                        # with open(os.path.join(path, scene, '07_bboxes', '{0:05d}.json'.format(frame)), 'w') as file:
                        #     json.dump(bboxes[0], file, indent = 2)

                        if frame > 29: # save 30 images --> 10fps
                            sc.save_trajectories_json(scene, file.split('.')[0])
                            ### move folder to "done" folder
                            path_tmp = path.split(f'/{folder_name}')[0]
                            shutil.move(os.path.join(path, scene), os.path.join(path_tmp, 'done'))
                            break
                        frame +=1
                        # last_loc = save_csv_post(world.get_actor(ego_id), world.get_actors(), radius, frame, start, last_loc, path)
                        # frame += 1
                        # get_snapshot_vehicles(world, path, str(image_rgb.frame))
                        # image_depth.save_to_disk('output/depth-{0:06d}.png'.format(image_depth.frame), carla.ColorConverter.LogarithmicDepth)

                # carla_proc, client, world = restart_carla_world(carla_proc, sensor_list)
        
        # kill_carla_world(carla_proc, sensor_list)
    
    finally:
        # kill_carla_world(carla_proc, sensor_list)
        print("Finished")
        print(f'total time: {time.time()-start_cc:.2f}s = {(time.time()-start_cc)/60:.2f}min = {(time.time()-start_cc)/3600:.2f}h')

if __name__ == '__main__':
    try:
        list_rec = ['recordings']
        for folder_name in list_rec:
            main(folder_name)
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')