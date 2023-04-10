import glob
import os
import sys
import random
import config as cfg
import json
from utils.tools import get_folder_name
from datetime import datetime

try:
    sys.path.append(glob.glob(cfg.path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class Weather:
    def __init__(self, client, preset):
        self.client = client
        self.world = self.client.get_world()
        # self.weather_preset = preset
        #fixed parameters:
        self.sun_angle_azi   = 0.0
        self.wind            = 0.0
        self.fog_concentration = 0.10000000149011612
        self.fog_distance    = 0.75
        self.set_weather(preset)

    def set_weather(self, preset):
        self.weather_preset = preset
        if preset=='clear':
            self.sun_angle_alt   = 90.0
            self.rain            = 0.0
            self.wetness         = 0.0
            self.puddles         = 0.0
            self.clouds          = 0.0
            self.fog_density     = 0.0
        
        if preset=='rain':
            self.sun_angle_alt   = 90.0
            self.rain            = 100.0
            self.wetness         = 80.0
            self.puddles         = 50.0
            self.clouds          = 30.0
            self.fog_density     = 0.0

        if preset=='fog':
            self.sun_angle_alt   = 90.0
            self.rain            = 0.0
            self.wetness         = 0.0
            self.puddles         = 0.0
            self.clouds          = 50.0
            self.fog_density     = 50.0

        if preset=='night':
            self.sun_angle_alt   = -90.0
            self.rain            = 0.0
            self.wetness         = 0.0
            self.puddles         = 0.0
            self.clouds          = 0.0
            self.fog_density     = 0.0
            

        weather = carla.WeatherParameters(
            cloudiness              = self.clouds,
            precipitation           = self.rain,
            precipitation_deposits  = self.puddles,
            wetness                 = self.wetness,
            sun_altitude_angle      = self.sun_angle_alt,
            sun_azimuth_angle       = self.sun_angle_azi,
            wind_intensity          = self.wind,
            fog_density             = self.fog_density,
            fog_distance            = self.fog_distance,
            fog_falloff             = self.fog_concentration,
        )
        self.world.set_weather(weather)
        
    def set_weather4campaign(self, preset):
        self.weather_preset = preset
        if preset=='clear':
            self.sun_angle_alt   = random.randint(0.0, 90.0)
            self.rain            = 0.0
            self.wetness         = 0.0
            self.puddles         = 0.0
            self.clouds          = random.randint(0.0, 40.0)
            self.fog_density     = 0.0
        
        if preset=='rain':
            self.sun_angle_alt   = random.randint(0.0, 90.0)
            self.rain            = random.randint(70.0, 100.0)
            self.wetness         = self.rain
            self.puddles         = random.randint(20.0, 50.0)
            self.clouds          = random.randint(abs(self.wetness - 10.0), self.wetness)
            self.fog_density     = 0.0

        if preset=='fog':
            self.sun_angle_alt   = random.randint(0.0, 90.0)
            self.rain            = 0.0
            self.wetness         = 0.0
            self.puddles         = 0.0
            self.clouds          = random.randint(40.0, 60.0)
            self.fog_density     = random.randint(50.0, 100.0)

        if preset=='night':
            self.sun_angle_alt   = -1*random.randint(0.0, 90.0)
            self.rain            = 0.0
            self.wetness         = 0.0
            self.puddles         = 0.0
            self.clouds          = random.randint(0.0, 40.0)
            self.fog_density     = 0.0
            

        weather = carla.WeatherParameters(
            cloudiness              = self.clouds,
            precipitation           = self.rain,
            precipitation_deposits  = self.puddles,
            wetness                 = self.wetness,
            sun_altitude_angle      = self.sun_angle_alt,
            sun_azimuth_angle       = self.sun_angle_azi,
            wind_intensity          = self.wind,
            fog_density             = self.fog_density,
            fog_distance            = self.fog_distance,
            fog_falloff             = self.fog_concentration,
        )
        self.world.set_weather(weather)
    
    def save_weather(self):
        setup = {}
        now = datetime.now()
        setup['date and time']      = now.strftime("%d/%m/%Y %H:%M:%S")
        setup['map']                = self.world.get_map().name
        setup['weather']            = {
                'cloudiness':           self.world.get_weather().cloudiness, 
                'rain intensity':       self.world.get_weather().precipitation, 
                'puddle coverage':      self.world.get_weather().precipitation_deposits,
                'wetness':              self.world.get_weather().wetness,
                'sun altitude':         self.world.get_weather().sun_altitude_angle,
                'sun azimuth':          self.world.get_weather().sun_azimuth_angle,
                'wind intensity':       self.world.get_weather().wind_intensity, 
                'fog density':          self.world.get_weather().fog_density,
                'fog distance':         self.world.get_weather().fog_distance,
                'fog concentration':    self.world.get_weather().fog_falloff,
        }
        if self.weather_preset:
            setup['weather_preset']    = self.weather_preset
        
        with open(os.path.join(get_folder_name(), '00_log', 'weather.json'), 'w') as f:
            json.dump(setup, f)