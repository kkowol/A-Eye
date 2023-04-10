import time
import json
import os
import csv
import glob
import os
import sys
import config as cfg
from datetime import datetime
from utils.carlaworld import count_vehicles_and_walkers
from utils.tools import get_folder_name
from utils.rec import Recording
from utils.carla_dataloader import Carla
from tkinter import Label, Button, Tk, Canvas, BooleanVar, Checkbutton, StringVar, Entry
# from tkinter.constants import BOTH, LEFT, TOP


class CheckCornerCase():
    def __init__(self, world, args, client, td, qrecording, weather):
        """
        check if the situation was a corner case
        :param td:          travel distance
        :param qrecording:  image queue
        :param weather:     weather preset (clear, rain, fog or night)
        """
        self.width = 600
        self.height = 300
        self.td = td
        self.qrecording = qrecording # integration of inference recording
        self.weather = weather
        now = datetime.now()
        nr_vehicles, nr_walkers = count_vehicles_and_walkers(world.world)
        self.rec = Recording()
        self.i_rec = 1
        self.client = client
        self.rec.start(self.client, self.i_rec)
        self.start = time.time()
        self.ego_car_loc= td.travelled_distance

        setup = {}
        setup['date and time']      = now.strftime("%d/%m/%Y %H:%M:%S")
        setup['semantic driver']    = args.semseg_name
        setup['safety driver']      = args.safety_name
        setup['map']                = world.world.get_map().name
        setup['number vehicles']    = nr_vehicles
        setup['number walkers']     = nr_walkers
        setup['weather_preset']     = self.weather.weather_preset
        setup['weather']            = {
                'cloudiness':           world.world.get_weather().cloudiness, 
                'rain intensity':       world.world.get_weather().precipitation, 
                'puddle coverage':      world.world.get_weather().precipitation_deposits,
                'wetness':              world.world.get_weather().wetness,
                'sun altitude':         world.world.get_weather().sun_altitude_angle,
                'sun azimuth':          world.world.get_weather().sun_azimuth_angle,
                'wind intensity':       world.world.get_weather().wind_intensity, 
                'fog concentration':    world.world.get_weather().fog_falloff,
                'fog density':          world.world.get_weather().fog_density,
                'fog distance':         world.world.get_weather().fog_distance,
                # 'fog light':            world.world.get_weather().scattering_intensity,
                # 'mie':                  world.world.get_weather().mie_scattering_scale,
                # 'rayleigh':             world.world.get_weather().rayleigh_scattering_scale,
        }
        setup['network']            = {
                'name':             cfg.model_name,
                'number classes':   Carla.num_train_ids
        }
        with open(os.path.join(get_folder_name(), '09_corner_cases', 'experimental_setup.json'), "w") as f:
            json.dump(setup, f)


    def gui(self, trigger, timer, sensor_name, ego_car_loc):
        """
        :param reason:  reason for triggering the CC ("brake" or "steer")
        :param client:  CARLA client
        :param rec:     recording object 
        :param i_rec:   counter for records (int)
        """
        self.trigger        = trigger
        self.timer          = timer
        self.sensor_name    = sensor_name
        self.ego_car_loc    = ego_car_loc
        if self.qrecording is not None: self.qrecording.wait_on()
        # time.sleep(2)
        self.rec.stop(self.client)

        self.window = Tk()
        self.window.title('POTENTIAL CORNER CASE')
        self.window.geometry(f'{self.width}x{self.height}')

        canvas = Canvas(self.window, width=self.width, height=self.height)
        canvas.grid(rowspan=20, columnspan=5 )

        Label(self.window, text='Was this a Corner Case?').grid(row=4, column=2)
        buttons =  [['Yes', self.cc_true, 10, 1], 
                    ['No', self.cc_false, 10, 3], 
                    ['Close', self.close, 19, 2]]
        for button, f, row, col in buttons:
            Button(self.window, text=button, command=f, padx=50, pady=15).grid(row=row, column=col) 

        self.c_w=BooleanVar(); self.c_v=BooleanVar()
        self.c_t=BooleanVar(); self.c_b=BooleanVar()
        self.t_c=StringVar()
        Checkbutton(self.window, text='walker overlooked', variable=self.c_w, onvalue=1, offvalue=0)\
        .grid(row=6, column=1)
        Checkbutton(self.window, text='vehicle overlooked', variable=self.c_v, onvalue=1, offvalue=0)\
        .grid(row=8, column=1)
        Checkbutton(self.window, text='disregard the traffic rules', variable=self.c_t, onvalue=1, offvalue=0)\
        .grid(row=6, column=2)
        Checkbutton(self.window, text='boredom intervention', variable=self.c_b, onvalue=1, offvalue=0)\
        .grid(row=8, column=2)
        Label(self.window, text='comment: ').grid(row=9, column=1)
        Entry(self.window, textvariable=self.t_c).grid(row=9, column=2)        
        self.window.mainloop()

        return self.timer
    
    def close(self):
        """
        the window will be closed and the scene will be continued without saving or deletion
        """
        if self.qrecording is not None: self.qrecording.wait_off()
        self.window.destroy()

    def cc_true(self):
        """
        save Corner Case
        """
        self.save_cc_log()
        # self.rec.stop(self.client)
        print('Corner Case saved ...')
        
        self.i_rec +=1
        self.rec.start(self.client, self.i_rec)
        self.close()
        self.timer = time.time()    # start new timing
        self.td.reset()
        if self.qrecording is not None: 
            self.qrecording.retrieve(cc_true = True)
            time.sleep(6)
        if self.qrecording is not None: self.qrecording.wait_off()

    def cc_false(self):
        """
        delete Corner Case
        """
        # self.rec.stop(self.client)
        print('Corner Case not saved ...')
        if self.qrecording is not None: self.qrecording.wait_off()
        self.delete_recording()
        self.rec.start(self.client, self.i_rec)
        self.close()
        

    def save_cc_log(self):
        """

        """
        reason = None
        if self.c_v.get(): reason='vehicle overlooked'
        if self.c_w.get(): reason='walker overlooked'
        if self.c_t.get(): reason='disregard the traffic rules'
        if self.c_b.get(): reason='boredom intervention'
        if reason is None:
            reason = '-'
        fields = [
            'time until cc',
            'driven m until cc',
            'reason for cc', 
            'triggering by',
            'network',
            'weather', 
            'comments'
        ]
        rows = []
        row = [    
            str(time.time() - self.timer), 
            self.ego_car_loc,
            reason, 
            self.trigger,
            self.sensor_name,
            self.weather.weather_preset,
            self.t_c.get()
        ]

        rows.append(row)   
        path = get_folder_name()

        with open(os.path.join(path, '09_corner_cases', 'cc.csv'), 'a') as file:
            csvwriter = csv.writer(file)
            if not os.path.exists(os.path.join(path, '09_corner_cases', 'cc.csv')) or os.stat(os.path.join(path, '09_corner_cases', 'cc.csv')).st_size == 0:
                csvwriter.writerow(fields)
            csvwriter.writerows(rows)


    def delete_recording(self):
        """
        if the scene should not be saved, then delete the recording to save memory
        """
        path = os.path.join( get_folder_name(), '00_log')
        os.remove(os.path.join(path, f'scene_recording_{self.i_rec}.log'))
