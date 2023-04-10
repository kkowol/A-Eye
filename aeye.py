from tkinter import Tk, Canvas, BooleanVar, StringVar, W, E, CENTER, IntVar
from tkinter import Label, Button, Checkbutton, Entry, Radiobutton
import os
import subprocess
import signal
import time
import sys
# sys.path.append("../config") # Adds higher directory to python modules path.
import config as cfg


class CornerCaseGen():
    def __init__(self):
        """
        start with the experimental setup
        """
        self.width = 600
        self.height = 300
        self.window = Tk()
        self.window.title('Setup for Corner Case generation')
        self.window.geometry('800x600')

        self.canvas = Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(rowspan=25, columnspan=5 )

        
    def gui(self):
        #--------------------------- actors --------------------------------
        Label(self.window, text='ACTORS: ').grid(row=1, column=1, sticky=W)
        self.nr_veh= StringVar(); self.nr_ped = StringVar()

        Label(self.window, text='vehicles: ').grid(row=2, column=1, sticky=E)
        ent_1 = Entry(self.window, width=5, textvariable=self.nr_veh)
        ent_1.grid(row=2, column=2, sticky=W)
        ent_1.insert(0,'50')

        Label(self.window, text='pedestrians: ').grid(row=3, column=1, sticky=E)
        ent_2 = Entry(self.window, width=5, textvariable=self.nr_ped)
        ent_2.grid(row=3, column=2, sticky=W)
        ent_2.insert(0, '150')
        #--------------------------- weather -------------------------------
        Label(self.window, text='WEATHER: ').grid(row=5, column=2, sticky=W)        
        self.weather= StringVar()
        weather_default = Radiobutton(self.window, text='clear', value="clear", variable=self.weather)
        weather_default.grid(row=6, column=1, sticky=W)
        weather_default.select()
        Radiobutton(self.window, text='rain', value="rain", variable=self.weather)\
            .grid(row=6, column=2, sticky=W)
        Radiobutton(self.window, text='fog', value="fog", variable=self.weather)\
            .grid(row=6, column=3, sticky=W)
        Radiobutton(self.window, text='night', value="night", variable=self.weather)\
            .grid(row=6, column=4, sticky=W)

        #--------------------------- map -----------------------------------
        Label(self.window, text='MAP: ').grid(row=8, column=1, sticky=W)
        #self.map=IntVar()
        self.map = StringVar()
        Radiobutton(self.window, text='basic', value="01", variable=self.map)\
            .grid(row=10, column=1, sticky=W)
        Radiobutton(self.window, text='small', value="02", variable=self.map)\
            .grid(row=10, column=2, sticky=W)
        rb_default = Radiobutton(self.window, text='complex', value="03", variable=self.map)
        rb_default.grid(row=10, column=3, sticky=W)
        rb_default.select()
        Radiobutton(self.window, text='loop', value="04", variable=self.map)\
            .grid(row=11, column=1, sticky=W)
        Radiobutton(self.window, text='multiple lanes', value="05", variable=self.map)\
            .grid(row=11, column=2, sticky=W)
        Radiobutton(self.window, text='highway', value="06", variable=self.map)\
            .grid(row=11, column=3, sticky=W)
        Radiobutton(self.window, text='rural', value="07", variable=self.map)\
            .grid(row=10, column=4, sticky=W)
        Radiobutton(self.window, text='hd', value="10HD", variable=self.map)\
            .grid(row=11, column=4, sticky=W)
        #--------------------------- drivers -----------------------------------
        Label(self.window, text='DRIVERS: ').grid(row=12, column=1, sticky=W)
        Label(self.window, text='semantic driver: ').grid(row=13, column=1, sticky=W)

        self.name_semseg= StringVar()
        Radiobutton(self.window, text='NameA', value='NameA', variable=self.name_semseg)\
            .grid(row=14, column=1, sticky=W)
        Radiobutton(self.window, text='NameB', value='NameB', variable=self.name_semseg)\
            .grid(row=15, column=1, sticky=W)
        Radiobutton(self.window, text='NameC', value='NameC', variable=self.name_semseg)\
            .grid(row=16, column=1, sticky=W)
        rb_guest = Radiobutton(self.window, text='Guest', value='Guest', variable=self.name_semseg)
        rb_guest.grid(row=17, column=1, sticky=W)
        rb_guest.select()
        
        Label(self.window, text='safety driver: ').grid(row=13, column=3, sticky=W)
        
        self.name_safety= StringVar()
        Radiobutton(self.window, text='NameA', value='NameA', variable=self.name_safety)\
            .grid(row=14, column=3, sticky=W)
        Radiobutton(self.window, text='NameB', value='NameB', variable=self.name_safety)\
            .grid(row=15, column=3, sticky=W)
        Radiobutton(self.window, text='NameC', value='NameC', variable=self.name_safety)\
            .grid(row=16, column=3, sticky=W)
        rb_guest = Radiobutton(self.window, text='Guest', value='Guest', variable=self.name_safety)
        rb_guest.grid(row=17, column=3, sticky=W)
        rb_guest.select()


        Label(self.window, text='Save Images: ').grid(row=22, column=1, sticky=W)
        self.save_inf=BooleanVar()
        self.save_inf.set(True)
        Checkbutton(self.window, variable=self.save_inf, onvalue=1)\
            .grid(row=22, column=2, sticky=W)
        Button(self.window, text='Start', command=self.start_carla, padx=50, pady=25).grid(row=25, column=1, sticky=W)
        Button(self.window, text='Close', command=self.close, padx=50, pady=25).grid(row=25, column=3, sticky=W)

        self.window.mainloop()

    def close(self):
        """
        the window will be closed and the scene will be continued without saving or deletion
        """
        self.window.destroy()


    def start_carla(self):
        """
        starting procedure:
        1. CARLA world starting
        2. vehicle and pedestrians spawning
        3. ego car client (the main manual control script) starts
        4. safety driver client starts
        """
        arg_map, arg_semseg, arg_safety, arg_weather, arg_save_inf = self.get_args()
        self.window.destroy()
        
        ### 1. CARLA
        #start the carla server as subprocess and associate an id to it
        carla_proc = subprocess.Popen(['./CarlaUE4.sh'], cwd=cfg.path_carla, preexec_fn=os.setsid) 
        time.sleep(10)

        # ### start map 
        map_proc = subprocess.Popen(['./config.py', '--map', f'Town{arg_map}'], cwd=os.path.join(cfg.path_carla, 'PythonAPI', 'util'), preexec_fn=os.setsid)
        map_proc.wait()
        time.sleep(5)

        time.sleep(1)
        ### 2. spawning
        if cfg.carla_version == 10:
            spawn_proc = subprocess.Popen(['./utils/spawn_npc.py', f'-n{self.nr_veh.get()}', f'-w{self.nr_ped.get()}'], cwd=cfg.path_buw, preexec_fn=os.setsid) # CARLA10
        else:
            spawn_proc = subprocess.Popen(['./utils/generate_traffic.py', f'-n{self.nr_veh.get()}', f'-w{self.nr_ped.get()}'], cwd=cfg.path_buw, preexec_fn=os.setsid) # CARLA13
        # spawn_proc.wait()
        time.sleep(15)
        
        ### 3. manual control semseg
        if arg_save_inf: 
            semseg_proc = subprocess.Popen(['./control.py', '-ccgm', f'-s{arg_semseg}', f'-f{arg_safety}', f'-w{arg_weather}', '-images'], cwd=cfg.path_buw, preexec_fn=os.setsid)
        else:
            semseg_proc = subprocess.Popen(['./control.py', '-ccgm', f'-s{arg_semseg}', f'-f{arg_safety}', f'-w{arg_weather}'], cwd=cfg.path_buw, preexec_fn=os.setsid)
        time.sleep(25)
        ### 4. safety driver
        safety_proc = subprocess.Popen('./safety_driver.py', cwd=cfg.path_buw, preexec_fn=os.setsid)
        
        semseg_proc.wait() # wait until manual control is closed
    
        ### kill the carla process and grab the exit status to avoid a zombie process
        os.killpg(os.getpgid(safety_proc.pid), signal.SIGKILL) # SIGKILL is kill -9 (needed to kill the safety driver process)
        outs, errs = safety_proc.communicate()
        os.killpg(os.getpgid(spawn_proc.pid), signal.SIGKILL)
        outs, errs = safety_proc.communicate()
        os.killpg(os.getpgid(carla_proc.pid), signal.SIGTERM)
        outs, errs = carla_proc.communicate()


    def get_args(self):
        arg_map = self.map.get()
        arg_semseg = self.name_semseg.get()
        arg_safety = self.name_safety.get()
        arg_weather = self.weather.get()
        arg_save_inf = self.save_inf.get()
        return arg_map, arg_semseg, arg_safety, arg_weather, arg_save_inf


def main():
    ccg=CornerCaseGen()
    ccg.gui()


if __name__ == '__main__':

    main()