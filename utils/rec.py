import os
import queue
import imageio as iio
from utils.tools import get_folder_name
import config as cfg

class Recording:
    def __init__(self):
        self.path_tmp = get_folder_name()
    def start(self, client, i_rec):
        client.start_recorder(os.path.join(self.path_tmp, '00_log', f'scene_recording_{i_rec}.log'), additional_data=True)
    def stop(self, client):
        client.stop_recorder()
    # def save_recorded_time(self, start):
    #     with open(os.path.join(self.path_tmp, '00_log', 'recording_time_seconds.txt'), 'a') as file:
    #             file.write('{}\n'.format(int(time.time()-start)))


class QRecording():
    """
    saves inference and rgb image of last few seconds in queues which can be emptied 
    when a cc occurs
    """
    def __init__(self, fps = 30, seconds_before_cc = 5, record_every_x_frames = 1):
        self.fps = fps
        self.seconds_before_cc = seconds_before_cc
        self.record_every_x_frames = record_every_x_frames
        self.maskqueue = queue.Queue(maxsize = self.seconds_before_cc * self.fps // self.record_every_x_frames + 1)
        self.camqueue =  queue.Queue(maxsize = self.seconds_before_cc * self.fps // self.record_every_x_frames + 1)
        #self.mask = None
        self.frame = 0
        self.qitems = 0
        self.path = get_folder_name()
        self.cc_counter = 0
        self.wait = False

    def add(self, mask, image):
        if not self.wait:
            if self.frame % self.record_every_x_frames == 0:
                if self.qitems < self.maskqueue.maxsize:
                    self.maskqueue.put(mask)
                    self.camqueue.put(image)
                    self.qitems += 1
                else:
                    self.maskqueue.get()
                    self.camqueue.get()
                    self.maskqueue.put(mask)
                    self.camqueue.put(image)
            #self.frame += self.record_every_x_frames
            self.frame += 1

    def wait_on(self):
        self.wait = True

    def wait_off(self):
        self.wait = False

    def retrieve(self, cc_true):
        try: 
            if cc_true:
                self.cc_counter += 1
                infpath = os.path.join(self.path, "10_inference", f"cc_{self.cc_counter}")
                campath = os.path.join(self.path, "01_cam", f"cc_{self.cc_counter}")
                os.mkdir(infpath)
                os.mkdir(campath)
                frame = self.frame # fixed current frame for saving images
        
                for i in range(self.qitems):
                    inf = self.maskqueue.get()
                    cam = self.camqueue.get()
                    iio.v3.imwrite(os.path.join(infpath, f"frame_{frame - self.record_every_x_frames * (self.qitems - i)}.jpg"), inf, plugin="pillow")
                    cam.save_to_disk(os.path.join(campath, f"frame_{frame - self.record_every_x_frames * (self.qitems - i)}.jpg"))
                # self.wait = False
                    
            
            del self.maskqueue
            del self.camqueue
            self.maskqueue = queue.Queue(maxsize = self.seconds_before_cc * self.fps // self.record_every_x_frames + 1)
            self.camqueue =  queue.Queue(maxsize = self.seconds_before_cc * self.fps // self.record_every_x_frames + 1)
            self.qitems = 0
            
        except queue.Empty:
            del self.maskqueue
            del self.camqueue
            self.maskqueue = queue.Queue(maxsize = self.seconds_before_cc * self.fps // self.record_every_x_frames + 1)
            self.camqueue =  queue.Queue(maxsize = self.seconds_before_cc * self.fps // self.record_every_x_frames + 1)
            self.qitems = 0
            
            return