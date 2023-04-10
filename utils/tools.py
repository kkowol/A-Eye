import torch
import os
import sys
sys.path.append(os.getcwd())
import config as cfg


class TimeMeasurement:
    def __init__(self):
        """Time measurement with CUDA"""
        self.starter, self.ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
    def start(self):
        self.starter.record()
    def end(self):
        self.ender.record()
        torch.cuda.synchronize()
        print(str(self.starter.elapsed_time(self.ender)/1000)) 

# def find_carla_module():
#     try:
#         sys.path.append(glob.glob(cfg.path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
#             sys.version_info.major,
#             sys.version_info.minor,
#             'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
#     except IndexError:
#         pass
#     import carla

def get_folder_name():
    """
    gets the output folder name
    """
    path = os.path.join(os.getcwd(), 'output')
    nr_files = len(os.listdir(path))
    folder_name = os.path.join(path, '{}_%04i'.format(cfg.name_out_folder)) % nr_files
    return folder_name


def get_model_name(ckpt_path):
    """
    :param ckpt_path: checkpoint path as string
    needed name format: 
    """
    tmp = ckpt_path.split('/')[-1].split('_')
    return tmp[0] + ' ' +tmp[1] + ' ' +tmp[2]


def output_folders_data_generator():
    """
    create output folders for data generator script with counting the existing folders
    """
    path = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(path): os.mkdir(path)
    nr_files = len(os.listdir(path))
    nr_files += 1
    path = os.path.join(path, '{}_%04i'.format(cfg.name_out_folder)) % nr_files
    os.mkdir(path)
    os.mkdir(os.path.join(path,'00_log'))
    os.mkdir(os.path.join(path,'01_cam'))
    os.mkdir(os.path.join(path,'02_semseg_raw'))
    os.mkdir(os.path.join(path,'02_semseg_cs'))
    os.mkdir(os.path.join(path,'03_inseg_raw'))
    os.mkdir(os.path.join(path,'03_inseg_cs'))
    os.mkdir(os.path.join(path,'04_lidar'))
    os.mkdir(os.path.join(path,'05_radar'))
    os.mkdir(os.path.join(path,'06_depth_raw'))
    os.mkdir(os.path.join(path,'06_depth_log'))
    os.mkdir(os.path.join(path,'07_bboxes'))
    os.mkdir(os.path.join(path,'08_trajectory'))
    os.mkdir(os.path.join(path,'09_corner_cases'))
    os.mkdir(os.path.join(path,'10_inference'))
    return path