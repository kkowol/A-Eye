import numpy as np
import torch
import os

from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage
from models.fast_scnn import FastSCNN
from models.bisenetv2 import BiSeNetV2
from utils.carla_dataloader import Carla
import config as cfg
import carla

def carla_colorize(arr):
    imc = arr.convert('P')
    imc.putpalette(Carla.color_palette_train_ids)
    return imc.convert('RGB')


def img_enlargement(img, width, height):
    """
    enlarge the input data with factor 3
    comparision of the available filters:
    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
    """
    return img.resize((width*3, height*3))


def output_folders_inference():
    """
    create output folders for inference script with counting the existing folders
    """
    path = os.path.join(os.getcwd(), 'output')
    nr_files = len(os.listdir(path))
    path = os.path.join(path, '{}_%04i'.format(cfg.name_out_folder)) % nr_files
    os.mkdir(path)
    return path


class Inference():
    def __init__(self, ckpt):
        if ckpt:
            self.ckpt = ckpt
            self.torch_transform = Compose([ToTensor(), Normalize(Carla.mean, Carla.std)])
            
            models =['FastSCNN', 'bisenetv2']
            for model in models:
                if model in ckpt:
                    self.model_name = model
            
            if self.model_name == models[0]:
                self.network = FastSCNN(in_channels=3, num_classes=Carla.num_train_ids)
                
            elif self.model_name == models[1]:
                self.network = BiSeNetV2(n_classes=Carla.num_train_ids)
                # self.network = TRTModule()
            self.network.load_state_dict(torch.load(os.path.join(os.getcwd(), 'weights', self.ckpt + '.pth')))
            self.network.cuda().eval()

    def processing(self, image):
        """
        inference function
        """
        #----------- preprocessing -----------
        image.convert(carla.ColorConverter.Raw)     # raw data needed!!!
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] # swap first and third channel (bgr-->rgb)
        img = array.copy()
        #----------- inference -----------
        x = self.torch_transform(img).unsqueeze_(0).cuda() # unsqueeze --> add a channel
        # with torch.no_grad():
        if self.model_name == 'bisenetv2':
            y_trt = self.network(x)[0]
        else:
            y_trt = self.network(x)
        pred = y_trt.argmax(dim=1)[0].cpu()#.numpy()
        pred = ToPILImage()(pred.to(dtype=torch.uint8))
        #----------- unique/converting -----------
        mask = np.array(carla_colorize(pred)) 
        return mask