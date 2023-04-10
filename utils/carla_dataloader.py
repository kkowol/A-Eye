import os
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image


class Carla(Dataset):
    """`
    selfmade CARLA Dataset
    Labels based on https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera

    colors in BGR!
    """
    CarlaClass = namedtuple( 'CarlaClass' , ['name', 'id', 'train_id', 'ignore_in_eval', 'color'])
    labels = [
        CarlaClass(  'Unlabeled'           ,  0 ,  255  , True, (  0,   0,   0) ),
        CarlaClass(  'Building'            ,  1 ,  0  , False, ( 70,  70,  70) ),
        CarlaClass(  'Fence'               ,  2 ,  255  , True, (100,  40,  40) ),
        CarlaClass(  'Other'               ,  3 ,  255  , True, ( 55,  90,  80) ),
        CarlaClass(  'Pedestrian'          ,  4 ,  1  , False, (220,  20,  60) ),
        CarlaClass(  'Pole'                ,  5 ,  2  , False, (153, 153, 153) ),
        CarlaClass(  'RoadLine'            ,  6 ,  3  , False, (157, 234,  50) ),
        CarlaClass(  'Road'                ,  7 ,  4  , False, (128,  64, 128) ),
        CarlaClass(  'SideWalk'            ,  8 ,  5  , False, (244,  35, 232) ),
        CarlaClass(  'Vegetation'          ,  9 ,  6  , False, (107, 142,  35) ),
        CarlaClass(  'Vehicles'            ,  10,  7 , False, (  0,   0, 142) ),
        CarlaClass(  'Wall'                ,  11,  8 , False, (102, 102, 156) ),
        CarlaClass(  'TrafficSign'         ,  12,  9 , False, (220, 220,   0) ),
        CarlaClass(  'Sky'                 ,  13,  10 , False, ( 70, 130, 180) ),
        CarlaClass(  'Ground'              ,  14,  255 , True, ( 81,   0,  81) ),
        CarlaClass(  'Bridge'              ,  15,  11 , False, (150, 100, 100) ),
        CarlaClass(  'RailTrack'           ,  16,  12 , False, (230, 150, 140) ),
        CarlaClass(  'GuardRail'           ,  17,  13 , False, (180, 165, 180) ),
        CarlaClass(  'TrafficLight'        ,  18,  14 , False, (250, 170,  30) ),
        CarlaClass(  'Static'              ,  19,  255 , True, (110, 190, 160) ),
        CarlaClass(  'Dynamic'             ,  20,  255 , True, (170, 120,  50) ),
        CarlaClass(  'Water'               ,  21,  255 , True, ( 45,  60, 150) ),
        CarlaClass(  'Terrain'             ,  22,  15 , False, (145, 170, 100) ),
    ]
    """Normalization parameters"""
    mean    = (0.5371, 0.5335, 0.4979)
    std     = (0.2539, 0.2369, 0.2394)

    """Class weights"""
    class_weights =[    1.00000000e+00, 1.01708435e+01, 1.31733587e+02, 2.08798907e+03, 5.78580198e+02, 
                        7.73639041e+01, 9.83769334e+01, 3.52479418e+00, 1.68213358e+01, 7.96039269e+00, 
                        4.90020924e+01, 6.83690339e+01, 1.59861797e+03, 3.64199494e+00, 4.55834227e+02, 
                        2.81387949e+02, 1.95248090e+02, 8.99032767e+01, 6.87239288e+02, 5.22137755e+02,
                        1.48718568e+03, 1.84858588e+03, 1.58213959e+01
                    ]
    weights = []
    for i in range(len(class_weights)):
        if not labels[i].train_id==255:
            weights.append(class_weights[i])
    

    id2train_id = {label.id: label.train_id for label in labels}

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    
    color_palette_train_ids = list(sum(color_palette_train_ids, ())) # needed for putpalette


    def __init__(self, split, root: str="/home/Raid/datasets/CARLA/", mode: str = "semSeg", target_type: str = "semantic_train_id",
                 transform: Optional[Callable] = None, predictions_root: Optional[str] = None) -> None:
        """
        CARLA dataset loader
        """
        self.root = root                # rgb
        self.split = split
        self.transform = transform
        self.images = []
        self.targets = []
        self.predictions = []

        for i, scene in enumerate(sorted(os.listdir(os.path.join(root, self.split)))):
            for file in sorted(os.listdir(os.path.join(root, self.split, scene, 'RGB'))):
                self.images.append(os.path.join(root, self.split, scene, 'RGB', file))
            for file_sem in sorted(os.listdir(os.path.join(root, self.split, scene, 'semSeg_id'))):
                self.targets.append(os.path.join(root, self.split, scene, 'semSeg_id', file_sem))
            # for file_pred in os.listdir(os.path.join(root, self.split, scene, 'inference')):
                self.predictions.append(os.path.join(root, 'inference', scene, file_sem))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        if self.split in ['train', 'val']:
            target = Image.open(self.targets[index])
        else:
            target = None
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.images)
