import json
import os
import sys
import glob

import glob
import os
import sys
import config as cfg 

try:
    sys.path.append(glob.glob(cfg.path_egg_file + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class StationaryActor:
    def __init__(self, actor_id=None, actor_bbox=None, actor_transform=None, from_dict=None):
        self.id = actor_id
        self.bounding_box = actor_bbox
        self.transform = actor_transform

    def dict(self):
        return {
            'id': self.id,
            'extent': (self.bounding_box.extent.x, self.bounding_box.extent.y, self.bounding_box.extent.z),
            'location': (self.transform.location.x, self.transform.location.y, self.transform.location.z),
            'rotation':  (self.transform.rotation.pitch, self.transform.rotation.yaw, self.transform.rotation.roll)
        }

    def __eq__(self, other):
        return self.bounding_box == other.bounding_box and self.transform == other.transform


def load_stationary_actors(carla_map, folder_path='./stationary_actors'):
    stationary_actors = []
    map_name = carla_map.name.split('/')[2] # needed for CARLA13

    # with open(os.path.join(folder_path, f'{carla_map.name}.json')) as rf:
    with open(os.path.join(folder_path, f'{map_name}.json')) as rf:
        for data in json.load(rf):
            transform = carla.Transform(
                carla.Location(*data.get('location')),
                carla.Rotation(*data.get('rotation'))
            )

            bbox = carla.BoundingBox(
                carla.Location(0, 0, 0),
                carla.Vector3D(*data.get('extent'))
            )
            bbox.rotation = transform.rotation

            stationary_actors.append(
                StationaryActor(
                    actor_id=data.get('id'),
                    actor_bbox=bbox,
                    actor_transform=transform
                )
            )

    return stationary_actors


def extract_bounding_boxes():
    """
        Save all stationary bounding boxes in a file
        Only run this step at fresh startup of carla without any additional actors!
    """

    # try:
    #     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    #         sys.version_info.major,
    #         sys.version_info.minor,
    #         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    # except IndexError:
    #     print('Could not find module')

    # import init_carla_module
    # import carla

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    debug = world.debug
    spectator = world.get_spectator()
    map_name = world.get_map().name
    map_name = map_name.split('/')[2]

    print(f'Working on {map_name}!')

    # bounding boxes are given in world coordinates instead of local coordinates
    # converting to local coordinates
    stationary_actors = []
    for i, vehicle_bbs in enumerate(world.get_level_bbs(actor_type=carla.CityObjectLabel.Vehicles)):
        bbox = carla.BoundingBox(carla.Location(0, 0, 0), vehicle_bbs.extent)
        transform = carla.Transform(vehicle_bbs.location, vehicle_bbs.rotation)
        stationary_actors.append(StationaryActor(i, bbox, transform))

    # Some actors got two bounding boxes (bug?)
    # filter out the one that is nested inside and only use the bigger one
    nested_actors = []
    for actor in stationary_actors:
        for other_actor in stationary_actors:
            if actor is not other_actor:
                # if all points are inside, consider it to be a nested bbox
                for point in other_actor.bounding_box.get_world_vertices(other_actor.transform):
                    if not actor.bounding_box.contains(point, actor.transform):
                        break
                else:
                    nested_actors.append(other_actor)

    # Only these will be used later
    reduced_actors = [actor for actor in stationary_actors if actor not in nested_actors]

    print('# World Actors:', len(stationary_actors))
    print('# Nested Actors:', len(nested_actors))
    print('# Reduced Actors:', len(reduced_actors))

    # Additional checking
    choice = input('Use Spectator to check individually? (y/N)')

    if choice.upper() == 'Y':
        for actor in stationary_actors:
            color = carla.Color(255, 0, 0) if actor in reduced_actors else carla.Color(0, 0, 255)
            bbox = carla.BoundingBox(actor.transform.location, actor.bounding_box.extent)
            debug.draw_box(bbox, actor.transform.rotation, color=color, life_time=300.0)

        for actor in stationary_actors:
            bbox = carla.BoundingBox(actor.transform.location, actor.bounding_box.extent)
            debug.draw_box(bbox, actor.transform.rotation, color=carla.Color(0, 255, 0), life_time=0.5)
            spectator.set_transform(carla.Transform(actor.transform.location + carla.Location(z=5), carla.Rotation(pitch=-90)))
            input(f'{actor.id} Actor')

    with open(f'stationary_actors/{map_name}.json', 'w') as wf:
        print('Saving to file...')
        data = [actor.dict() for actor in reduced_actors]
        json.dump(data, wf, indent=2)


if __name__ == '__main__':
    extract_bounding_boxes()

