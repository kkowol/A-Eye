#!/usr/bin/env python
import time
import numpy as np
import cv2

from StationaryActor import StationaryActor, load_stationary_actors

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

def from_buffer(sensor_data):
    image = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
    return np.reshape(image, (sensor_data.height, sensor_data.width, 4))


class BoundingBoxes:
    def __init__(self, world, sensor_depth, width=800, height=600):
        self.world = world
        self.carla_map = world.get_map()
        self.sensor_depth = sensor_depth

        self.width = width
        self.height = height

        # (Intrinsic) (3, 3) K Matrix
        self.k = np.identity(3)
        self.k[0, 2] = self.width / 2.0
        self.k[1, 2] = self.height / 2.0
        self.k[0, 0] = self.k[1, 1] = self.width / (2.0 * np.tan(90.0 * np.pi / 360.0))
        self.k_inv = np.linalg.inv(self.k)

        # Precalculate our matrix to keep away work from the main loop
        base_matrix = np.zeros((width, height, 3))

        # Todo: This is too slow (...and wrong)
        for x in range(width):
            for y in range(height):
                point = np.array([[x], [y], [1]])
                p_3d = np.matmul(self.k_inv, point)

                base_matrix[x][y][0] = p_3d[0]
                base_matrix[x][y][1] = p_3d[1]
                base_matrix[x][y][2] = p_3d[2]

        base_matrix = np.stack(
            [base_matrix[:, :, 0].transpose(), base_matrix[:, :, 1].transpose(), base_matrix[:, :, 2].transpose()],
            axis=2)

        self.reshaped_matrix = base_matrix.reshape(-1, base_matrix.shape[-1])
        self.stationary_actors = load_stationary_actors(self.carla_map)

    def on_tick(self, snapshot, image_semseg, image_depth):
        sensor_transform = snapshot.find(self.sensor_depth.id).get_transform()

        semantic_image = from_buffer(image_semseg)[..., 2]  # Data only in red channel
        depth_information = from_buffer(image_depth)[:, :, :3]

        # Masking Area of Interest
        # cv2.inRange() ~ 25% faster compared to np.where()
        walker_mask = cv2.inRange(semantic_image, 4, 4)
        vehicle_mask = cv2.inRange(semantic_image, 10, 10)

        depth_information_32 = depth_information.astype(np.dtype('uint32'))
        depth_information_32[..., 0] <<= 16
        depth_information_32[..., 1] <<= 8

        # Create array of actual distances for each pixel
        # np.bitwise_or() is ~ 15% faster compared to np.sum()
        distance = np.bitwise_or(depth_information_32[..., 0], depth_information_32[..., 1])
        distance = np.bitwise_or(distance, depth_information_32[..., 2]) * (1000 / 16777215)
        reshaped_distance = distance.flatten()

        # cc_depth = carla.ColorConverter.Depth  # Not good on close distances
        cc_depth = carla.ColorConverter.LogarithmicDepth  # Not good on long distances
        image_depth.convert(cc_depth)

        # Only retrieving actors that were active when this snapshot was taken
        snapshot_actors = {actor.id: actor.get_transform() for actor in snapshot}
        real_actors = self.world.get_actors(actor_ids=list(snapshot_actors.keys()))

        bounding_boxes = {
            'frame': snapshot.frame,
            '3d': [],
            '2d': {
                'annotations': [],
                'image_width': self.width,
                'image_height': self.height
            }
        }

        debug_data = {
            'checked_points': 0,
            'rect_mask': np.zeros(depth_information.shape, dtype='uint8'),
            'time': time.time()
        }

        # Handling vehicles and walkers separately
        for actor_type, mask, color in [
            ('vehicle', vehicle_mask, (0, 0, 255)),
            ('walker', walker_mask, (255, 0, 0))
        ]:

            # [actor, tested_points, actor_transform]
            actors = [[actor, [], snapshot_actors[actor.id]]
                      for actor in real_actors.filter(f'{actor_type}.*')]

            # Vehicles that are placed on the map as static actors don't show up
            # in the Carla.Actors list
            if actor_type == 'vehicle':
                for stationary_actor in self.stationary_actors:
                    actors.append([stationary_actor, [], stationary_actor.transform])

            filtered_depth = cv2.bitwise_and(depth_information, depth_information, mask=mask)
            semseg_edges = cv2.Canny(mask, 6, 6)
            edges = cv2.Canny(filtered_depth, 6, 6)
            edges = cv2.bitwise_or(edges, semseg_edges)

            if len(actors) > 0:
                pos = edges.nonzero()
                reshaped_edges = edges.reshape(-1)
                nonzero_edges = reshaped_edges.nonzero()

                # Calculating relative 3D-Points for every Pixel
                reduced_matrix = np.multiply(
                    self.reshaped_matrix[nonzero_edges].transpose(),
                    reshaped_distance[nonzero_edges]
                )

                ones = np.ones(reduced_matrix.shape[1])
                p4ds = np.array([reduced_matrix[2], reduced_matrix[0], -reduced_matrix[1], ones])
                p3ds = np.matmul(sensor_transform.get_matrix(), p4ds)[:3]

                debug_data['checked_points'] += len(pos[0])

                for x, y, p3d in zip(pos[0], pos[1], p3ds.transpose()):
                    # p3d = location.transform(carla.Vector3D(p3d[2], p3d[0], -p3d[1]))
                    p3d = carla.Vector3D(p3d[0], p3d[1], p3d[2])
                    for actor, valid_points, actor_transform in actors:
                        bbox = carla.BoundingBox(actor.bounding_box.location, actor.bounding_box.extent)

                        if actor_type == 'walker':
                            # Make bbox slightly larger to accommodate big steps
                            bbox.extent += carla.Vector3D(0.25, 0, 0)

                        if bbox.contains(p3d, actor_transform):
                            debug_data['rect_mask'][x][y] = (255, 255, 255)
                            valid_points.append((y, x))
                            break

            # Wrap everything up
            for actor, valid_points, actor_transform in actors:
                bbox = actor.bounding_box
                location = actor_transform.location
                rotation = actor_transform.rotation
                bounding_boxes['3d'].append({
                    'id': actor.id,
                    'class': actor_type,
                    'static': isinstance(actor, StationaryActor),
                    'extent': (bbox.extent.x, bbox.extent.y, bbox.extent.z),
                    'location': (bbox.location.x, bbox.location.y, bbox.location.z),
                    'transform_location': (location.x, location.y, location.z),
                    'transform_rotation': (rotation.pitch, rotation.yaw, rotation.roll)
                })

                if len(valid_points) > 3:
                    bbox = cv2.boundingRect(np.array(valid_points))
                    bounding_boxes['2d']['annotations'].append({
                        'id': actor.id,
                        'class': actor_type,
                        'static': isinstance(actor, StationaryActor),
                        'bbox': bbox
                    })

                    cv2.rectangle(debug_data['rect_mask'], bbox, thickness=1, color=color)

        debug_data['time'] = time.time() - debug_data['time']

        return bounding_boxes, debug_data