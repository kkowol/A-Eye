import time

import config as cfg  # Replace with own config!

import time
import carla
import numpy as np
#import open3d as o3d  # Only needed for demonstration


class SensorData:
    """
    Helper class to store incoming data from sensor first and process them later
    Only needed for demonstration in asynchronous mode
    """

    def __init__(self):
        self.data = []

    def collect(self, radar_data):
        self.data.append(radar_data)

    def clear(self):
        self.data = []

    def __iter__(self):
        for radar_data in self.data:
            yield radar_data

    def __len__(self):
        return len(self.data)


def polar_to_cartesian(radar_measurement, radar_location=carla.Location(0, 0, 0)):
    """
    The radar data is given in polar coordinates and needs to be converted into cartesian coordinates
    in order to work with the lidar data

    :param radar_measurement: Contains all points that have been recorded
    :param radar_location: If the sensor is placed with an offset on the vehicle, we can add that offset too
    :return: Numpy array of detected points in cartesian plus velocity of the detected object (x,y,z,v)

    Note:
        'All the sensors use the UE coordinate system (x-forward, y-right, z-up), and return coordinates in local space.
        When using any visualization software, pay attention to its coordinate system.
        Many invert the Y-axis, so visualizing the sensor data directly may result in mirrored outputs.'

    ! The coordinate system for the output has been changed and does not use the UE coordinate system
    """

    points = np.frombuffer(radar_measurement.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (len(radar_measurement), 4))

    v, phi, rho, r = points.T
    x = r * np.cos(phi) * np.cos(rho) + radar_location.x
    y = r * np.cos(phi) * np.sin(rho) + radar_location.z
    z = r * np.sin(phi) + radar_location.y

    return np.stack((x, y, z, v)).T


def save_to_disk(points, path):
    data = ('ply\n'
            'format ascii 1.0\n'
            f'element vertex {len(points)}\n'
            'property float32 x\n'
            'property float32 y\n'
            'property float32 z\n'
            'property float32 V\n'
            'end_header\n')

    for point_4d in points:
        x, y, z, v = point_4d.T
        data += f'{x:.4f} {y:.4f} {z:.4f} {v:.4f}\n'

    with open(path, 'w') as wf:
        wf.write(data)


def main():
    actors = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Getting the vehicle of the manual control script
        heroes = [actor for actor in world.get_actors() if actor.attributes.get('role_name') == 'hero']
        vehicle = heroes[0]

        # Offset of the sensor on the vehicle
        radar_transform = carla.Transform(carla.Location(x=1.5, z=1.5), carla.Rotation(pitch=0))

        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('points_per_second', '5000')

        radar = world.spawn_actor(
            radar_bp,
            radar_transform,
            attach_to=vehicle)
        actors.append(radar)

        radar_data = SensorData()
        radar.listen(radar_data.collect)

        num_scans = 1

        # Collecting data and wait until num_scans has been reached
        while len(radar_data) < num_scans:
            continue

        # Stop listening
        radar.stop()

        # Process each Frame
        for radar_measurement in radar_data:
            points = polar_to_cartesian(radar_measurement, radar_transform.location)
            save_to_disk(points, f"radar/{radar_measurement.frame}.ply")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            o3d.visualization.draw_geometries([pcd],
                                              zoom=0.3412,
                                              front=[0.4257, -0.2125, -0.8795],
                                              lookat=[2.6172, 2.0475, 1.532],
                                              up=[-0.0694, -0.9768, 0.2024])

    finally:
        for actor in actors:
            actor.destroy()


if __name__ == '__main__':
    main()
