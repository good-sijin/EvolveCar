"""
 Copyright (c) 2025 Jim Li
 
 Most of the code is forked from https://github.com/cjy1992/gym-carla.

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 """


import copy
import logging
import os
import random
import signal
import subprocess
import time
from typing import Optional

import carla
import gymnasium as gym
import numpy as np
import psutil
from evolve_car.simulator.core.misc import *
from evolve_car.simulator.core.render import BirdeyeRender
from evolve_car.simulator.core.route_planner import RoutePlanner
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils import seeding
from skimage.transform import resize


class CarlaEnv(gym.Env):
    """
    Carla simulator wrapper which is compatible with the latest ray api.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}

        # screen size of bird-eye render
        self.display_size = config.get('display_size', 256)
        # the number of past steps to draw
        self.max_past_step = config.get('max_past_step', 1)
        self.number_of_vehicles = config.get('number_of_vehicles', 0)
        self.number_of_walkers = config.get('number_of_walkers', 0)
        # time interval between two frames
        self.dt = config.get('dt', 0.1)
        self.task_mode = config.get('task_mode', 'random')
        # maximum timesteps per episode
        self.max_time_episode = config.get('max_time_episode', 1000)
        # maximum number of waypoints
        self.max_waypt = config.get('max_waypt', 12)
        self.obs_range = config.get('obs_range', 32)
        self.lidar_bin = config.get('lidar_bin', 0.125)
        # distance behind the ego vehicle (meter)
        self.d_behind = config.get('d_behind', 12)
        # threshold for out of lane
        self.out_lane_thres = config.get('out_lane_thres', 2.0)
        # desired speed (m/s)
        self.desired_speed = config.get('desired_speed', 8)
        self.max_ego_spawn_times = config.get('max_ego_spawn_times', 200)
        self.display_route = config.get('display_route', False)
        self.pixor = config.get('pixor', True)
        self.pixor_size = config.get('pixor_size', 64)
        self.with_lidar = config.get('with_lidar', False)

        # -----------------Carla
        self.carla_quality_level = config.get('quality_level', 'low')

        self.dests = None
        # Action space for the car are acc and steering angle.
        acc_range = config.get("acc_range", [-3.0, 3.0])
        steer_range = config.get("steer_range", [-0.3, 0.3])
        self.action_space = Box(np.array([acc_range[0], steer_range[0]]),
                                np.array([acc_range[1], steer_range[1]]), dtype=np.float32)

        # Observation range and resolution, which can deduce the grid map size.
        self.obs_range = config.get("obs_range", 32)
        self.obs_resolution = config.get("obs_resolution", 0.125)
        self.obs_size = int(self.obs_range / self.obs_resolution)

        # Obsevation space for the car are camera, lidar, bireye, car-state, etc.
        observation_space_dict = {
            # 'camera': Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'birdeye': Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            # 'state': Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
        }
        if self.with_lidar:
            observation_space_dict.update({
                'lidar': Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            }
            )

        if self.pixor:
            pass
            # observation_space_dict.update({
            #     'roadmap': Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            #     'vh_clas': Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
            #     'vh_regr': Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
            #     'pixor_state': Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
            # })
        self.observation_space = Dict(observation_space_dict)

        # Connect to carla server and get world object
        logging.info("connecting to Carla server...")
        self.init_server()
        self.connect_client()
        self.client.set_timeout(10.0)

        self.world = self.client.load_world(
            config.get("town", "/Game/Carla/Maps/Town10HD_Opt"))
        logging.info("Carla server connected!")

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.vehicle_spawn_points = list(
            self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(config.get(
            'ego_vehicle_filter', 'vehicle.lincoln*'), color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        if self.with_lidar:
            # Lidar sensor
            self.lidar_data = None
            self.lidar_height = 2.1
            self.lidar_trans = carla.Transform(
                carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '32')
            self.lidar_bp.set_attribute('range', '5000')

        # Camera sensor
        self.camera_img = np.zeros(
            (self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Initialize the renderer
        self._init_renderer()
        # Get pixel grid points
        if self.pixor:
            x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(
                self.pixor_size))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T

    def is_used(self, port):
        """Checks whether or not a port is used"""
        return port in [conn.laddr.port for conn in psutil.net_connections()]

    def init_server(self):
        """Start a server on a random port"""
        self.server_port = random.randint(15000, 32000)

        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))

        uses_server_port = self.is_used(self.server_port)
        uses_stream_port = self.is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + self.server_port)
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port+1))
            self.server_port += 2
            uses_server_port = self.is_used(self.server_port)
            uses_stream_port = self.is_used(self.server_port+1)

        server_command = [
            "{}/CarlaUE4.sh".format(os.environ.get("CARLA_ROOT", "/home/carla")),
        ]

        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.carla_quality_level),
            "-RenderOffScreen",
        ]

        server_command_text = " ".join(map(str, server_command))
        self.server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )
        print(server_command_text)

    def connect_client(self, host="localhost", retries_on_error=5, timeout=10):
        """Connect to the client"""

        for i in range(retries_on_error):
            try:
                self.client = carla.Client(host, self.server_port)
                self.client.set_timeout(timeout)
                self.world = self.client.get_world()
                return

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(
                    e, i + 1, retries_on_error))
                time.sleep(3)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def reset(self, *, seed=None, options=None):
        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision',
                                'sensor.lidar.ray_cast',
                               'sensor.camera.rgb',
                                'vehicle.*',
                                'controller.ai.walker',
                                'walker.*'])

        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()

            if self.task_mode == 'random':
                transform = random.choice(self.vehicle_spawn_points)
            if self.task_mode == 'roundabout':
                self.start = [
                    52.1+np.random.uniform(-5, 5), -4.2, 178.66]  # random
                # self.start=[52.1,-4.2, 178.66] # static
                transform = set_carla_transform(self.start)
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
            self.collision_hist = []

        if self.with_lidar:
            def get_lidar_data(data):
                self.lidar_data = data
            # Add lidar sensor
            self.lidar_sensor = self.world.spawn_actor(
                self.lidar_bp, self.lidar_trans, attach_to=self.ego)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(
            self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)

        return self._get_obs(), {"env_state": "reset"}

    def step(self, action):
        acc, steer = action

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc/3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc/8, 0, 1)

        # Apply control
        act = carla.VehicleControl(throttle=float(
            throttle), steer=float(-steer), brake=float(brake))

        self.ego.apply_control(act)

        self.world.tick()

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        truncated = False
        return (self._get_obs(), self._get_reward(), self._terminal(), truncated, copy.deepcopy(info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
        actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
        bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + \
                [x for x in blueprints if int(
                    x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(
                    bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
        """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (
            self.obs_range/2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint(
            'vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(
            self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(
                walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(
                self.world.get_random_location_from_navigation())
            # random max speed
            # max speed between 1 and 2 (default is 1.4 m/s)
            walker_controller_actor.set_max_speed(1 + random.random())
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
        transform: the carla transform object.
        Returns:
        Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
        filt: the filter indicating what type of actors we'll look at.

        Returns:
        actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw/180*np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array(
                [[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + \
                np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self):
        """Get the observations."""
        # Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.waypoints

        # birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
            birdeye_render_types.append('waypoints')
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0:self.display_size, :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)

        # Roadmap
        if self.pixor:
            roadmap_render_types = ['roadmap']
            if self.display_route:
                roadmap_render_types.append('waypoints')
            self.birdeye_render.render(self.display, roadmap_render_types)
            roadmap = pygame.surfarray.array3d(self.display)
            roadmap = roadmap[0:self.display_size, :, :]
            roadmap = display_to_rgb(roadmap, self.obs_size)
            # Add ego vehicle
            for i in range(self.obs_size):
                for j in range(self.obs_size):
                    if abs(birdeye[i, j, 0] - 255) < 20 and abs(birdeye[i, j, 1] - 0) < 20 and abs(birdeye[i, j, 0] - 255) < 20:
                        roadmap[i, j, :] = birdeye[i, j, :]
        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        self.display.blit(birdeye_surface, (0, 0))

        if self.with_lidar:
            # Lidar image generation
            point_cloud = []
            # Get point cloud data
            for lidar_detection in self.lidar_data:
                # The api interface has change to LidarDetection.
                location = lidar_detection.point
                point_cloud.append([location.x, location.y, -location.z])
            point_cloud = np.array(point_cloud)
            # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
            # and z is set to be two bins.
            y_bins = np.arange(-(self.obs_range - self.d_behind),
                               self.d_behind+self.lidar_bin, self.lidar_bin)
            x_bins = np.arange(-self.obs_range/2, self.obs_range /
                               2+self.lidar_bin, self.lidar_bin)
            z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]
            # Get lidar image according to the bins
            lidar, _ = np.histogramdd(
                point_cloud, bins=(x_bins, y_bins, z_bins))
            lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
            lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
            # Add the waypoints to lidar image
            if self.display_route:
                wayptimg = (birdeye[:, :, 0] <= 10) * \
                    (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)

            else:
                wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
            wayptimg = np.expand_dims(wayptimg, axis=2)
            wayptimg = np.fliplr(np.rot90(wayptimg, 3))

            # Get the final lidar image
            lidar = np.concatenate((lidar, wayptimg), axis=2)
            lidar = np.flip(lidar, axis=1)
            lidar = np.rot90(lidar, 1)
            lidar = lidar * 255

            # Display lidar image
            lidar_surface = rgb_to_display_surface(lidar, self.display_size)
            self.display.blit(lidar_surface, (self.display_size, 0))

        # Display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size * 2, 0))

        # Display on pygame
        pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw/180*np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w,
                                       np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

        if self.pixor:
            # Vehicle classification and regression maps (requires further normalization)
            vh_clas = np.zeros((self.pixor_size, self.pixor_size))
            vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

        # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
        # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
        for actor in self.world.get_actors().filter('vehicle.*'):
            x, y, yaw, l, w = get_info(actor)
            x_local, y_local, yaw_local = get_local_pose(
                (x, y, yaw), (ego_x, ego_y, ego_yaw))
            if actor.id != self.ego.id:
                if abs(y_local) < self.obs_range/2+1 and x_local < self.obs_range-self.d_behind+1 and x_local > -self.d_behind-1:
                    x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
                        local_info=(x_local, y_local, yaw_local, l, w),
                        d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
                    cos_t = np.cos(yaw_pixel)
                    sin_t = np.sin(yaw_pixel)
                    logw = np.log(w_pixel)
                    logl = np.log(l_pixel)
                    pixels = get_pixels_inside_vehicle(
                        pixel_info=(x_pixel, y_pixel, yaw_pixel,
                                    l_pixel, w_pixel),
                        pixel_grid=self.pixel_grid)
                    for pixel in pixels:
                        vh_clas[pixel[0], pixel[1]] = 1
                        dx = x_pixel - pixel[0]
                        dy = y_pixel - pixel[1]
                        vh_regr[pixel[0], pixel[1], :] = np.array(
                            [cos_t, sin_t, dx, dy, logw, logl])

        # Flip the image matrix so that the origin is at the left-bottom
        vh_clas = np.flip(vh_clas, axis=0)
        vh_regr = np.flip(vh_regr, axis=0)

        # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
        pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

        obs = {
            # 'camera': camera.astype(np.uint8),
            'birdeye': birdeye.astype(np.uint8),
            # 'state': state,
        }
        if self.with_lidar:
            obs['lidar_state'] = lidar.astype(np.uint8)

        if self.pixor and False:
            obs.update({
                'roadmap': roadmap.astype(np.uint8),
                'vh_clas': np.expand_dims(vh_clas, -1).astype(np.float32),
                'vh_regr': vh_regr.astype(np.float32),
                'pixor_state': pixor_state,
            })

        return obs

    def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        r_speed = -abs(speed - self.desired_speed)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer**2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

        r = 20*r_collision + 1*lspeed_lon + 10 * \
            r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 4:
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            return True

        return False

    def _clear_all_actors(self, actor_filters=['sensor.other.collision',
                                'sensor.lidar.ray_cast',
                               'sensor.camera.rgb',
                                'vehicle.*',
                                'controller.ai.walker',
                                'walker.*']):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()

    def close(self):
        print("Terminating carla...")
        self.server_process.kill()
        self.server_process.wait()
        os.killpg(self.server_process.pid, signal.SIGTERM)
        # Only way to kill the carla.
        os.system("pkill -f CarlaUE4-Linux-Shipping")


if __name__ == "__main__":
    # Start carla outside of the script:
    # /home/carla/CarlaUnreal.sh  -carla-port=22912 -quality-level=Low -RenderOffScreen
    from gymnasium.envs.registration import register
    register(
        id='carla-v0',
        entry_point='evolve_car.simulator.core.carla_env:CarlaEnv',
    )
    # Set gym-carla environment
    env = gym.make('carla-v0', config={"port": 4010})

    def signal_handler(signal, frame):
        import sys
        print('\nSignal Catched! You have just type Ctrl+C!')
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    obs, _ = env.reset()
    while True:
        action = [2.0, 0.1]
        obs, r, done, _, info = env.step(action)
        if done:
            obs, _ = env.reset()
            print(obs.keys())
