import gym
import gym.spaces as spaces
import numpy as np
from lidar_gym.tools import math_processing as processing
import voxel_map as vm
from lidar_gym.tools import camera
import sys
from lidar_gym.tools import map_parser

const_min_value = -sys.maxsize - 1
const_max_value = 0


class LidarGym(gym.Env):
    """
    Class representing solid-state lidar sensor training environment.
    Parameters:
        :param lidar_range: float
        :param voxel_size: float
        :param fov: tuple, angles in degrees for (width, height)
        :param density: tuple, number of points over fov (width, height)
        :param max_rays: integer, maximum number of rays per timestamp
        :param T_forecast: integer, determines how many steps forward is environment returning position
        :param weight: float, it is used to calculated reward (see class RewardCounter)
        :param map_shape: tuple, size of input map (x, y, z)
    """

    metadata = {
        "render.modes": ["human"],
    }

    def __init__(self, lidar_range, voxel_size, max_rays, density, fov, T_forecast, weight, map_shape):
        # Parse arguments:
        self.__lidar_range = lidar_range
        self.__voxel_size = voxel_size
        self.__max_rays = max_rays
        self.__density = np.asarray(density, dtype=np.float)
        self.__fov = np.asarray(fov)
        self.__T_forecast = T_forecast
        self.__weight = weight
        self.__map_shape = np.asarray(map_shape)  # sth like (80, 80, 4)

        self.__input_map_shape = (self.__map_shape / voxel_size) + np.ones((1, 3))

        # Observation and action space
        max_val = sys.float_info.max
        min_val = -sys.float_info.min
        max_obs_pts = int((lidar_range / voxel_size) * max_rays)
        self.action_space = spaces.Dict({"rays": spaces.MultiBinary((self.__density[1], self.__density[0])),
                                         "map": spaces.Box(low=min_val,
                                                           high=max_val,
                                                           shape=self.__input_map_shape[0].astype(int))})

        self.observation_space = spaces.Dict({"Ts": spaces.Box(low=min_val, high=max_val, shape=(T_forecast, 4, 4)),
                                              "points": spaces.Box(low=min_val, high=max_val, shape=(max_obs_pts, 3)),
                                              "values": spaces.Box(low=min_val, high=max_val, shape=(max_obs_pts, 1))})

        # additional init ... mostly set to None before reset.
        self.__initial_position = np.zeros((1, 3))
        self.__next_timestamp = 0
        self.__curr_position = None
        self.__curr_T = None
        self.__done = False
        self.__render_init = False
        self.__rays_endings = None
        self.__map_length = None
        self.__map = None
        self.__T_matrices = None
        self.__reward_counter = None

        # init classes
        self.__camera = camera.Camera(self.__fov, self.__density, self.__max_rays)
        self.__maps = map_parser.MapParser(self.__voxel_size)

    def _reset(self):
        self.__next_timestamp = 0
        self.__done = False
        self.__map, self.__T_matrices = self.__maps.get_next_map()
        self.__reward_counter = processing.RewardCounter(self.__map, self.__voxel_size, self.__map_shape, self.__weight)
        self.__map_length = len(self.__T_matrices)
        self.__to_next()
        obv = {"T": self.__create_matrix_array(), "points": None, "values": None}
        return obv

    def _close(self):
        super(LidarGym, self)._close()

    def _step(self, action):
        assert type(action["rays"]) == np.ndarray and type(action["map"]) == np.ndarray, "wrong input types"

        if not self.__done:
            directions = self.__camera.calculate_directions(action["rays"], self.__curr_T)
            if directions is not None:
                init_point = np.asmatrix(self.__curr_position)
                x, v = self.__create_observation(init_point, directions)
                reward = self.__reward_counter.compute_reward(action["map"], self.__curr_T)
                self.__to_next()
                observation = {"T": self.__create_matrix_array(), "points": np.transpose(x), "values": v}
            else:
                reward = self.__reward_counter.compute_reward(action["map"], self.__curr_T)
                self.__to_next()
                observation = {"T": self.__create_matrix_array(), "points": None, "values": None}

            return observation, reward, self.__done, None
        else:
            return None, None, True, None

    def _render(self, mode='human', close=False):
        if not self.__done:
            if not self.__render_init:
                from lidar_gym.testing import plot_map
                self.plotter = plot_map.Potter()
                self.__render_init = True

            g_t, a_m, sensor = self.__reward_counter.get_render_data()
            self.plotter.plot_action(g_t, a_m, np.transpose(self.__rays_endings), self.__voxel_size, sensor)

    def __to_next(self):
        if not self.__done:
            if self.__next_timestamp == self.__map_length:
                self.__curr_T = None
                self.__done = True
                self.__next_timestamp += 1
                return
            self.__curr_T = self.__T_matrices[self.__next_timestamp]
            self.__curr_position = processing.transform_points(self.__initial_position, self.__curr_T)
            self.__next_timestamp += 1

    def __create_observation(self, init_point, directions):
        if len(directions) == 0:
            return None, None

        init_points = np.repeat(init_point, len(directions), axis=0)
        self.__rays_endings, v = self.__map.trace_rays(np.transpose(init_points),
                                                       np.transpose(directions),
                                                       self.__lidar_range, const_min_value, const_max_value, 0)

        tmp_map = vm.VoxelMap()
        tmp_map.voxel_size = self.__voxel_size
        tmp_map.free_update = - 1.0
        tmp_map.hit_update = 1.0
        init_points = np.repeat(init_point, len(v), axis=0)
        tmp_map.update_lines(np.transpose(init_points), self.__rays_endings)
        # correct empty pts

        bools = processing.values_to_bools(v)
        indexes_empty = np.where(~bools)
        if len(indexes_empty[0]) > 0:
            free_pts = np.asmatrix(self.__rays_endings[:, indexes_empty])
            tmp_map.set_voxels(free_pts, np.zeros((free_pts.shape[1],)), -np.ones((free_pts.shape[1],)))

        x, l, v = tmp_map.get_voxels()
        return x, v

    def __create_matrix_array(self):
        """
        create array of transformation matrix with current position
        :return: numpy array Nx(4x4)
        """
        t = self.__next_timestamp - 1
        if self.__map_length >= (t + self.__T_forecast):
            return self.__T_matrices[t:(t + self.__T_forecast)]
        else:
            diff = (self.__map_length - t)
            ret = np.zeros((diff, 4, 4))
            if diff > 0:
                ret[0:diff] = self.__T_matrices[t:t + diff]
            else:
                ret = [None]
            return ret


class Lidarv0(LidarGym):

    # trying to register environment described in paper
    def __init__(self):
        super(Lidarv0, self).__init__(70, 0.2, 100, (10, 10), (120, 90), 5, 1, (64, 64, 6))
