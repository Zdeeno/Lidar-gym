import gym
import gym.spaces as spaces
import numpy as np
from lidar_gym.tools import math_processing as processing
import voxel_map as vm
import lidar_gym.testing.test_map as tm
from lidar_gym.tools import camera
import sys
from lidar_gym.tools import map_parser

const_min_value = -sys.maxsize
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
        self._lidar_range = lidar_range
        self._voxel_size = voxel_size
        self._max_rays = max_rays
        self._density = np.asarray(density, dtype=np.float)
        self._fov = np.asarray(fov)
        self._T_forecast = T_forecast
        self._weight = weight
        self._map_shape = np.asarray(map_shape)  # sth like (80, 80, 4)

        self._input_map_shape = (self._map_shape / voxel_size) + np.ones((1, 3))
        self._camera = camera.Camera(self._fov, self._density, self._max_rays)

        max_val = sys.float_info.max
        min_val = -sys.float_info.min
        max_obs_pts = int((lidar_range / voxel_size) * max_rays)
        self.action_space = spaces.Dict({"rays": spaces.MultiBinary((self._density[1], self._density[0])),
                                         "map": spaces.Box(low=min_val,
                                                           high=max_val,
                                                           shape=self._input_map_shape[0].astype(int))})

        self.observation_space = spaces.Dict({"Ts": spaces.Box(low=min_val, high=max_val, shape=(T_forecast, 4, 4)),
                                              "points": spaces.Box(low=min_val, high=max_val, shape=(max_obs_pts, 3)),
                                              "values": spaces.Box(low=min_val, high=max_val, shape=(max_obs_pts, 1))})

        self._initial_position = np.zeros((1, 3))
        # use test_map.py or map_parser.py
        self._map, self._T_matrices = map_parser.parse_map(self._voxel_size)
        # self._map, self._T_matrices = tm.create_test_map()
        self._map_length = len(self._T_matrices)
        self._next_timestamp = 0
        self._curr_position = None
        self._curr_T = None
        self._done = False

        self._reward_counter = processing.RewardCounter(self._map, self._voxel_size, self._map_shape, self._weight)

    def _reset(self):
        self._next_timestamp = 0
        self._done = False
        self._to_next()
        obv = {"T": self._create_matrix_array(), "points": None, "values": None}
        return obv

    def _close(self):
        super(LidarGym, self)._close()

    def _step(self, action):
        assert type(action["rays"]) == np.ndarray and type(action["map"]) == np.ndarray, "wrong input types"

        if not self._done:
            directions = self._camera.calculate_directions(action["rays"], self._curr_T)
            if directions is not None:
                init_point = np.asmatrix(self._curr_position)
                x, v = self._create_observation(init_point, directions)
                reward = self._reward_counter.compute_reward(action["map"], self._curr_T)
                self._to_next()
                observation = {"T": self._create_matrix_array(), "points": np.transpose(x), "values": v}
            else:
                reward = self._reward_counter.compute_reward(action["map"], self._curr_T)
                self._to_next()
                observation = {"T": self._create_matrix_array(), "points": None, "values": None}

            return observation, reward, self._done, None
        else:
            return None, None, True, None

    def _render(self, mode='human', close=False):
        from lidar_gym.testing import plot_map

        if not self._done:
            g_t, a_m, sensor = self._reward_counter.get_render_data()
            plot_map.plot_action(g_t, None, None, self._voxel_size, sensor)

    def _to_next(self):
        if not self._done:
            if self._next_timestamp == self._map_length:
                self._curr_T = None
                self._done = True
                self._next_timestamp += 1
                return
            self._curr_T = self._T_matrices[self._next_timestamp]
            self._curr_position = processing.transform_points(self._initial_position, self._curr_T)
            self._next_timestamp += 1

    def _create_observation(self, init_point, directions):
        init_points = np.repeat(init_point, len(directions), axis=0)
        coords, v = self._map.trace_rays(np.transpose(init_points),
                                         np.transpose(directions),
                                         self._lidar_range, const_min_value, const_max_value, 0)
        if len(coords) == 0:
            return None, None

        tmp_map = vm.VoxelMap()
        tmp_map.voxel_size = self._voxel_size
        tmp_map.free_update = - 1.0
        tmp_map.hit_update = 1.0
        init_points = np.repeat(init_point, len(v), axis=0)
        tmp_map.update_lines(np.transpose(init_points), coords)
        # correct empty pts

        bools = processing.values_to_bools(v)
        indexes_empty = np.where(~bools)
        if len(indexes_empty) > 0:
            free_pts = np.asmatrix(coords[:, indexes_empty])
            tmp_map.set_voxels(free_pts, np.zeros((free_pts.shape[1],)), -np.ones((free_pts.shape[1],)))

        x, l, v = tmp_map.get_voxels()
        return x, v

    def _create_matrix_array(self):
        t = self._next_timestamp - 1
        if self._map_length >= (t + self._T_forecast):
            return self._T_matrices[t:(t + self._T_forecast)]
        else:
            diff = (self._map_length - t)
            ret = np.zeros((diff, 4, 4))
            if diff > 0:
                ret[0:diff] = self._T_matrices[t:t+diff]
            else:
                ret = [None]
            return ret


class Lidarv0(LidarGym):

    # trying to register environment described in paper
    def __init__(self):
        super(Lidarv0, self).__init__(70, 0.2, 100, (10, 10), (120, 90), 5, 1, (64, 64, 6))
