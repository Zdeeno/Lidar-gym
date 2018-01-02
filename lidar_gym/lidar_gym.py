import gym
import gym.spaces as spaces
import numpy as np
from lidar_gym.tools import math_processing as processing
import voxel_map as vm
import lidar_gym.testing.test_map as tm
from lidar_gym.tools import camera
import sys

const_min_value = -100
const_max_value = 0


class LidarGym(gym.Env):
    '''
    Class representing solid state lidar sensor training environment.
    Parameters:
        :param lidar_range: double
        :param voxel_size: double
        :param fov: tuple, angles in degrees for (width, height)
        :param density: tuple, number of points over fov (width, height)
        :param max_rays: integer, maximum number of rays per timestamp
    '''

    def __init__(self, lidar_range, voxel_size, max_rays, density, fov):
        # TODO Constants to define in constructors:
        self._lidar_range = lidar_range
        self._voxel_size = voxel_size
        self._max_rays = max_rays
        self._density = np.asarray(density, dtype=np.float)
        self._fov = np.asarray(fov)
        # TODO check sizes!!!
        self._map_shape = np.asarray((80, 80, 4))  # without zeros ... add 1
        self._map_shift_length = np.asarray((40, 20, 2))
        self._input_map_shape = (self._map_shape / voxel_size) + np.ones((1, 3))
        self._camera = camera.Camera(self._fov, self._density, self._max_rays)

        max = sys.maxsize
        min = -sys.maxsize - 1
        max_obs_pts = int((lidar_range / voxel_size) * max_rays)
        self.action_space = spaces.Tuple((spaces.MultiBinary((self._density[1], self._density[0])),
                                          spaces.Box(low=min,
                                                     high=max,
                                                     shape=self._input_map_shape[0].astype(int))))

        self.observation_space = spaces.Tuple((spaces.Box(low=min, high=max, shape=(4, 4)),
                                               spaces.Box(low=min, high=max, shape=(max_obs_pts, 3)),
                                               spaces.Box(low=min, high=max, shape=(max_obs_pts, 1))))

        self._initial_position = np.zeros((1, 3))
        # use test_map.py or map_parser.py
        # self._map, self._T_matrixes = parsing.parse_map()
        self._map, self._T_matrixes = tm.create_test_map()
        self._map_length = len(self._T_matrixes)
        self._next_timestamp = 0
        self._curr_position = None
        self._curr_T = None
        self._done = False

        self._reward_counter = processing.RewardCounter(self._map, self._voxel_size, self._map_shape,
                                                        self._map_shift_length)

    def _reset(self):
        self._next_timestamp = 0
        self._done = False
        self._to_next()
        obv = (self._curr_T, None)
        return obv

    def _close(self):
        super(LidarGym, self)._close()

    def _step(self, action):
        if not self._done:
            directions = self._camera.calculate_directions(action[0], self._curr_T)
            init_point = np.asmatrix(self._curr_position)
            x, v = self._create_observation(init_point, directions)

            #print('traced indexes:\n', indexes)
            reward = self._reward_counter.compute_reward(action[1], self._curr_T)
            self._to_next()

            observation = (self._curr_T, np.transpose(x), v)
            return observation, reward, self._done, None
        else:
            return None, None, True, None

    def _render(self, mode='human', close=False):
        super(LidarGym, self)._render(mode, close)

    def _to_next(self):
        if not self._done:
            if self._next_timestamp == self._map_length:
                self._curr_T = None
                self._done = True
                return
            self._curr_T = self._T_matrixes[self._next_timestamp]
            self._curr_position = processing.transform_points(self._initial_position, self._curr_T)
            self._next_timestamp += 1

    def _create_observation(self, init_point, directions):
        # TODO: dont forget missed rays
        init_points = np.repeat(init_point, len(directions), axis=0)
        coords, v = self._map.trace_rays(np.transpose(init_points),
                                         np.transpose(directions),
                                         self._lidar_range, const_min_value, const_max_value, 0)
        bools = processing.values_to_bools(v)
        indexes = np.where(bools)
        hit_voxels = np.asmatrix(coords[:, indexes])

        tmp_map = vm.VoxelMap()
        tmp_map.voxel_size = 0.5
        tmp_map.free_update = - 1.0
        tmp_map.hit_update = 1.0
        init_points = np.repeat(init_point, np.shape(indexes)[1], axis=0)
        tmp_map.update_lines(np.transpose(init_points), hit_voxels)
        x, l, v = tmp_map.get_voxels()
        return x, v
