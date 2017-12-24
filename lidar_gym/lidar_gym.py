import gym
import gym.spaces as spaces
import numpy as np

import testing.test_map as tm
from tools import math_processing as processing
from tools import camera

const_min_value = -100
const_max_value = 0


class LidarGym(gym.Env):

    def __init__(self):
        # Constants to define in constructors:
        self._lidar_range = 100
        self._voxel_size = 1
        self._max_rays = 10
        self._density = (10, 10)
        self._fov = (90, 45)
        self._input_map_size = (80, 80, 4)
        self._input_map_shift_ratio = (0.5, 0.25, 0.5)

        self._camera = camera.Camera(self._fov, self._density, self._max_rays)
        # Action space is box of size 80x80x4, where lidar is placed into point [40, 20, 2]
        self.action_space = spaces.Tuple((spaces.MultiBinary((self._density[1], self._density[0])),
                                          spaces.MultiBinary((self._input_map_size[0]/self._voxel_size,
                                                              self._input_map_size[1]/self._voxel_size,
                                                              self._input_map_size[2]/self._voxel_size))))

        self.observation_space = spaces.Tuple((spaces.Box(-float('inf'), float('inf'), (4, 4)),
                                               spaces.Box(-float('inf'), float('inf'), (self._max_rays, 3))))

        self._initial_position = np.zeros((1, 3))
        # use test_map.py or map_parser.py
        # self._map, self._T_matrixes = parsing.parse_map()
        self._map, self._T_matrixes = tm.create_test_map()
        self._map_length = len(self._T_matrixes)
        self._next_timestamp = 0
        self._curr_position = None
        self._curr_T = None
        self._done = False
        self._reward_counter = processing.RewardCounter(self._map, self._voxel_size, self._input_map_size,
                                                        self._input_map_shift_ratio)

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
            # create data to trace rays
            directions = self._camera.calculate_directions(action[0], self._curr_T)
            init_points = np.asmatrix(self._curr_position)
            init_points = np.repeat(init_points, len(directions), axis=0)
            print('tracing rays from:\n', self._curr_position, '\nwith directions:\n', directions)
            coords, v = self._map.trace_rays(np.transpose(init_points),
                                             np.transpose(directions),
                                             self._lidar_range, const_min_value, const_max_value, 0)
            bools = processing.values_to_bools(v)
            indexes = np.where(bools)
            print('traced indexes:\n', indexes)
            reward = self._reward_counter.compute_reward(action[1], self._curr_T)
            self._to_next()
            observation = (self._curr_T, np.transpose(coords)[indexes])
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
