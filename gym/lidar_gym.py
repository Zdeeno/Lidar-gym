import gym
import gym.spaces as spaces
import camera
import tools.map_parser as parsing
import tools.math_processing as processing
import build.voxel_map as vm
import numpy as np

const_min_value = -100
const_max_value = 0


class LidarGym(gym.Env):

    def __init__(self):
        # Constants to define in constructors:
        self._lidar_range = 60
        self._voxel_size = 0.5
        self._max_rays = 100
        self._density = (100, 100)
        self._fov = (90, 45)
        self._input_map_size = (80, 80, 4)
        self._input_map_shift_ratio = (0.5, 0.25, 0.5)

        self._camera = camera.Camera(self._fov, self._density, self._max_rays)
        # Action space is box of size 80x80x4, where lidar is placed into point [40, 20, 2]
        self.action_space = spaces.Tuple((spaces.MultiBinary((self._density[1], self._density[0])),
                                          spaces.MultiBinary((self._input_map_size[0]/self._voxel_size,
                                                              self._input_map_size[1]/self._voxel_size,
                                                              self._input_map_size[2]/self._voxel_size))))

        self._initial_position = (0, 0, 0)
        self._map, self._T_matrixes = parsing.parse_map()
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

    def _close(self):
        super(LidarGym, self)._close()

    def _step(self, action):
        if not self._done:
            # Check if vectors have correct dims! TODO: finish observation_space
            coords, _ = vm.trace_ray(self._curr_position,
                                     np.transpose(self._camera.calculate_directions(action[0], self._curr_T)),
                                     self._lidar_range, const_min_value, const_max_value, 0)
            reward = self._reward_counter.compute_reward(action[1])
            self._to_next()
            observation = (self._curr_T, np.transpose(coords))
            return observation, reward, self._done, None
        else:
            return None, None, True, None

    def _render(self, mode='human', close=False):
        super(LidarGym, self)._render(mode, close)

    def _to_next(self):
        self._curr_T = self._T_matrixes[self._next_timestamp]
        self._curr_position = processing.transform_points(self._initial_position, self._curr_T)
        if self._next_timestamp < (self._map_length - 1):
            self._next_timestamp += 1
        else:
            self._done = True
