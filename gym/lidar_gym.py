import gym
import camera
import tools.map_parser as parsing
import tools.math_processing as processing
import build.voxel_map as vm

fov = (90, 45)
density = (100, 100)
max_rays = 100


class LidarGym(gym.Env):

    def __init__(self):
        self._camera = camera.Camera(fov, density, max_rays)
        self._initial_position = (0, 0, 0)
        self._map, self._T_matrixes = parsing.parse_map()
        self._map_length = len(self._T_matrixes)
        self._next_timestamp = 0
        self._curr_position = None
        self._curr_T = None
        self._done = False

    def _reset(self):
        self._next_timestamp = 0
        self._done = False
        self._to_next()

    def _close(self):
        super(LidarGym, self)._close()

    def _step(self, action):
        if not self._done:
            # Check if vectors have correct dims!
            observation = vm.trace_ray(self._curr_position, self._camera.calculate_directions(action.input, self._curr_T), -100, 0)
            reward = processing.compute_reward(self._map, action.map)
            self._to_next()
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
