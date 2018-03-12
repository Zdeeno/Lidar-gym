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
        :param map_shape: tuple, size of input map (x, y, z)
        :param T_cuboid: numpy matrix 4x4, shift of local map in meters
    """

    metadata = {
        "render.modes": ["human"],
    }

    def __init__(self, lidar_range, voxel_size, max_rays, density, fov, T_forecast, map_voxel_shape, T_cuboid):
        # Parse arguments:
        self.__lidar_range = lidar_range
        self.__voxel_size = voxel_size
        self.__max_rays = max_rays
        self.__density = np.asarray(density, dtype=np.float)
        self.__fov = np.asarray(fov)
        self.__map_shape = np.asarray(map_voxel_shape) * voxel_size  # sth like (80, 80, 4)
        self.__initial_position = np.zeros((1, 3))
        self.__input_map_shape = np.asarray(map_voxel_shape)

        if T_forecast == 0:
            self.__T_forecast = sys.maxsize
        else:
            self.__T_forecast = T_forecast

        # init classes
        self.__camera = camera.Camera(self.__fov, self.__density, self.__max_rays)
        self.__maps = map_parser.MapParser(self.__voxel_size)
        self.__cuboid_getter = processing.CuboidGetter(voxel_size, self.__map_shape)
        self.__reward_counter = processing.RewardCounter(self.__voxel_size, self.__map_shape, T_cuboid,
                                                         self.__cuboid_getter)

    def _reset(self):
        # reset values
        self.__next_timestamp = 0
        self.__curr_position = None
        self.__curr_T = None
        self.__done = False
        self.__render_init = False
        self.__rays_endings = None
        self.__map_length = None
        self.__map = None
        self.__T_matrices = None

        print('RESETING')
        # parse new map
        self.__map, self.__T_matrices = self.__maps.get_next_map()
        self.__reward_counter.reset(self.__map)
        self.__map_length = len(self.__T_matrices)
        self.__to_next()
        obv = {"T": self.__create_matrix_array(), "points": None, "values": None}
        return obv

    def _close(self):
        super(LidarGym, self).close()

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
            if self.__next_timestamp > 1:
                g_t, a_m, sensor = self.__reward_counter.get_render_data()
                self.plotter.plot_action(g_t, a_m, np.transpose(self.__rays_endings), self.__voxel_size, sensor)

    def _seed(self, seed=None):
        map_parser.set_seed(seed)

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


class LidarMultiBinary(spaces.MultiBinary):
    """
    Random ray generator.
    """
    def __init__(self, n, maxrays):
        self.maxrays = maxrays
        self.n = n
        super(LidarMultiBinary, self).__init__(n)

    def sample(self):
        x = np.random.random_integers(0, high=self.n[0] - 1, size=self.maxrays)
        y = np.random.random_integers(0, high=self.n[1] - 1, size=self.maxrays)
        ret = np.zeros(shape=self.n, dtype=bool)
        ret[x, y] = True
        return ret


class LidarBox(spaces.Box):
    """
    Random map reconstruction generator.
    """
    def __init__(self, low, high, shape=None):
        super(LidarBox, self).__init__(low, high, shape)

    def sample(self):
        return np.random.uniform(low=-10, high=0.01, size=self.low.shape)


class Lidarv0(LidarGym):
    """
    Simplified environment for basic training.
    """
    def __init__(self):

        map_voxel_shape = (320, 320, 32)
        forecast = 0
        fov = (120, 90)
        density = (160, 120)
        max_rays = 200
        voxel_size = 0.2
        lidar_range = 48

        self.__shift_T = np.eye(4, dtype=float)
        self.__shift_T[0, 3] = -0.25 * map_voxel_shape[0] * voxel_size
        self.__shift_T[1, 3] = -0.5 * map_voxel_shape[1] * voxel_size
        self.__shift_T[2, 3] = -0.5 * map_voxel_shape[2] * voxel_size

        super(Lidarv0, self).__init__(lidar_range, voxel_size, max_rays, density, fov,
                                      forecast, map_voxel_shape, self.__shift_T)

        self.action_space = spaces.Dict(spaces.Box(low=-1000, high=1000, shape=(320, 320, 32)))
        self.observation_space = spaces.Dict(spaces.Box(low=-1000, high=1000, shape=(320, 320, 32)))
        self.reward_range = (-float('Inf'), 0)

        self.__action_generator = LidarMultiBinary(density, max_rays)

        # lidarv0 specific:
        self.__obs_voxel_map = None

    def _close(self):
        super(Lidarv0).close()

    def _reset(self):
        self.__obs_voxel_map = vm.VoxelMap()
        self.__obs_voxel_map.voxel_size = self.__voxel_size
        self.__obs_voxel_map.free_update = - 1.0
        self.__obs_voxel_map.hit_update = 1.0
        self.__obs_voxel_map.occupancy_threshold = 0.0
        obs = super(Lidarv0, self).reset()
        self.curr_T = obs['T'][0]
        return np.zeros(shape=(320, 320, 32))

    def _step(self, action):
        rand_action = self.__action_generator.sample()
        rays = rand_action['rays']
        obs, rew, done, info = super(Lidarv0, self).step({'rays': rays, 'map': action})
        obs = self._preprocess_obs(obs)
        return obs, rew, done, info

    def _seed(self, seed=None):
        super(Lidarv0, self).seed(seed)

    def _render(self, mode='human', close=False):
        super(Lidarv0, self).render(mode)

    def _preprocess_obs(self, obs):
        new_points = np.transpose(obs['points'])
        new_vals = obs['values']
        last_vals = self.__obs_voxel_map.get_voxels(new_points)
        new_vals = last_vals + new_vals
        self.__obs_voxel_map.set_voxels(new_points, new_vals)

        # get CNN input
        points, values = self.__cuboid_getter.get_map_cuboid(self.__obs_voxel_map, self.curr_T, self.__shift_T)
        points = points/self.__voxel_size
        points = np.asarray(points, dtype=int)
        ret = np.zeros(shape=self.__map_shape)
        ret[points[0], points[1], points[2]] = values
        self.curr_T = obs['T'][0]
        return ret


class Lidarv1(LidarGym):

    # environment described in paper
    def __init__(self):

        map_voxel_shape = (320, 320, 32)
        forecast = 0
        fov = (120, 90)
        density = (160, 120)
        max_rays = 200
        voxel_size = 0.2
        lidar_range = 48

        shift_T = np.eye(4, dtype=float)
        shift_T[0, 3] = -0.25 * map_voxel_shape[0] * voxel_size
        shift_T[1, 3] = -0.5 * map_voxel_shape[1] * voxel_size
        shift_T[2, 3] = -0.5 * map_voxel_shape[2] * voxel_size

        # Define state spaces and reward bounds
        max_val = sys.float_info.max
        min_val = -sys.float_info.min
        max_obs_pts = int((lidar_range / voxel_size) * max_rays)
        self.action_space = spaces.Dict({"rays": LidarMultiBinary((int(self.__density[1]), int(self.__density[0])), max_rays),
                                         "map": LidarBox(low=min_val,
                                                         high=max_val,
                                                         shape=self.__input_map_shape.astype(int))})

        self.observation_space = spaces.Dict({"T": spaces.Box(low=min_val, high=max_val, shape=(100, 4, 4)),
                                              "points": spaces.Box(low=min_val, high=max_val, shape=(max_obs_pts, 3)),
                                              "values": spaces.Box(low=min_val, high=max_val, shape=(max_obs_pts, 1))})
        self.reward_range = (-float('Inf'), 0)

        super(Lidarv1, self).__init__(lidar_range, voxel_size, max_rays, density, fov,
                                      forecast, map_voxel_shape, shift_T)

    def reset(self):
        return super(Lidarv1, self).reset()

    def close(self):
        super(Lidarv1, self).close()

    def seed(self, seed=None):
        return super(Lidarv1, self).seed(seed)

    def step(self, action):
        return super(Lidarv1, self).step(action)

    def render(self, mode='human'):
        super(Lidarv1, self).render(mode)
