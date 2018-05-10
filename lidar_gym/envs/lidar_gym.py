import sys
import warnings

import gym
import gym.spaces as spaces
import numpy as np
import voxel_map as vm

from lidar_gym.tools import camera
from lidar_gym.tools import map_parser
from lidar_gym.tools import math_processing as processing
from lidar_gym.visualiser.printer import ray_string

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
        :param map_voxel_shape: tuple, size of input map (x, y, z)
        :param T_cuboid: numpy matrix 4x4, shift of local map in meters
    """

    metadata = {
        "render.modes": ["human", "ASCII"],
    }

    def __init__(self, lidar_range, voxel_size, max_rays, density, fov, T_forecast, map_voxel_shape, T_cuboid):
        # Parse arguments:
        self._lidar_range = lidar_range
        self._voxel_size = voxel_size
        self._max_rays = max_rays
        self._density = np.asarray(density, dtype=np.float)
        self._fov = np.asarray(fov)
        self._map_shape = np.asarray(map_voxel_shape) * voxel_size  # sth like (80, 80, 4)
        self._initial_position = np.zeros((1, 3))
        self._input_map_shape = np.asarray(map_voxel_shape)

        if T_forecast == 0:
            self._T_forecast = sys.maxsize
        else:
            self._T_forecast = T_forecast

        # init classes
        self._camera = camera.Camera(self._fov, self._density, self._max_rays)
        self._maps = map_parser.MapParser(self._voxel_size)
        self._cuboid_getter = processing.CuboidGetter(voxel_size, self._map_shape)
        self._reward_counter = processing.RewardCounter(self._voxel_size, self._map_shape, T_cuboid,
                                                        self._cuboid_getter)

    def _reset(self):
        # reset values
        self._last_rays = 0
        self._next_timestamp = 0
        self._curr_position = None
        self._curr_T = None
        self._last_T = None
        self._done = False
        self._render_init = False
        self._rays_endings = None
        self._map_length = None
        self._map = None
        self._T_matrices = None

        print('RESETING')
        # parse new map
        self._map, self._T_matrices = self._maps.get_next_map()
        self._reward_counter.reset(self._map)
        self._map_length = len(self._T_matrices)
        self._curr_T = self._T_matrices[0]
        self._to_next()
        obv = {"T": self._create_matrix_array(), "points": None, "values": None}
        return obv

    def _close(self):
        super(LidarGym, self).close()

    def _step(self, action):
        assert type(action["rays"]) == np.ndarray and type(action["map"]) == np.ndarray, "wrong input types"

        self._last_rays = action["rays"]
        if not self._done:
            directions = self._camera.calculate_directions(action["rays"], self._curr_T)
            if directions is not None:
                init_point = np.asmatrix(self._curr_position)
                x, v = self._create_observation(init_point, directions)
                reward = self._reward_counter.compute_reward(action["map"], self._last_T)
                self._to_next()
                observation = {"T": self._create_matrix_array(), "points": np.transpose(x), "values": v}
            else:
                reward = self._reward_counter.compute_reward(action["map"], self._last_T)
                self._to_next()
                observation = {"T": self._create_matrix_array(), "points": None, "values": None}

            return observation, reward, self._done, None
        else:
            return None, None, True, None

    def _render(self, mode='human', close=False):
        if not self._done:
            if mode == 'human':
                if not self._render_init:
                    from lidar_gym.visualiser import plot3d
                    self.plotter = plot3d.Plotter()
                    self._render_init = True
                if self._next_timestamp > 1:
                    g_t, a_m, sensor = self._reward_counter.get_render_data()
                    self.plotter.plot_action(g_t, a_m, np.transpose(self._rays_endings), self._voxel_size, sensor)
            if mode == 'ASCII':
                print(ray_string(self._last_rays.T))

    def _seed(self, seed=None):
        map_parser.set_seed(seed)

    def _to_next(self):
        if not self._done:
            if self._next_timestamp == self._map_length:
                self._last_T = self._curr_T
                self._curr_T = None
                self._done = True
                self._next_timestamp += 1
                return
            self._last_T = self._curr_T
            self._curr_T = self._T_matrices[self._next_timestamp]
            self._curr_position = processing.transform_points(self._initial_position, self._curr_T)
            self._next_timestamp += 1

    def _create_observation(self, init_point, directions):
        if len(directions) == 0:
            return None, None

        init_points = np.repeat(init_point, len(directions), axis=0)
        self._rays_endings, v = self._map.trace_rays(np.transpose(init_points),
                                                      np.transpose(directions),
                                                      self._lidar_range, const_min_value, const_max_value, 0)

        tmp_map = vm.VoxelMap()
        tmp_map.voxel_size = self._voxel_size
        tmp_map.free_update = - 1.0
        tmp_map.hit_update = 1.0
        init_points = np.repeat(init_point, len(v), axis=0)
        tmp_map.update_lines(np.transpose(init_points), self._rays_endings)
        # correct empty pts

        bools = processing.values_to_bools(v)
        indexes_empty = np.asarray(np.where(~bools))
        if indexes_empty.size > 0:
            free_pts = np.asmatrix(self._rays_endings[:, indexes_empty])
            # Here if indexes_empty is of size 1, free pts is of wrong shape and needs transposing
            if free_pts.shape[0] == 3:
                tmp_map.set_voxels(free_pts, np.zeros((free_pts.shape[1],)), -np.ones((free_pts.shape[1],)))
            else:
                if free_pts.shape == (1, 3):
                    tmp_map.set_voxels(np.transpose(free_pts),
                                       np.zeros((free_pts.shape[0],)), -np.ones((free_pts.shape[0],)))

        x, l, v = tmp_map.get_voxels()
        return x, v

    def _create_matrix_array(self):
        """
        create array of transformation matrix with current position
        :return: numpy array Nx(4x4)
        """
        t = self._next_timestamp - 1
        if self._map_length >= (t + self._T_forecast):
            return self._T_matrices[t:(t + self._T_forecast)]
        else:
            diff = (self._map_length - t)
            ret = np.zeros((diff, 4, 4))
            if diff > 0:
                ret[0:diff] = self._T_matrices[t:t + diff]
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


# ------------------------------- INHERITED ENVIRONMENTS -----------------------------------------------
class LidarTrain(LidarGym):
    """
    Inherited environment, simplified for basic training.
    observation is dictionary of sparse and ground thruth maps: {'x': np.array, 'y': np.array}
    and action is dictionary of reconstructed map and planed rays {'map' np.array, 'rays': np.array}
    """
    def __init__(self, map_voxel_shape, fov, density, max_rays, voxel_size, lidar_range):

        forecast = 0

        self._shift_T = np.eye(4, dtype=float)
        self._shift_T[0, 3] = -0.25 * map_voxel_shape[0] * voxel_size
        self._shift_T[1, 3] = -0.5 * map_voxel_shape[1] * voxel_size
        self._shift_T[2, 3] = -0.5 * map_voxel_shape[2] * voxel_size

        super(LidarTrain, self).__init__(lidar_range, voxel_size, max_rays, density, fov,
                                         forecast, map_voxel_shape, self._shift_T)

        self.action_space = spaces.Dict({'map': LidarBox(low=-100, high=100, shape=map_voxel_shape),
                                         'rays': LidarMultiBinary(n=density, maxrays=max_rays)})
        self.observation_space = spaces.Box(low=-100, high=100, shape=map_voxel_shape)
        self.reward_range = (-float('Inf'), 0)

        self._action_generator = LidarMultiBinary((density[1], density[0]), max_rays)

        # lidarv0 specific:
        self._obs_voxel_map = None

    def _reset(self):
        self._obs_voxel_map = vm.VoxelMap()
        self._obs_voxel_map.voxel_size = self._voxel_size
        self._obs_voxel_map.free_update = - 1.0
        self._obs_voxel_map.hit_update = 1.0
        self._obs_voxel_map.occupancy_threshold = 0.0
        obs = super(LidarTrain, self)._reset()
        self.curr_T = obs['T'][0]
        return np.zeros(shape=self._input_map_shape)

    def _step(self, action):
        assert action['rays'].dtype is np.dtype('bool')
        obs, rew, done, info = super(LidarTrain, self)._step({'rays': action['rays'].T, 'map': action['map']})
        obs = self._preprocess_obs(obs)
        return obs, rew, done, info

    def _preprocess_obs(self, obs):
        new_points = np.transpose(obs['points'])
        l = np.zeros((new_points.shape[1],), dtype=np.float64)
        new_vals = obs['values']
        last_vals = self._obs_voxel_map.get_voxels(new_points, l)
        last_vals[np.isnan(last_vals)] = 0
        new_vals = last_vals + new_vals
        self._obs_voxel_map.set_voxels(new_points, l, new_vals)

        # get CNN input
        points, values = self._cuboid_getter.get_map_cuboid(self._obs_voxel_map, self.curr_T, self._shift_T)
        points = np.asmatrix(points/self._voxel_size)
        points = np.transpose(points)
        points = np.asarray(points, dtype=int)
        ret = np.zeros(shape=self._input_map_shape, dtype=float)
        values[np.isnan(values)] = 0
        ret[points[0], points[1], points[2]] = values

        # get normalized ground truth
        points, values = self._cuboid_getter.get_map_cuboid(self._map, self.curr_T, self._shift_T)
        points = np.asmatrix(points / self._voxel_size)
        points = np.transpose(points)
        points = np.asarray(points, dtype=int)
        gt = np.zeros(shape=self._input_map_shape, dtype=float)
        with warnings.catch_warnings():
            # suppress warning for NaN / NaN
            warnings.simplefilter("ignore")
            values = np.asarray(values // np.abs(values))
        values[np.isnan(values)] = 0
        gt[points[0], points[1], points[2]] = values

        self.curr_T = obs['T'][0]
        return {'X': ret, 'Y': gt}


# ------------------------------------ Ready to use environments ---------------------------------------
class Lidarv0(LidarTrain):

    def __init__(self):
        super(Lidarv0, self).__init__((320, 320, 32), (120, 90), (160, 120), 200, 0.2, 48)


class LidarSmallv0(LidarTrain):

    def __init__(self):
        super(LidarSmallv0, self).__init__((160, 160, 16), (120, 90), (120, 90), 100, 0.4, 48)


class LidarToyv0(LidarTrain):

    def __init__(self):
        super(LidarToyv0, self).__init__((80, 80, 8), (120, 90), (40, 30), 15, 0.8, 48)


class Lidarv1(LidarGym):
    """
    Inherited environment prepared for use, based on work in paper: https://arxiv.org/abs/1708.02074.
    The most complex environment.
    """
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
        self.action_space = spaces.Dict({"rays": LidarMultiBinary((int(density[1]), int(density[0])), max_rays),
                                         "map": LidarBox(low=min_val,
                                                         high=max_val,
                                                         shape=map_voxel_shape)})

        self.observation_space = spaces.Dict({"T": spaces.Box(low=min_val, high=max_val, shape=(100, 4, 4)),
                                              "points": spaces.Box(low=min_val, high=max_val, shape=(max_obs_pts, 3)),
                                              "values": spaces.Box(low=min_val, high=max_val, shape=(max_obs_pts, 1))})
        self.reward_range = (-float('Inf'), 0)

        super(Lidarv1, self).__init__(lidar_range, voxel_size, max_rays, density, fov,
                                      forecast, map_voxel_shape, shift_T)

    def _reset(self):
        # first obs should not be zeros
        first_action = np.zeros(shape=self._input_map_shape)
        obs, _, _, _ = self._step(first_action)
        super(Lidarv1, self)._reset()
        return obs


class Lidarv2(Lidarv0):
    """
    action space is only reconstructed map, rays are chosen randomly, suited for supervised learning
    """
    def __init__(self):
        super(Lidarv2, self).__init__()
        self.action_space = spaces.Box(low=-100, high=100, shape=(320, 320, 32))
        self._action_generator = LidarMultiBinary((160, 120), 200)

    def _reset(self):
        ret = super(Lidarv2, self)._reset()
        ret, _, _, _ = self._step(ret)
        return ret

    def _step(self, action):
        rand_rays = self._action_generator.sample()
        return super(Lidarv2, self)._step({'map': action, 'rays': rand_rays})


class LidarSmallv2(LidarSmallv0):
    """
    action space is only reconstructed map, rays are chosen randomly, suited for supervised learning
    """
    def __init__(self):
        super(LidarSmallv2, self).__init__()
        self.action_space = spaces.Box(low=-100, high=100, shape=(160, 160, 16))
        self._action_generator = LidarMultiBinary((120, 90), 100)

    def _reset(self):
        ret = super(LidarSmallv2, self)._reset()
        ret, _, _, _ = self._step(ret)
        return ret

    def _step(self, action):
        rand_rays = self._action_generator.sample()
        return super(LidarSmallv2, self)._step({'map': action, 'rays': rand_rays})


class LidarToyv2(LidarToyv0):
    """
    action space is only reconstructed map, rays are chosen randomly, suited for supervised learning
    """
    def __init__(self):
        super(LidarToyv2, self).__init__()
        self.action_space = spaces.Box(low=-100, high=100, shape=(80, 80, 8))
        self._action_generator = LidarMultiBinary((40, 30), 15)

    def _reset(self):
        ret = super(LidarToyv2, self)._reset()
        ret, _, _, _ = self._step(ret)
        return ret

    def _step(self, action):
        rand_rays = self._action_generator.sample()
        return super(LidarToyv2, self)._step({'map': action, 'rays': rand_rays})


class LidarEval(Lidarv0):
    """
    one map only for agent evaluation
    """
    def __init__(self):
        super(LidarEval, self).__init__()

    def _reset(self):
        self._obs_voxel_map = vm.VoxelMap()
        self._obs_voxel_map.voxel_size = self._voxel_size
        self._obs_voxel_map.free_update = - 1.0
        self._obs_voxel_map.hit_update = 1.0
        self._obs_voxel_map.occupancy_threshold = 0.0

        # reset values
        self._next_timestamp = 0
        self._curr_position = None
        self._curr_T = None
        self._last_T = None
        self._done = False
        self._render_init = False
        self._rays_endings = None
        self._map_length = None
        self._map = None
        self._T_matrices = None

        print('RESETING')
        # parse new map
        self._map, self._T_matrices = self._maps.get_validation_map()
        self._reward_counter.reset(self._map)
        self._map_length = len(self._T_matrices)
        self._curr_T = self._T_matrices[0]
        self._to_next()

        self.curr_T = self._curr_T
        return np.zeros(shape=self._input_map_shape)


class LidarSmallEval(LidarSmallv0):
    """
    one map only for agent evaluation
    """
    def __init__(self):
        super(LidarSmallEval, self).__init__()

    def _reset(self):
        self._obs_voxel_map = vm.VoxelMap()
        self._obs_voxel_map.voxel_size = self._voxel_size
        self._obs_voxel_map.free_update = - 1.0
        self._obs_voxel_map.hit_update = 1.0
        self._obs_voxel_map.occupancy_threshold = 0.0

        # reset values
        self._next_timestamp = 0
        self._curr_position = None
        self._curr_T = None
        self._last_T = None
        self._done = False
        self._render_init = False
        self._rays_endings = None
        self._map_length = None
        self._map = None
        self._T_matrices = None

        print('\nRESETING')
        # parse new map
        self._map, self._T_matrices = self._maps.get_validation_map()
        self._reward_counter.reset(self._map)
        self._map_length = len(self._T_matrices)
        self._curr_T = self._T_matrices[0]
        self._to_next()

        self.curr_T = self._curr_T
        return np.zeros(shape=self._input_map_shape)


class LidarToyEval(LidarToyv0):
    """
    one map only for agent evaluation
    """
    def __init__(self):
        super(LidarToyEval, self).__init__()

    def _reset(self):
        self._obs_voxel_map = vm.VoxelMap()
        self._obs_voxel_map.voxel_size = self._voxel_size
        self._obs_voxel_map.free_update = - 1.0
        self._obs_voxel_map.hit_update = 1.0
        self._obs_voxel_map.occupancy_threshold = 0.0

        # reset values
        self._next_timestamp = 0
        self._curr_position = None
        self._curr_T = None
        self._last_T = None
        self._done = False
        self._render_init = False
        self._rays_endings = None
        self._map_length = None
        self._map = None
        self._T_matrices = None

        print('RESETING')
        # parse new map
        self._map, self._T_matrices = self._maps.get_validation_map()
        self._reward_counter.reset(self._map)
        self._map_length = len(self._T_matrices)
        self._curr_T = self._T_matrices[0]
        self._to_next()

        self.curr_T = self._curr_T
        return np.zeros(shape=self._input_map_shape)
