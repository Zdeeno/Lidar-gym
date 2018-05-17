import numpy as np
import voxel_map as vm
import warnings
from itertools import product


def values_to_bools(v):
    for i in range(len(v)):
        if np.isnan(v[i]):
            v[i] = False
        else:
            v[i] = True
    return np.asarray(v, dtype=np.bool)


def get_shape(array):
    if array.ndim == 1:
        return [1, array.shape[0]]
    else:
        return np.shape(array)


def transform_points(points, transform_mat):
    """
    Method adds ones to the end of all vectors and transform them using transform matrix.
    :param points: set of 3D row vectors, for example: [x1 y1 z1; x2 y2 z2; ...]
    :param transform_mat: 4x4 transform matrix
    :return: transformed vectors as set of 3D row vectors
    """
    transposed = np.transpose(points)
    if len(transposed) > 3:
        transposed = np.delete(transposed, 3, axis=0)
    to_conc = np.ones((1, get_shape(transposed)[1]))
    transposed = np.concatenate((transposed, to_conc), 0)
    mult = np.dot(transform_mat, transposed)
    return np.transpose(mult[0:-1])


class CuboidGetter:

    def __init__(self, voxel_size, map_size):
        map_size = np.asarray(map_size)
        mins = np.zeros((1, 3))
        maxs = map_size/voxel_size
        getter_size = map_size/voxel_size
        num_pts = int(getter_size[0] * getter_size[1] * getter_size[2])

        self.l = np.zeros((num_pts,), dtype=np.float64)
        self.voxel_size = voxel_size
        tmp = zip(tuple(mins[0].astype(int)), tuple(maxs.astype(int)), (1, 1, 1))
        self._cuboid_points = list(product(*(range(*x) for x in tmp)))
        self._cuboid_points = np.asarray(self._cuboid_points)*voxel_size + voxel_size/2

    def get_map_cuboid(self, voxel_map, T_pos, T_shift):
        """
        Obtain cuboid from Voxel_map class
        :param voxel_map: Voxel_map instance
        :param T_pos: 4x4 transformation matrix to the position of sensor
        :param T_shift: 4x4 transformation matrix, translating cuboid around the sensor
        :return: points (Nx3), values (1xN)
        """
        T = np.dot(T_pos, T_shift)
        points = transform_points(self._cuboid_points, T)
        values = voxel_map.get_voxels(np.transpose(points), self.l)
        return self._cuboid_points, values

    def update_map_cuboid(self, voxel_map, map_action, T_pos, T_shift):
        T = np.dot(T_pos, T_shift)
        points = transform_points(self._cuboid_points, T)
        values_old = voxel_map.get_voxels(np.transpose(points), self.l)
        voxel_cuboid = np.round(self._cuboid_points*self.voxel_size)
        values_update = map_action[voxel_cuboid]
        values_new = values_old + values_update
        voxel_map.set_voxels(points, self.l, values_new)


class RewardCounter:

    def __init__(self, voxel_size, map_shape, T_shift, cuboid_getter):
        self._ground_truth = None
        self._voxel_size = voxel_size
        self._a_s_size = map_shape/voxel_size

        # calculate shift matrix
        self._shift_T = T_shift

        self._cuboid_getter = cuboid_getter
        self._last_action = None
        self._last_T = None

    def reset(self, ground_truth):
        assert type(ground_truth) == vm.VoxelMap
        self._last_action = None
        self._last_T = None
        self._ground_truth = ground_truth

    def compute_reward(self, action_map, T):
        """
        Computes reward from input map.
        :param action_map: map is part of action space
        :param T: transform matrix 4x4 to local coordinate system
        :return: double reward (-inf, 0), higher is better
        """
        assert np.array_equal(action_map.shape, self._a_s_size), print(action_map.shape, self._a_s_size)

        self._last_action = action_map
        self._last_T = T
        # obtain ground truth local map values
        points, values_g_t = self._cuboid_getter.get_map_cuboid(self._ground_truth, T, self._shift_T)
        with warnings.catch_warnings():
            # suppress warning for NaN / NaN
            warnings.simplefilter("ignore")
            values_g_t = np.asarray(values_g_t // np.abs(values_g_t))
        values_g_t = np.nan_to_num(values_g_t)

        # obtain action space local map values
        points = points / self._voxel_size
        points = points.astype(int)
        values_a_m = action_map[points[:, 0], points[:, 1], points[:, 2]]

        # obtain weights
        weights_positive = 0.5/np.sum(values_g_t > 0)
        weights_negative = 0.5/np.sum(values_g_t < 0)
        weights = np.zeros(len(values_g_t))
        weights[values_g_t > 0] = weights_positive
        weights[values_g_t < 0] = weights_negative

        # calculate reward as log loss: np.sum(-weights * (np.log(1 + np.exp(-values_a_m * values_g_t))))
        a = -values_a_m * values_g_t
        b = np.maximum(0.0, a)
        t = b + np.log(np.exp(-b) + np.exp(a - b))
        reward = np.sum(-weights * t)
        return reward

    def get_render_data(self):
        points, values_g_t = self._cuboid_getter.get_map_cuboid(self._ground_truth, self._last_T, self._shift_T)
        getter = points / self._voxel_size
        getter = getter.astype(int)
        values_a_m = self._last_action[getter[:, 0], getter[:, 1], getter[:, 2]]
        T = np.dot(self._last_T, self._shift_T)
        to_render = transform_points(points, T)
        values_g_t[np.isnan(values_g_t)] = 0
        sensor_pos = transform_points(np.asmatrix((0, 0, 0)), self._last_T)
        return np.copy(to_render[values_g_t > 0]), np.copy(to_render[values_a_m > 0]), sensor_pos
