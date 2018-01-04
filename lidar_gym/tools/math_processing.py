import numpy as np
import voxel_map as vm
import math


def values_to_bools(v):
    for i in range(len(v)):
        if np.isnan(v[i]):
            v[i] = False
        else:
            v[i] = True
    return np.asarray(v, dtype=np.bool)


def get_numpy_shape(array):
    if array.ndim == 1:
        return [1, np.shape(array)[0]]
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
    to_conc = np.ones((1, get_numpy_shape(transposed)[1]))
    transposed = np.concatenate((transposed, to_conc), 0)
    mult = np.dot(transform_mat, transposed)
    return np.transpose(mult[0:-1])


def _reward_formula(y1, y2):
    """
    Computes logistic loss per one element.
    :param y1, y2: int +1 or -1
    :return: reward: double in interval (-inf, 0)
    """
    weight = 1
    return -weight*math.log(1 + math.exp(-y1*y2))


class CuboidGetter:

    def __init__(self, voxel_size, map_size, shift):
        mins = np.zeros((1, 3)) - shift
        maxs = map_size - shift
        getter_size = map_size/voxel_size + (1, 1, 1)
        num_pts = int(getter_size[0] * getter_size[1] * getter_size[2])
        self._cuboid_points = np.zeros((num_pts, 3))
        counter = 0
        print('\nCreating cuboid getters ...')
        for x in np.arange(mins[0, 0], maxs[0] + voxel_size, voxel_size):
            for y in np.arange(mins[0, 1], maxs[1] + voxel_size, voxel_size):
                for z in np.arange(mins[0, 2], maxs[2] + voxel_size, voxel_size):
                    self._cuboid_points[counter] = np.asmatrix((x, y, z))
                    counter = counter + 1
        self.l = np.zeros((len(self._cuboid_points),), dtype=np.float64)
        print('\nDone')

    def get_map_cuboid(self, voxel_map, T=None):
        '''
        Obtain cuboid from Voxel_map class
        :param map: Voxel_map instance
        :param T: transformation matrix
        :return: points (Nx3), values (1xN)
        '''
        if T is None:
            values = voxel_map.get_voxels(np.transpose(self._cuboid_points), self.l)
            return self._cuboid_points, values
        else:
            points = transform_points(self._cuboid_points, T)
            values = voxel_map.get_voxels(np.transpose(points), self.l)
            return self._cuboid_points, values


class RewardCounter:

    def __init__(self, ground_truth, voxel_size, action_space_size, shift_length):
        assert type(ground_truth) == vm.VoxelMap

        self._ground_truth = ground_truth
        self._voxel_size = voxel_size
        self._a_s_size = action_space_size/voxel_size + (1, 1, 1)
        self._shift_rate = np.asarray(shift_length)/voxel_size
        self._cuboid_getter = CuboidGetter(voxel_size, action_space_size, shift_length)

    def compute_reward(self, action_map, T):
        """
        Computes reward from input map.
        :param action_map: map is part of action space
        :param T: transform matrix 4x4
        :return: double reward (-inf, 0), higher is better
        """
        assert np.array_equal(action_map.shape, self._a_s_size)

        reward = 0
        points, values = self._cuboid_getter.get_map_cuboid(self._ground_truth, T)
        points = (points/self._voxel_size) + self._shift_rate
        for idx, point in enumerate(np.round(points).astype(int)):
            if values[idx] < 0:
                reward = reward + _reward_formula(action_map[point[0], point[1], point[2]], -1)
                continue
            if values[idx] > 0:
                reward = reward + _reward_formula(action_map[point[0], point[1], point[2]], 1)
        return reward
