import numpy as np
import voxel_map as vm
import warnings


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
        mins = np.zeros((1, 3))
        maxs = map_size
        getter_size = map_size/voxel_size
        num_pts = int(getter_size[0] * getter_size[1] * getter_size[2])

        self.l = np.zeros((num_pts,), dtype=np.float64)
        self._cuboid_points = np.zeros((num_pts, 3))
        counter = 0
        # create cuboid of map size
        print('\nCreating cuboid getters ...')
        for x in np.arange(mins[0, 0], maxs[0], voxel_size):
            for y in np.arange(mins[0, 1], maxs[1], voxel_size):
                for z in np.arange(mins[0, 2], maxs[2], voxel_size):
                    self._cuboid_points[counter] = np.asmatrix((x, y, z)) + voxel_size/2
                    counter = counter + 1
        print('\nDone')

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


class RewardCounter:

    def __init__(self, ground_truth, voxel_size, map_shape, weight):
        assert type(ground_truth) == vm.VoxelMap

        self._ground_truth = ground_truth
        self._voxel_size = voxel_size
        self._a_s_size = map_shape/voxel_size
        self._weight = weight

        # calculate shift matrix
        self._shift_T = np.eye(4, dtype=float)
        self._shift_T[0, 3] = -0.25 * map_shape[0]
        self._shift_T[1, 3] = -0.5 * map_shape[1]
        self._shift_T[2, 3] = -0.5 * map_shape[2]

        self._cuboid_getter = CuboidGetter(voxel_size, map_shape)

    def compute_reward(self, action_map, T):
        """
        Computes reward from input map.
        :param action_map: map is part of action space
        :param T: transform matrix 4x4 to local coordinate system
        :return: double reward (-inf, 0), higher is better
        """
        assert np.array_equal(action_map.shape, self._a_s_size)

        # obtain ground truth local map values
        points, values_g_t = self._cuboid_getter.get_map_cuboid(self._ground_truth, T, self._shift_T)
        with warnings.catch_warnings():
            # suppress warning for NaN / NaN
            warnings.simplefilter("ignore")
            values_g_t = values_g_t // np.abs(values_g_t)
        values_g_t[np.isnan(values_g_t)] = 0

        # obtain action space local map values
        points = points / self._voxel_size
        points = points.astype(int)
        values_a_m = action_map[points[:, 0], points[:, 1], points[:, 2]]

        # obtain weights
        weights = np.zeros(len(values_g_t))
        weights[np.abs(values_g_t) > 0] = self._weight

        # calculate reward
        reward = np.sum(-weights * (np.log(1 + np.exp(-values_a_m * values_g_t))))
        return reward
