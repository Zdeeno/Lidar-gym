import numpy as np
import build.voxel_map as vm
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
    :return: double reward in interval (-inf, 0)
    """
    weight = 1
    return -weight*math.log(1 + math.exp(-y1*y2))


class RewardCounter:

    def __init__(self, ground_truth, voxel_size, action_space_size, shift_ratios):
        assert type(ground_truth) == vm.VoxelMap

        self._shift_ratios = shift_ratios
        self._ground_truth = ground_truth
        self._voxel_size = voxel_size
        self._a_s_size = action_space_size

    def _create_queries(self, action_map):
        """
        Method create queries to ground truth Voxel_map class.
        :param action_map: map is part of action space
        :return: set of 3D row vectors, for example: [x1 y1 z1; x2 y2 z2; ...]
        """
        init_true = False
        init_false = False
        true_mat = None
        false_mat = None
        for a in range(self._a_s_size[0]):
            for b in range(self._a_s_size[1]):
                for c in range(self._a_s_size[2]):
                    if action_map[a][b][c]:
                        if not init_true:
                            to_conc = np.asarray((a, b, c)) * self._voxel_size
                            to_conc = (to_conc - (to_conc * self._shift_ratios)) + self._voxel_size / 2
                            true_mat = np.asmatrix(to_conc)
                            init_true = True
                        else:
                            to_conc = np.asarray((a, b, c)) * self._voxel_size
                            to_conc = (to_conc - (to_conc * self._shift_ratios)) + self._voxel_size / 2
                            true_mat = np.concatenate((true_mat, np.asmatrix(to_conc)), 0)
                    else:
                        if not init_false:
                            to_conc = np.asarray((a, b, c)) * self._voxel_size
                            to_conc = (to_conc - (to_conc * self._shift_ratios)) + self._voxel_size / 2
                            false_mat = np.asmatrix(to_conc)
                            init_false = True
                        else:
                            to_conc = np.asarray((a, b, c)) * self._voxel_size
                            to_conc = (to_conc - (to_conc * self._shift_ratios)) + self._voxel_size / 2
                            false_mat = np.concatenate((false_mat, np.asmatrix(to_conc)), 0)
        return true_mat, false_mat

    def compute_reward(self, action_map, T):
        """
        Computes reward from input map.
        :param action_map: map is part of action space
        :param T: transform matrix 4x4
        :return: double reward (-inf, 0), higher is better
        """
        true_mat, false_mat = self._create_queries(action_map)
        t_inv = np.linalg.inv(T)
        reward = 0

        if true_mat is not None:
            true_mat = transform_points(true_mat, t_inv)
            l = np.zeros((len(true_mat),), dtype=np.float64)
            v_true = self._ground_truth.get_voxels(np.transpose(true_mat), l)
            for value in v_true:
                if value > 0:
                    reward += _reward_formula(1, 1)
                else:
                    reward += _reward_formula(-1, 1)

        if false_mat is not None:
            false_mat = transform_points(false_mat, t_inv)
            l = np.zeros((len(false_mat),), dtype=np.float64)
            v_false = self._ground_truth.get_voxels(np.transpose(false_mat), l)
            for value in v_false:
                if value > 0:
                    reward += _reward_formula(1, -1)
                else:
                    reward += _reward_formula(-1, -1)
            return reward
