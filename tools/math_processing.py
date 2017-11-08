import numpy as np


def transform_points(points, transform_mat):
    transposed = np.transpose(points)
    transposed = np.delete(transposed, 3, axis=0)
    to_conc = np.ones((1, np.shape(transposed)[1]))
    transposed = np.concatenate((transposed, to_conc), 0)
    mult = np.dot(transform_mat, transposed)
    return np.transpose(mult[0:-1])


def compute_reward(map1, map2):
    #TODO implement logistic loss
    pass