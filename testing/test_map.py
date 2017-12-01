import build.voxel_map as vm
import numpy as np


def create_test_map():
    m = vm.VoxelMap()
    m.voxel_size = 1
    m.free_update = - 1.0
    m.hit_update = 1.0

    #floor [-5-5, 0-100, -2]
    z = -2
    ymin = 0
    ymax = 100
    xmin = -5
    xmax = 5
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            m.set_voxels(np.asarray([[x], [y], [z]]), np.asarray([0]), np.asarray([1]))

    #right wall [5, 0-100, -2-2]
    x = 5
    ymin = 0
    ymax = 100
    zmin = -2
    zmax = 2
    for y in range(ymin, ymax):
        for z in (zmin, zmax):
            m.set_voxels(np.asarray([[x], [y], [z]]), np.asarray([0]), np.asarray([1]))

    #left wall [-5, 0-100, -2-2]
    x = 5
    ymin = 0
    ymax = 100
    zmin = -2
    zmax = 2
    for y in range(ymin, ymax):
        for z in (zmin, zmax):
            m.set_voxels(np.asarray([[x], [y], [z]]), np.asarray([0]), np.asarray([1]))

    stay = np.eye(4)
    left = np.asarray([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    right = np.asarray([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    forward = np.eye(4)
    forward[1, 3] = 10
    transform_mat = [stay, left, right, forward]

    return m, transform_mat
