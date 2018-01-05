import itertools

import numpy as np
import pykitti
import voxel_map as vm

from lidar_gym.tools import math_processing as mp

def parse_map(voxel_size):
    # Change this to the directory where you store KITTI data
    basedir = '/home/zdeeno/Downloads/dataset'
    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0009'

    # VoxelMap initialization
    m = vm.VoxelMap()
    m.voxel_size = voxel_size
    m.free_update = - 1.0
    m.hit_update = 1.0
    m.occupancy_threshold = 0.0

    # Load the data. Optionally, specify the frame range to load.
    dataset = pykitti.raw(basedir, date, drive)
    size = len(dataset)

    # dataset.timestamps: Timestamps are parsed into a list of datetime objects
    # dataset.oxts:       Generator to load OXTS packets as named tuples
    # dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

    print('STARTED ITERATING ARRAYS')
    T_matrixes = []
    anchor_initial = np.zeros((1, 4))
    np.set_printoptions(precision=4, suppress=True)
    iterator_oxts = iter(itertools.islice(dataset.oxts, 0, None))
    iterator_velo = iter(itertools.islice(dataset.velo, 0, None))
    T_imu_to_velo = np.linalg.inv(dataset.calib.T_velo_imu)

    print('\nThis map has', size, 'timestamps.')
    # Grab some data
    # I have error in my dataset, 4 voxel records (bins) are missing -> size - 5
    #for i in range(size-5):
    for i in range(10):
        print('\nProcessing point cloud from position number - ', i)
        transform_matrix = np.dot(next(iterator_oxts).T_w_imu, T_imu_to_velo)
        T_matrixes.append(np.asarray(transform_matrix))
        anchor = mp.transform_points(anchor_initial, transform_matrix)
        velo_points = next(iterator_velo)
        pts = mp.transform_points(velo_points, transform_matrix)
        anchors = np.tile(np.transpose(anchor), (1, len(pts)))
        m.update_lines(anchors, np.transpose(pts))


    # Display some of the data
    return m, T_matrixes
