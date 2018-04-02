import itertools

import numpy as np
import pykitti
import voxel_map as vm
import pkg_resources as pkg
import os
import random

from lidar_gym.tools import math_processing as mp


def set_seed(seed=None):
    random.seed(seed)


def get_seed():
    return random.seed


class MapParser:

    def __init__(self, voxel_size):
        self._voxel_size = voxel_size
        self._basedir = '/usr/local/kitti_dataset'
        assert os.path.isdir(self._basedir), 'Your kitti dataset must be located at usr/local/kitti_dataset,' \
                                             ' see download_dataset.sh'
        # set of drives is hardcoded due to drives in bash script 'download_dataset.sh'
        self._drives = []
        self._date = '2011_09_26'
        # city
        self._drives.extend(['0002', '0005', '0011'])
        # residential
        self._drives.extend(['0019', '0020', '0022', '0039', '0061'])
        # road
        self._drives.extend(['0027', '0015', '0070', '0016', '0047'])
        set_seed()

    def get_next_map(self):
        # VoxelMap initialization
        m = vm.VoxelMap()
        m.voxel_size = self._voxel_size
        m.free_update = - 1.0
        m.hit_update = 1.0
        m.occupancy_threshold = 0.0

        # Load the data. Optionally, specify the frame range to load.
        index = random.randint(0, len(self._drives)-1)
        dataset = pykitti.raw(self._basedir, self._date, self._drives[index])
        size = len(dataset)

        T_matrixes = []
        anchor_initial = np.zeros((1, 4))
        np.set_printoptions(precision=4, suppress=True)
        iterator_oxts = iter(itertools.islice(dataset.oxts, 0, None))
        iterator_velo = iter(itertools.islice(dataset.velo, 0, None))
        T_imu_to_velo = np.linalg.inv(dataset.calib.T_velo_imu)

        print('\nParsing drive', self._drives[index], 'with length of', size, 'timestamps.\n')
        # Grab some data
        for i in range(size):
        # for i in range(8):
            if (i % 10) == 0:
                print('.', end='', flush=True)
            transform_matrix = np.dot(next(iterator_oxts).T_w_imu, T_imu_to_velo)
            T_matrixes.append(np.asarray(transform_matrix))
            anchor = mp.transform_points(anchor_initial, transform_matrix)
            velo_points = next(iterator_velo)
            pts = mp.transform_points(velo_points, transform_matrix)
            anchors = np.tile(np.transpose(anchor), (1, len(pts)))
            m.update_lines(anchors, np.transpose(pts))

        return m, T_matrixes
