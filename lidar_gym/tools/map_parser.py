import itertools

import numpy as np
import pykitti
import voxel_map as vm
import pkg_resources as pkg
import os
import random
import pickle

from lidar_gym.tools import math_processing as mp
from os.path import expanduser


def set_seed(seed=None):
    random.seed(seed)


def get_seed():
    return random.seed


DRIVES_CITY = ['0002', '0001', '0005', '0011', '0013', '0014', '0017', '0018', '0048', '0051',
               '0056', '0057', '0059', '0060', '0084', '0091', '0093', '0095', '0096', '0104']


class MapParser:
    # Class for parsing maps from files or from serialized objects
    def __init__(self, voxel_size):
        self._voxel_size = voxel_size
        home = expanduser("~")
        self._basedir = os.path.join(home, 'kitti_dataset')
        assert os.path.isdir(self._basedir), 'Your kitti dataset must be located in your home directory,' \
                                             '(viz os.path.expanduser) ... see also download_dataset.sh'
        # set of drives is hardcoded due to drives in bash script 'download_dataset.sh'
        self._drives = []
        self._date = '2011_09_26'
        # city
        self._drives.extend(DRIVES_CITY)
        # residential
        # self._drives.extend(['0019', '0020', '0022', '0039', '0061'])
        # road
        # self._drives.extend(['0027', '0015', '0070', '0029', '0032'])

        # testing
        # self._drives = ['0002', '0020', '0027']

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
        serialized = os.path.join(self._basedir, self._drives[index] + '.vs')
        # load serialized map if available
        if os.path.isfile(serialized):
            print('Deserializing map number:' + self._drives[index])
            return pickle.load(open(serialized, 'rb'))

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
        # for i in range(100):
            if (i % 10) == 0:
                print('.', end='', flush=True)
            transform_matrix = np.dot(next(iterator_oxts).T_w_imu, T_imu_to_velo)
            T_matrixes.append(np.asarray(transform_matrix))
            anchor = mp.transform_points(anchor_initial, transform_matrix)
            velo_points = next(iterator_velo)
            pts = mp.transform_points(velo_points, transform_matrix)
            anchors = np.tile(np.transpose(anchor), (1, len(pts)))
            m.update_lines(anchors, np.transpose(pts))
            # TODO: car position must make negative voxels

        return m, T_matrixes


class DatasetTester:
    """Class to check whether is dataset correct"""
    def __init__(self):
        home = expanduser("~")
        self._basedir = os.path.join(home, 'kitti_dataset')
        assert os.path.isdir(self._basedir), 'Your kitti dataset must be located in your home directory,' \
                                             '(viz os.path.expanduser) ... see also download_dataset.sh'
        # set of drives is hardcoded due to drives in bash script 'download_dataset.sh'
        self._drives = []
        self._date = '2011_09_26'
        # city
        self._drives.extend(DRIVES_CITY)


        for drive in self._drives:
            dataset = pykitti.raw(self._basedir, self._date, drive)
            size = len(dataset)
            iterator_oxts = iter(itertools.islice(dataset.oxts, 0, None))
            iterator_velo = iter(itertools.islice(dataset.velo, 0, None))
            T_imu_to_velo = np.linalg.inv(dataset.calib.T_velo_imu)
            print('\nParsing drive', drive, 'with length of', size, 'timestamps.\n')
            for i in range(size):
                # for i in range(100):
                if (i % 10) == 0:
                    print('.', end='', flush=True)
                transform_matrix = np.dot(next(iterator_oxts).T_w_imu, T_imu_to_velo)
                velo_points = next(iterator_velo)


class DatasetSerializer():
    # class for serialization of voxel mapsv
    def __init__(self, voxel_size):
        self._voxel_size = voxel_size
        home = expanduser("~")
        self._basedir = os.path.join(home, 'kitti_dataset')
        assert os.path.isdir(self._basedir), 'Your kitti dataset must be located in your home directory,' \
                                             '(viz os.path.expanduser) ... see also download_dataset.sh'
        # set of drives is hardcoded due to drives in bash script 'download_dataset.sh'
        self._drives = []
        self._date = '2011_09_26'
        # city
        self._drives.extend(DRIVES_CITY)

        for drive in self._drives:
            # VoxelMap initialization
            m = vm.VoxelMap()
            m.voxel_size = self._voxel_size
            m.free_update = - 1.0
            m.hit_update = 1.0
            m.occupancy_threshold = 0.0

            # Load the data. Optionally, specify the frame range to load.
            dataset = pykitti.raw(self._basedir, self._date, drive)
            size = len(dataset)

            T_matrixes = []
            anchor_initial = np.zeros((1, 4))
            np.set_printoptions(precision=4, suppress=True)
            iterator_oxts = iter(itertools.islice(dataset.oxts, 0, None))
            iterator_velo = iter(itertools.islice(dataset.velo, 0, None))
            T_imu_to_velo = np.linalg.inv(dataset.calib.T_velo_imu)

            print('\nParsing drive', drive, 'with length of', size, 'timestamps.\n')
            # Grab some data
            for i in range(size):
                if (i % 10) == 0:
                    print('.', end='', flush=True)
                transform_matrix = np.dot(next(iterator_oxts).T_w_imu, T_imu_to_velo)
                T_matrixes.append(np.asarray(transform_matrix))
                anchor = mp.transform_points(anchor_initial, transform_matrix)
                velo_points = next(iterator_velo)
                pts = mp.transform_points(velo_points, transform_matrix)
                anchors = np.tile(np.transpose(anchor), (1, len(pts)))
                m.update_lines(anchors, np.transpose(pts))

            print('Serializing')
            serialize = m, T_matrixes
            drive = drive + '.vs'
            file = open(os.path.join(self._basedir, drive), 'wb')
            pickle.dump(serialize, file)
            file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='-t for test dataset, -s for serialize dataset')
    parser.add_argument('-t', '--test', help='Test dataset', action='store_true')
    parser.add_argument('-s', '--serialize', help='serialize dataset', action='store_true')
    args = vars(parser.parse_args())
    if args['test']:
        DatasetTester()
        print('\nIf testing doesnt end up with error - dataset is probably correctly downloaded!')

    if args['serialize']:
        voxel_size = 0.2
        print('\nSerializing with voxel size: ' + str(voxel_size))
        DatasetSerializer(0.2)
        print('Serialization done!')

