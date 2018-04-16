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

VALIDATION = ['0009']


def _iterate_map(dataset, size, voxel_size):
    m = vm.VoxelMap()
    m.voxel_size = voxel_size
    m.free_update = - 1.0
    m.hit_update = 1.0
    m.occupancy_threshold = 0.0

    T_matrixes = []
    anchor_initial = np.zeros((1, 4))
    np.set_printoptions(precision=4, suppress=True)
    iterator_oxts = iter(itertools.islice(dataset.oxts, 0, None))
    iterator_velo = iter(itertools.islice(dataset.velo, 0, None))
    T_imu_to_velo = np.linalg.inv(dataset.calib.T_velo_imu)

    # next line depends on lidar-car setup in metersW
    car_x = np.mgrid[-0.8:0.8:voxel_size, -1.2:1.2:voxel_size, -1.6:0:voxel_size]
    car_x = car_x.reshape(3, -1).T
    car_x = np.asmatrix(car_x)
    car_v = -np.ones((len(car_x), ))
    car_l = np.zeros((len(car_x), ))

    # Grab some data
    next_oxts = next(iterator_oxts)
    i = 0
    while next_oxts is not None:
        if (i % 10) == 0:
            print('.', end='', flush=True)
        transform_matrix = np.dot(next_oxts.T_w_imu, T_imu_to_velo)
        T_matrixes.append(np.asarray(transform_matrix))
        anchor = mp.transform_points(anchor_initial, transform_matrix)
        velo_points = next(iterator_velo)
        pts = mp.transform_points(velo_points, transform_matrix)
        anchors = np.tile(np.transpose(anchor), (1, len(pts)))
        m.update_lines(anchors, np.transpose(pts))

        # we want car to give negative voxels
        car_pts = mp.transform_points(car_x, transform_matrix)
        m.set_voxels(car_pts.T, car_l, car_v)

        # asking for next lidar position to check, whether to continue iterating
        next_oxts = next(iterator_oxts, None)
        i += 1

    return m, T_matrixes


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

        set_seed()

    def get_next_map(self):
        # Load the data. Optionally, specify the frame range to load.
        index = random.randint(0, len(self._drives)-1)
        serialized = os.path.join(self._basedir, self._drives[index] + '.vs')
        # load serialized map if available
        if os.path.isfile(serialized):
            print('Deserializing map number:' + self._drives[index])
            return pickle.load(open(serialized, 'rb'))

        dataset = pykitti.raw(self._basedir, self._date, self._drives[index])
        size = len(dataset)

        print('\nParsing drive', self._drives[index], 'with length of', size, 'timestamps.\n')
        # Grab some data
        return _iterate_map(dataset, size, self._voxel_size)

    def get_validation_map(self):
        index = 0
        serialized = os.path.join(self._basedir, VALIDATION[index] + '.vs')
        # load serialized map if available
        if os.path.isfile(serialized):
            print('Deserializing validation map 0009:')
            return pickle.load(open(serialized, 'rb'))

        dataset = pykitti.raw(self._basedir, self._date, VALIDATION[index])
        size = len(dataset)

        print('\nParsing drive', VALIDATION[index], 'with length of', size, 'timestamps.\n')
        # Grab some data
        return _iterate_map(dataset, size, self._voxel_size)


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
        # self._drives.extend(DRIVES_CITY)
        self._drives.extend(VALIDATION)

        for drive in self._drives:
            dataset = pykitti.raw(self._basedir, self._date, drive)
            size = len(dataset)
            iterator_oxts = iter(itertools.islice(dataset.oxts, 0, None))
            iterator_velo = iter(itertools.islice(dataset.velo, 0, None))
            T_imu_to_velo = np.linalg.inv(dataset.calib.T_velo_imu)
            print('\nParsing drive', drive, 'with length of', size, 'timestamps.\n')
            next_oxts = next(iterator_oxts)
            i = 0
            while next_oxts is not None:
                # for i in range(100):
                if (i % 10) == 0:
                    print('.', end='', flush=True)
                transform_matrix = np.dot(next_oxts.T_w_imu, T_imu_to_velo)
                velo_points = next(iterator_velo)
                next_oxts = next(iterator_oxts, None)
                i += 1


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
        self._drives.extend(VALIDATION)

        for drive in self._drives:
            # Load the data. Optionally, specify the frame range to load.
            dataset = pykitti.raw(self._basedir, self._date, drive)
            size = len(dataset)

            print('\nParsing drive', drive, 'with length of', size, 'timestamps.\n')

            m, T_matrixes = _iterate_map(dataset, size, voxel_size)
            serialize = m, T_matrixes
            print('Serializing')
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

