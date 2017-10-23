"""Example of pykitti.raw usage."""
import itertools
import numpy as np
import pykitti
import voxel_map as vm


def transform_points(points, transform_mat):
    transposed = np.transpose(points)
    transposed = np.delete(transposed, 3, axis=0)
    to_conc = np.ones((1, np.shape(transposed)[1]))
    transposed = np.concatenate((transposed, to_conc), 0)
    mult = np.dot(transform_mat, transposed)
    return np.transpose(mult[0:-1])


# Change this to the directory where you store KITTI data
basedir = '/home/zdeeno/Downloads/dataset'
# Specify the dataset to load
date = '2011_09_26'
drive = '0009'

# VoxelMap initialization
m = vm.VoxelMap()
m.voxel_size = 0.5
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

anchor_initial = np.zeros((1, 4))

np.set_printoptions(precision=4, suppress=True)

print '\nThis map has', size, 'timestamps.'
# Grab some data
for i in range(size):
    print '\nProcessing point cloud from position: ', i
    transform_matrix = next(iter(itertools.islice(dataset.oxts, i, None))).T_w_imu
    anchor = transform_points(anchor_initial, transform_matrix)
    velo_points = next(iter(itertools.islice(dataset.velo, i, None)))
    pts = transform_points(velo_points, transform_matrix)
    anchors = np.tile(np.transpose(anchor), (1, len(pts)))
    m.update_lines(anchors, np.transpose(pts))
    #print '\n------------- AT ANCHOR ', anchor, ' ------------\n'
    #print pts

# Display some of the data

