import itertools
import os
import numpy as np
import pykitti
import mayavi.mlab

# ONLY PYTHON 2 --- MAYAVI

basedir = '/home/zdeeno/Downloads/dataset'
# Specify the dataset to load
date = '2011_09_26'
drive = '0009'

dataset = pykitti.raw(basedir, date, drive)
i = 5
velo_points = next(iter(itertools.islice(dataset.velo, i, None)))


fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
mayavi.mlab.points3d(
    velo_points[:, 0],  # x
    velo_points[:, 1],  # y
    velo_points[:, 2],  # z
    velo_points[:, 2],  # Height data used for shading
    mode="point",  # How to render each point {'point', 'sphere' , 'cube' }
    colormap='spectral',  # 'bone', 'copper',
    # color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
    scale_factor=100,  # scale of the points
    line_width=10,  # Scale of the line, if any
    figure=fig,
)
# velo[:, 3], # reflectance values
mayavi.mlab.show()
