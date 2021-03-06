import numpy as np
from mayavi import mlab
import math


class Plotter:
    # Plot your map using Mayavi

    def __init__(self):
        self.fig = None

    def plot_action(self, ground_truth, action_map, rays, voxel_size, sensor, dir):
        # TODO: Change mlab.show to mlab.animate!
        self.fig = mlab.figure(size=(1280, 720))
        # plot ground truth

        # here set what should be invisible
        ground_truth = None

        if ground_truth is not None:
            mlab.points3d(
                ground_truth[:, 0],  # x
                ground_truth[:, 1],  # y
                ground_truth[:, 2],  # z
                ground_truth[:, 2] + 2,  # Height data used for shading
                mode="cube",  # How to render each point {'point', 'sphere' , 'cube' }
                # colormap='spectral',  # 'bone', 'copper',
                # colormap='copper',     # Used a fixed (r,g,b) color instead of colormap
                colormap='gist_earth',  # 'bone', 'copper',
                scale_factor=voxel_size,  # scale of the points
                line_width=10,  # Scale of the line, if any
                figure=self.fig,
                scale_mode='none'
            )
        # plot sensor position
        mlab.points3d(
            sensor[0, 0],  # x
            sensor[0, 1],  # y
            sensor[0, 2],  # z
            mode="sphere",  # How to render each point {'point', 'sphere' , 'cube' }
            # colormap='spectral',  # 'bone', 'copper',
            color=(1, 0, 0),     # Used a fixed (r,g,b) color instead of colormap
            scale_factor=1,  # scale of the points
            line_width=10,  # Scale of the line, if any
            figure=self.fig,
        )
        # plot action map
        if action_map is not None:
            mlab.points3d(
                action_map[:, 0],  # x
                action_map[:, 1],  # y
                action_map[:, 2],  # z
                action_map[:, 2] + 2,  # Height data used for shading
                mode="cube",  # How to render each point {'point', 'sphere' , 'cube' }
                colormap='gist_earth',  # 'bone', 'copper',
                #color=(0, 0, 1),  # Used a fixed (r,g,b) color instead of colormap
                scale_factor=voxel_size,  # scale of the points
                figure=self.fig,
                scale_mode='none'
            )
        # plot rays
        if rays is not None:
            length = (rays.shape[0]*2)+1
            plot_rays = np.empty((length, 3))
            plot_rays[0:length:2] = sensor
            plot_rays[1:length-1:2] = rays
            mlab.plot3d(
                plot_rays[:, 0],  # x
                plot_rays[:, 1],  # y
                plot_rays[:, 2],  # z
                color=(1, 0, 0),  # Used a fixed (r,g,b) color instead of colormap
                tube_radius=0.075,
                figure=self.fig,
            )

        azimuth = math.degrees(math.atan(dir[0, 0]/dir[0, 1]))
        print(azimuth)
        print(dir)
        mlab.view(235 + azimuth, 70, distance=100,
                  focalpoint=(dir[1, 0], dir[1, 1], dir[1, 2]), figure=self.fig)
        # mlab.savefig('screen.png')
        mlab.show()

