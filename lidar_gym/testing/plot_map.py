import mayavi.mlab

# Plot your map using Mayavi


def plot_action(ground_truth, action_map, rays, voxel_size, sensor):
    # TODO: Change mlab.show to mlab.animate!

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    mayavi.mlab.points3d(
        ground_truth[:, 0],  # x
        ground_truth[:, 1],  # y
        ground_truth[:, 2],  # z
        # ground_truth[:, 2],  # Height data used for shading
        mode="cube",  # How to render each point {'point', 'sphere' , 'cube' }
        # colormap='spectral',  # 'bone', 'copper',
        color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
        scale_factor=voxel_size,  # scale of the points
        line_width=10,  # Scale of the line, if any
        figure=fig,
    )
    mayavi.mlab.points3d(
        sensor[:, 0],  # x
        sensor[:, 1],  # y
        sensor[:, 2],  # z
        # ground_truth[:, 2],  # Height data used for shading
        mode="sphere",  # How to render each point {'point', 'sphere' , 'cube' }
        # colormap='spectral',  # 'bone', 'copper',
        color=(1, 0, 0),     # Used a fixed (r,g,b) color instead of colormap
        scale_factor=1,  # scale of the points
        line_width=10,  # Scale of the line, if any
        figure=fig,
    )

    mayavi.mlab.show(stop=True)
