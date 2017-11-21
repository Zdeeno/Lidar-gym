import numpy as np
import math
from tools import math_processing as mp


class Camera:

    def __init__(self, fov, density, max_rays):
        '''
        Class representing solid state lidar sensor.
        :param fov: tuple, angles in degrees for (width, height)
        :param density: tuple, number of points over fov (width, height)
        :param max_rays: integer, maximum number of rays per timestamp
        '''
        self.fov = math.radians(fov)
        self.density = density
        self.center_index = (density-1)/2
        self.max_rays = max_rays
        self.angle_per_bucket = self.fov/density

    def calculate_directions(self, matrix, T):
        '''
        From given binary matrix and position (transform matrix T), calculates vectors of rays directions
        :param matrix: binary matrix
        :param T: transform matrix 4x4
        :return: set of vectors
        '''

        assert type(matrix) is np.ndarray and matrix.shape[0] == self.density[0]\
               and matrix.shape[1] == self.density[1], 'wrong ray input'
        counter = 0
        vectors = np.empty(0)
        for x in range(self.density[0]):
            for y in range(self.density[1]):
                if matrix[x][y]:
                    counter += 1
                    x_dir = math.tan((x - self.center_index)*self.angle_per_bucket)
                    y_dir = 1
                    z_dir = math.tan((-y + self.center_index)*self.angle_per_bucket)
                    vectors = np.r_['0,2', vectors, (x_dir, y_dir, z_dir)]
        assert counter < self.max_rays, 'Too many rays'
        return mp.transform_points(vectors, T)
