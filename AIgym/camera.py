import numpy as np
import math
from tools import math_processing as mp


class Camera:

    def __init__(self, fov, density, max_rays):
        """
        Class representing solid state lidar sensor.
        :param fov: tuple, angles in degrees for (width, height)
        :param density: tuple, number of points over fov (width, height)
        :param max_rays: integer, maximum number of rays per timestamp
        """
        self._fov = np.asarray((math.radians(float(fov[0])), math.radians(float(fov[1]))))
        self._density = np.asarray(density, np.float)
        self._center_index = (self._density - 1)/2
        self._max_rays = max_rays
        self._angle_per_bucket = self._fov/self._density

    def calculate_directions(self, matrix, T):
        """
        From given binary matrix and position (transform matrix T), calculates vectors of rays directions
        :param matrix: binary matrix
        :param T: transform matrix 4x4
        :return: set of vectors
        """
        assert type(matrix) is np.ndarray and matrix.shape[0] == self._density[0]\
            and matrix.shape[1] == self._density[1], 'wrong ray input'
        vectors = None
        init = True
        for x in range(int(self._density[0])):
            for y in range(int(self._density[1])):
                if matrix[x][y]:
                    x_dir = math.tan((x - self._center_index[0])*self._angle_per_bucket[0])
                    y_dir = 1
                    z_dir = math.tan((-y + self._center_index[1])*self._angle_per_bucket[1])
                    if init:
                        vectors = np.asmatrix((x_dir, y_dir, z_dir))
                        init = False
                    else:
                        vectors = np.concatenate((vectors, np.asmatrix((x_dir, y_dir, z_dir))), 0)
        if vectors is None:
            # TODO cover this empty return wherever this function is called
            return None
        assert mp.get_numpy_shape(vectors)[0] <= self._max_rays, 'Too many rays'
        return mp.transform_points(vectors, T)
