import numpy as np
import math
from tools import math_processing as mp


class Camera:

    def __init__(self, fov, density, max_rays):
        '''

        :param fov: tuple, angles in degrees for (width, height)
        :param density: tuple, number of points over fov (width, height)
        :param max_rays: integer, maximum number of rays per timestamp
        '''
        self.half_fov = fov/2
        self.density = density
        self.center_index = (density-1)/2
        self.max_rays = max_rays

    def calculate_directions(self, matrix, T):
        assert type(matrix) is np.ndarray and matrix.shape[0] == self.density[0] and matrix.shape[1] == self.density[1], 'wrong ray input'
        counter = 0
        vectors = np.ndarray
        for x in range(self.density[0]):
            for y in range(self.density[1]):
                if matrix[x][y]:
                    counter += 1
                    # TODO FINISH METHOD WITH TG
                    math.radians()
                    np.append(vectors, )

