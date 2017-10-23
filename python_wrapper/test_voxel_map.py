from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
import voxel_map as vm


class TestVoxelMap(unittest.TestCase):
    def setUp(self):
        print("SET UP CLASSES")
        self.map = vm.VoxelMap()
        self.map.voxel_size = 1.0
        self.map.free_update = 1.0
        self.map.hit_update = 1.0
        self.map.occupancy_threshold = 0.0
        self.assertIsNotNone(self.map)

        n = 1
        x0 = 5 * np.ones((3, n))
        l0 = np.ones((n,), dtype=np.float64)
        v0 = 2 * np.random.rand(n) - 1
        print("SETTING VALUES:")
        print(x0, '\n', l0, '\n', v0)
        self.map.set_voxels(x0, l0, v0)
        
        print("CLASSES CREATED")

    def test_setget_voxels(self):
        print("TESTING SETTERS AND GETTERS")
        n = 5
        x0 = 10 * np.random.rand(3, n)
        l0 = np.zeros((n,), dtype=np.float64)
        v0 = 2 * np.random.rand(n) - 1
        print("SETTING VALUES:")
        print(x0, '\n', l0, '\n', v0)
        self.map.set_voxels(x0, l0, v0)
        
        print("GETTED CHOSEN VALUES:")
        v1 = self.map.get_voxels(x0, l0)
        print(v1)
        for el0, el1 in zip(v0.tolist(), v1.tolist()):
            self.assertAlmostEqual(el0, el1)
        print("GETTED ALL VALUES:")
        [x2, l2, v2] = self.map.get_voxels()
        print(x2, '\n', l2, '\n', v2)

        v3 = self.map.get_voxels(x2, l2)
        for el0, el1 in zip(v2.tolist(), v3.tolist()):
            self.assertEqual(el0, el1)


    def test_update_lines(self):
        n = 2
        x0 = np.zeros((3, n), dtype=np.float64)
        x1 = 10 * np.random.rand(3, n)

        self.map.update_lines(x0, x1)

        [x2, l2, v2] = self.map.get_voxels()

    def test_pass(self):
        pass

    def test_trace_lines(self):
        self.map.clear()
        self.map.voxel_size = 0.5
        self.map.free_update = -1.0
        self.map.hit_update = 1.0
        n = 1
        x0 = np.zeros((3, n), dtype=np.float64)
        x1 = np.ones((3, n))

        print('Tracing rays from:\n{},\nto:\n{}...'.format(x0, x1))
        min_val = -100.0
        max_val = 1
        [h, v] = self.map.trace_lines(x0, x1, min_val, max_val, 0)
        print('Lines traced to:\n{}\nwith values\n{}.'.format(h, v))

    def test_trace_rays(self):
        self.map.clear()
        self.map.voxel_size = 0.5
        self.map.free_update = -1.0
        self.map.hit_update = 1.0
        n = 2
        x0 = np.zeros((3, n), dtype=np.float64)
        x1 = np.ones((3, n))
        print('Tracing rays from:\n{},\nto:\n{}...'.format(x0, x1))
        max_range = 100
        min_val = -100.0
        max_val = 1
        [h, v] = self.map.trace_rays(x0, x1, max_range, min_val, max_val, 0)
        print('Lines traced to:\n{}\nwith values\n{}.'.format(h, v))


if __name__ == '__main__':
    unittest.main()
