from lidar_gym.visualiser import printer
import importlib
mayavi_spec = importlib.util.find_spec("mayavi")
found = mayavi_spec is not None

if found:
    from lidar_gym.visualiser import plot3d
