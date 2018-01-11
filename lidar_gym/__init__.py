from gym.envs.registration import register

register(
    id='sslidar-v1',
    entry_point='lidar_gym.envs:Lidarv1',
    timestep_limit=1000000,
)
