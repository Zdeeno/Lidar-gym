from gym.envs.registration import register

register(
    id='sslidar-v0',
    entry_point='lidar_gym.envs:Lidarv0',
    timestep_limit=10000,
)
