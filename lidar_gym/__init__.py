from gym.envs.registration import register

register(
    id='lidar-v1',
    entry_point='lidar_gym.envs:Lidarv1',
    timestep_limit=1000000,
)

register(
    id='lidar-v0',
    entry_point='lidar_gym.envs:Lidarv0',
    timestep_limit=1000000,
)
