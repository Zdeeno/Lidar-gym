from gym.envs.registration import register
from lidar_gym import tools, agent, envs

register(
    id='lidareval-v0',
    entry_point='lidar_gym.envs:LidarEval',
    timestep_limit=1000000,
)

register(
    id='lidar-v2',
    entry_point='lidar_gym.envs:Lidarv2',
    timestep_limit=1000000,
)

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

register(
    id='lidarsmall-v0',
    entry_point='lidar_gym.envs:LidarSmallv0',
    timestep_limit=1000000,
)

register(
    id='lidarsmall-v2',
    entry_point='lidar_gym.envs:LidarSmallv2',
    timestep_limit=1000000,
)

register(
    id='lidarsmalleval-v0',
    entry_point='lidar_gym.envs:LidarSmallEval',
    timestep_limit=1000000,
)

register(
    id='lidartoy-v0',
    entry_point='lidar_gym.envs:LidarToyv0',
    timestep_limit=1000000,
)

register(
    id='lidartoy-v2',
    entry_point='lidar_gym.envs:LidarToyv2',
    timestep_limit=1000000,
)

register(
    id='lidartoyeval-v0',
    entry_point='lidar_gym.envs:LidarToyEval',
    timestep_limit=1000000,
)
