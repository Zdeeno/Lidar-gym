import numpy as np

from lidar_gym.envs import lidar_gym as mygym

# create environment, consts
voxel_size = 1
max_rays = 100
lidar_range = 70
density = (9, 10)
fov = (120, 90)

# stupid input
input_map_size = np.asarray((80, 80, 4)) / voxel_size + (1, 1, 1)
myMap = np.asarray(-np.ones(input_map_size.astype(int), dtype=int))

env = mygym.LidarGym(lidar_range, voxel_size, max_rays, density, fov)

# import lidar_gym
# env = gym.make("sslidar-v0")

done = False
obv = env.reset()
myDirections = np.zeros((10, 9))
for i in range(10):
    myDirections[i][4] = True
print('Testing following ray matrix:\n', myDirections)
counter = 1
print('------------------- Iteration number 0 -------------------------')
print('Observation:\nNext position:\n', obv['T'], '\nPoints:\n', obv['points'], '\nValues:\n', obv['values'])

while not done:
    print('------------------- Iteration number ', counter , '-------------------------')
    obv, reward, done, info = env.step({"rays": myDirections, "map": myMap})
    print('Observation:\nNext position:\n', obv['T'], '\nPoints:\n', obv['points'], '\nValues\n', obv['values'])
    print('\nHited ', np.shape(np.where(obv['values'] == 1))[1], ' points!\n')
    print('reward:\n', reward, '\n')
    counter = counter + 1
