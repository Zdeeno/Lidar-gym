import numpy as np

from lidar_gym.envs import lidar_gym as mygym

# create environment
# import lidar_gym
# env = gym.make("sslidar-v0")
env = mygym.LidarGym(100, 0.5, 100, (9, 10), (120, 90))
done = False
obv = env.reset()
#myDirections = np.eye(10)
myDirections = np.zeros((10, 9))
for i in range(10):
    myDirections[i][4] = True
print('Testing following ray matrix:\n', myDirections)
myMap = np.asarray(np.ones((161, 161, 9), dtype=int))
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