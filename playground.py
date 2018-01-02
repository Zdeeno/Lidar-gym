import lidar_gym.lidar_gym as mygym
import numpy as np


# create environment
env = mygym.LidarGym(100, 0.5, 100, (10, 10), (120, 90))
done = False
obv = env.reset()
#myDirections = np.eye(10)
myDirections = np.zeros((10, 10))
for i in range(10):
    myDirections[i][4] = True
print('Testing following ray matrix:\n', myDirections)
myMap = np.ones((161, 161, 10), dtype=int)
counter = 1
print('------------------- Iteration number 0 -------------------------')
print('Observation:\nNext position:\n', obv[0], '\nPoints:\n', obv[1], '\nValues\n', obv[2])

while not done:
    print('------------------- Iteration number ', counter , '-------------------------')
    obv, reward, done, info = env.step((myDirections, myMap))
    print('Observation:\nNext position:\n', obv[0], '\nPoints:\n', obv[1], '\nValues\n', obv[2])
    print('Hited ', np.shape(np.where(obv[2] == 1))[1], ' points!\n')
    print('reward:\n', reward, '\n')
    counter = counter + 1
