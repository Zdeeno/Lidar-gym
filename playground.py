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
print(myDirections)
myMap = np.ones((161, 161, 10), dtype=int)
counter = 1
print('------------------- Iteration number 0 -------------------------')
print('observation:\n', obv[0], '\n', obv[1])

while not done:
    print('------------------- Iteration number ', counter , '-------------------------')
    obv, reward, done, info = env.step((myDirections, myMap))
    print('observation:\n', obv[0], '\n', obv[1], '\n', obv[2])
    print('reward:\n', reward, '\n')
    counter = counter + 1
