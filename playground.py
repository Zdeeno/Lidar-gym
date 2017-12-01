import AIgym.lidar_gym as gym
import numpy as np


# create environment
env = gym.LidarGym()
done = False
obv = env.reset()
myDirections = np.eye(10)
myMap = np.zeros((80, 80, 4))
counter = 1

print '------------------- Iteration number 0 -------------------------'
print 'observation:\n', obv[0], '\n', obv[1]

while not done:
    print '------------------- Iteration number ', counter , '-------------------------'
    obv, reward, done, info = env.step((myDirections, myMap))
    print 'observation:\n', obv[0], '\n', obv[1]
    print 'reward:\n', reward, '\n'
    counter = counter + 1
