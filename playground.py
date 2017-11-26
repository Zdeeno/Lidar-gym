import AIgym.lidar_gym as gym
import numpy as np


# create environment
env = gym.LidarGym()
done = False
obv = env.reset()
print(obv)
myDirections = np.eye(10)
myMap = np.zeros((80, 80, 4))
counter = 0

while not done:
    obv, reward, done, info = env.step((myDirections, myMap))
    print 'Iteration number ', counter
    print obv
    print reward
    print done
    counter = counter + 1
