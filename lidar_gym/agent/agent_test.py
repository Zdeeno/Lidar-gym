from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import tensorflow as tf
import tflearn
import lidar_gym
import tensorflow.contrib.keras as keras

# Constants
MAP_SIZE = (320, 320, 32)
BATCH_SIZE = 1

model = tflearn.DNN.load(model_file='trained_models/my_model.tflearn')

env = gym.make('lidar-v0')
done = False
random_action = env.action_space.sample()
episode = 1
my_input = np.empty((BATCH_SIZE, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))
env.seed(1)

while True:
    obv = env.reset()
    print('\n------------------- Drive number', episode, '-------------------------')
    epoch = 0
    while not done:

        my_input[0] = obv['X']
        action = model.predict(my_input)
        obv, reward, done, info = env.step(action[0])
        print('reward: ' + str(reward))
        if epoch % 5 == 0:
            env.render()
        epoch = epoch + 1

    episode += 1
    done = False