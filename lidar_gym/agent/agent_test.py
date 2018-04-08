from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import tensorflow as tf
import tflearn
import lidar_gym
from os.path import expanduser
import os

# Constants
MAP_SIZE = (320, 320, 32)
BATCH_SIZE = 4


def build_network():
    # 3D convolutional network building
    cnn_input = tflearn.input_data(shape=[None, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]])
    cnn_input = tf.expand_dims(cnn_input, -1)
    net = tflearn.conv_3d(cnn_input, 2, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_3d(net, 4, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.max_pool_3d(net, 2, strides=2)
    net = tflearn.conv_3d(net, 8, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.max_pool_3d(net, 2, strides=2)
    net = tflearn.conv_3d(net, 16, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_3d(net, 32, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_3d(net, 1, 8, strides=1, activation='linear', regularizer='L2')
    net = tflearn.layers.conv.conv_3d_transpose(net, 1, 8, [MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]],
                                                strides=[1, 4, 4, 4, 1], activation='linear', regularizer='L2')
    net = tf.squeeze(net, [4])
    # optimizer = tflearn.Momentum(learning_rate=0.01, lr_decay=(1/8), decay_step=10, momentum=0.99)
    optimizer = tflearn.Adam(learning_rate=0.001)
    net = tflearn.regression(net, optimizer=optimizer, loss=logistic_loss)
    return net


'''
def build_network():
    # 2D convolutional network building
    cnn_input = tflearn.input_data(shape=[None, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]])
    net = tflearn.conv_2d(cnn_input, 64, 5, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_2d(net, 128, 5, strides=1, activation='relu', regularizer='L2')
    net = tflearn.max_pool_2d(net, 2, strides=2)
    net = tflearn.conv_2d(net, 256, 5, strides=1, activation='relu', regularizer='L2')
    net = tflearn.max_pool_2d(net, 2, strides=2)
    net = tflearn.conv_2d(net, 512, 5, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_2d(net, 1024, 5, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_2d(net, 32, 5, strides=1, activation='linear', regularizer='L2')
    net = tflearn.layers.conv.conv_2d_transpose(net, 32, 8, [MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]],
                                                strides=[1, 4, 4, 1], activation='linear', regularizer='L2')
    # optimizer = tflearn.Momentum(learning_rate=0.01, lr_decay=(1/8), decay_step=10, momentum=0.99)
    optimizer = tflearn.Adam(learning_rate=0.001)
    net = tflearn.regression(net, optimizer=optimizer, loss=logistic_loss)
    return net
'''

# Create model on GPU
mydir = expanduser("~")
mydir = os.path.join(mydir, 'tflearn_logs/')
model = tflearn.DNN(build_network(), tensorboard_verbose=0, tensorboard_dir=mydir)
model.load('trained_models/my_model.tflearn')

env = gym.make('lidar-v0')
done = False
random_action = env.action_space.sample()
episode = 1
env.seed(5)

while True:
    obv = env.reset()
    print('\n------------------- Drive number', episode, '-------------------------')

    while not done:

        action = model.predict(obv['X'])
        obv, reward, done, info = env.step(action)
        env.render()

    episode += 1
    done = False
