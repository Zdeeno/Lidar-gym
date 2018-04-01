from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import tensorflow as tf
import tflearn
import lidar_gym


# Constants
MAP_SIZE = (320, 320, 32)
BATCH_SIZE = 4


# Convolutional network building
cnn_input = tflearn.input_data(shape=[None, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]])
cnn_input = tf.expand_dims(cnn_input, -1)
net = tflearn.conv_3d(cnn_input, 2, 2, strides=1, activation='relu', regularizer='L2')
net = tflearn.conv_3d(net, 4, 4, strides=1, activation='relu', regularizer='L2')
net = tflearn.max_pool_3d(net, 2, strides=2)
net = tflearn.conv_3d(net, 8, 4, strides=1, activation='relu', regularizer='L2')
net = tflearn.max_pool_3d(net, 2, strides=2)
net = tflearn.conv_3d(net, 16, 4, strides=1, activation='relu', regularizer='L2')
net = tflearn.conv_3d(net, 1, 8, strides=1, activation='relu', regularizer='L2')
net = tflearn.layers.conv.conv_3d_transpose(net, 1, 8, [MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]],
                                            strides=[1, 4, 4, 4, 1], regularizer='L2')
net = tf.squeeze(net, [4])
net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001)

# Create model
model = tflearn.DNN(net, tensorboard_verbose=3)
env = gym.make('lidar-v0')
done = False
random_action = env.action_space.sample()
episode = 1
env.seed(7)


def init_buffer():
    a = np.zeros((BATCH_SIZE, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))
    b = np.zeros((BATCH_SIZE, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))
    c = 0
    return a, b, c


def append_to_buffer(obs):
    global buffer_X, buffer_Y, buffer_size
    buffer_X[buffer_size] = obs['X']
    buffer_Y[buffer_size] = obs['Y']
    buffer_size = buffer_size + 1


buffer_X, buffer_Y, buffer_size = init_buffer()

while True:

    counter = 1
    obv = env.reset()
    print('------------------- Drive number', episode, '-------------------------')
    append_to_buffer(obv)

    while not done:
        obv, reward, done, info = env.step(random_action)
        append_to_buffer(obv)
        if buffer_size == BATCH_SIZE:
            model.fit(buffer_X, buffer_Y, n_epoch=1, shuffle=True, show_metric=True, batch_size=BATCH_SIZE,
                      run_id='lidar_cnn')
            buffer_X, buffer_Y, buffer_size = init_buffer()
        counter += 1

    episode += 1
    done = False
