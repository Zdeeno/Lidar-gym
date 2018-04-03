from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import tensorflow as tf
import tflearn
import lidar_gym


# Constants
MAP_SIZE = (320, 320, 32)
BATCH_SIZE = 1

# Variables
# buffer
buffer_X, buffer_Y, buffer_size = None, None, 0


def logistic_loss(y_pred, y_true):
    # own objective function
    bigger = tf.cast(tf.greater(y_true, 0.0), tf.float32)
    smaller = tf.cast(tf.greater(0.0, y_true), tf.float32)

    weights_positive = 0.5 / tf.reduce_sum(bigger)
    weights_negative = 0.5 / tf.reduce_sum(smaller)

    weights = bigger*weights_positive + smaller*weights_negative

    return tf.reduce_sum(weights * (tf.log(1 + tf.exp(-y_pred * y_true))))


def init_buffer():
    global buffer_X, buffer_Y, buffer_size
    buffer_X = np.zeros((BATCH_SIZE, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))
    buffer_Y = np.zeros((BATCH_SIZE, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))
    buffer_size = 0


def append_to_buffer(obs):
    global buffer_X, buffer_Y, buffer_size
    buffer_X[buffer_size] = obs['X']
    buffer_Y[buffer_size] = obs['Y']
    buffer_size = buffer_size + 1


def build_network():
    # Convolutional network building
    cnn_input = tflearn.input_data(shape=[None, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]])
    cnn_input = tf.expand_dims(cnn_input, -1)
    net = tflearn.conv_3d(cnn_input, 2, 2, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_3d(net, 4, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.max_pool_3d(net, 2, strides=2)
    net = tflearn.conv_3d(net, 8, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.max_pool_3d(net, 2, strides=2)
    net = tflearn.conv_3d(net, 16, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_3d(net, 32, 4, strides=1, activation='relu', regularizer='L2')
    net = tflearn.conv_3d(net, 1, 8, strides=1, activation='linear', regularizer='L2')
    net = tflearn.layers.conv.conv_3d_transpose(net, 1, 8, [MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]],
                                                strides=[1, 4, 4, 4, 1], regularizer='L2', activation='linear')
    net = tf.squeeze(net, [4])
    optimizer = tflearn.Momentum(learning_rate=0.001, lr_decay=(1/8), decay_step=10, momentum=0.99)
    net = tflearn.regression(net, optimizer=optimizer, loss=logistic_loss, learning_rate=0.001)
    return net


# Create model on GPU
model = tflearn.DNN(build_network(), tensorboard_verbose=0, tensorboard_dir="~/Projekt/tflearn_logs/")
# model.load('trained_models/my_model.tflearn')

env = gym.make('lidar-v0')
done = False
random_action = env.action_space.sample()
episode = 1
env.seed(5)

while True:
    init_buffer()
    obv = env.reset()
    print('\n------------------- Drive number', episode, '-------------------------')
    if (episode % 5) == 0 and episode != 0:
        model.save('trained_models/my_model.tflearn')

    while not done:
        append_to_buffer(obv)

        if buffer_size == BATCH_SIZE:
            model.fit(buffer_X, buffer_Y, n_epoch=1, shuffle=True, show_metric=True, batch_size=BATCH_SIZE,
                      run_id='lidar_cnn')
            init_buffer()

        env.render()
        obv, reward, done, info = env.step(obv['X'])

    episode += 1
    done = False
