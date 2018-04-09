from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import tensorflow as tf
from os.path import expanduser
import os
import tensorflow.contrib.keras as keras


# Constants
MAP_SIZE = (320, 320, 32)
BATCH_SIZE = 4

# Variables
# buffer
buffer_X, buffer_Y, buffer_size = None, None, 0


def logistic_loss(y_true, y_pred):
    # own objective function
    bigger = tf.cast(tf.greater(y_true, 0.0), tf.float32)
    smaller = tf.cast(tf.greater(0.0, y_true), tf.float32)

    weights_positive = 0.5 / tf.reduce_sum(bigger)
    weights_negative = 0.5 / tf.reduce_sum(smaller)

    weights = bigger*weights_positive + smaller*weights_negative

    # Here often occurs numeric instability -> nan or inf
    # return tf.reduce_sum(weights * (tf.log(1 + tf.exp(-y_pred * y_true))))
    a = -y_pred*y_true
    b = tf.maximum(0.0, a)
    t = b + tf.log(tf.exp(-b) + tf.exp(a-b))
    return tf.reduce_sum(weights*t)


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
    # 3D convolutional network building
    inputs = keras.layers.Input(shape=(MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))
    reshape = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, -1))(inputs)

    c1 = keras.layers.Conv3D(2, 4, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                             activation='relu')(reshape)
    c2 = keras.layers.Conv3D(4, 4, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                             activation='relu')(c1)
    p1 = keras.layers.MaxPooling3D(pool_size=2)(c2)
    c3 = keras.layers.Conv3D(8, 4, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                             activation='relu')(p1)
    p2 = keras.layers.MaxPooling3D(pool_size=2)(c3)
    c4 = keras.layers.Conv3D(16, 4, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                             activation='relu')(p2)
    c5 = keras.layers.Conv3D(32, 4, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                             activation='relu')(c4)
    c6 = keras.layers.Conv3D(1, 8, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                             activation='linear')(c5)
    out = keras.layers.Conv3DTranspose(1, 8, strides=[4, 4, 4], padding='same', activation='linear',
                                           kernel_regularizer=keras.regularizers.l2(0.01))(c6)
    outputs = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 4))(out)

    return keras.models.Model(inputs, outputs)

'''
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
opt = keras.optimizers.Adam()
model = build_network()
model.compile(optimizer=opt, loss=logistic_loss)

mydir = expanduser("~")
savedir = os.path.join(mydir, 'trained_models/my_keras_model.h5')
mydir = os.path.join(mydir, 'training_logs/')
tfboard = keras.callbacks.TensorBoard(log_dir=mydir, batch_size=BATCH_SIZE)

env = gym.make('lidar-v0')
done = False
random_action = env.action_space.sample()
episode = 1
env.seed(5)

while True:
    init_buffer()
    obv = env.reset()
    print('\n------------------- Drive number', episode, '-------------------------')
    if (episode % 5) == 0:
        model.save(savedir)

    while not done:
        append_to_buffer(obv)

        if buffer_size == BATCH_SIZE:
            model.fit(x=buffer_X, y=buffer_Y, epochs=2, shuffle=True, batch_size=BATCH_SIZE, callbacks=[tfboard])
            init_buffer()

        obv, reward, done, info = env.step(obv['X'])

    episode += 1
    done = False
