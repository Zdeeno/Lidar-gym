from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import tensorflow as tf
from os.path import expanduser
import os
from tensorflow.contrib.keras.api.keras.layers import Conv3D, MaxPool3D, Input, Lambda, Conv3DTranspose
from tensorflow.contrib.keras.api.keras.backend import squeeze, expand_dims
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
from tensorflow.contrib.keras.api.keras.optimizers import Adam


def logistic_loss(y_true, y_pred):
    # own objective function
    bigger = tf.cast(tf.greater(y_true, 0.0), tf.float32)
    smaller = tf.cast(tf.greater(0.0, y_true), tf.float32)

    weights_positive = 0.5 / tf.reduce_sum(bigger)
    weights_negative = 0.5 / tf.reduce_sum(smaller)

    weights = bigger * weights_positive + smaller * weights_negative

    # Here often occurs numeric instability -> nan or inf
    # return tf.reduce_sum(weights * (tf.log(1 + tf.exp(-y_pred * y_true))))
    a = -y_pred * y_true
    b = tf.maximum(0.0, a)
    t = b + tf.log(tf.exp(-b) + tf.exp(a - b))
    return tf.reduce_sum(weights * t)


class Supervised:
    def __init__(self):
        # Constants
        self._map_shape = (320, 320, 32)
        self._batch_size = 4
        self._epochs_per_batch = 1
        self._learning_rate = 0.001
        self._l2reg = 0.0001

        # buffer
        self._buffer_X, self._buffer_Y, self._buffer_size = self.init_buffer()

        # model
        self._model = self.create_model()
        home = expanduser("~")
        logdir = os.path.join(home, 'supervised_logs/')
        self._tfboard = TensorBoard(log_dir=logdir, batch_size=self._batch_size, write_graph=False)

    def init_buffer(self):
        buffer_X = np.zeros((self._batch_size, self._map_shape[0], self._map_shape[1], self._map_shape[2]))
        buffer_Y = np.zeros((self._batch_size, self._map_shape[0], self._map_shape[1], self._map_shape[2]))
        buffer_size = 0
        return buffer_X, buffer_Y, buffer_size

    def append_to_buffer(self, obs):
        self._buffer_X[self._buffer_size] = obs['X']
        self._buffer_Y[self._buffer_size] = obs['Y']
        self._buffer_size += 1

    def create_model(self):
        # 3D convolutional network building
        inputs = Input(shape=(self._map_shape[0], self._map_shape[1], self._map_shape[2]))
        reshape = Lambda(lambda x: expand_dims(x, -1))(inputs)

        c1 = Conv3D(2, 4, padding='same', kernel_regularizer=l2(self._l2reg), activation='relu')(reshape)
        c2 = Conv3D(4, 4, padding='same', kernel_regularizer=l2(self._l2reg), activation='relu')(c1)
        p1 = MaxPool3D(pool_size=2)(c2)
        c3 = Conv3D(8, 4, padding='same', kernel_regularizer=l2(self._l2reg), activation='relu')(p1)
        p2 = MaxPool3D(pool_size=2)(c3)
        c4 = Conv3D(16, 4, padding='same', kernel_regularizer=l2(self._l2reg), activation='relu')(p2)
        c5 = Conv3D(32, 4, padding='same', kernel_regularizer=l2(self._l2reg), activation='relu')(c4)
        c6 = Conv3D(1, 8, padding='same', kernel_regularizer=l2(self._l2reg), activation='linear')(c5)
        out = Conv3DTranspose(1, 8, strides=[4, 4, 4], padding='same', activation='linear',
                              kernel_regularizer=l2(self._l2reg))(c6)
        outputs = Lambda(lambda x: squeeze(x, 4))(out)
        opt = Adam(lr=self._learning_rate)
        model = Model(inputs, outputs)
        model.compile(optimizer=opt, loss=logistic_loss)
        return model

    def train_model(self):
        if self._buffer_size == self._batch_size:
            self._model.fit(x=self._buffer_X, y=self._buffer_Y, epochs=self._epochs_per_batch, shuffle=True,
                            batch_size=self._batch_size, callbacks=[self._tfboard])
            # clean buffer
            self._buffer_X, self._buffer_Y, self._buffer_size = self.init_buffer()

    def load_weights(self, weights_dir):
        self._model.load_weights(filepath=weights_dir)

    def save_weights(self, save_dir):
        self._model.save(filepath=save_dir)

    def predict(self, input_X):
        input_X = np.expand_dims(input_X, axis=0)
        return self._model.predict(input_X)[0]


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


def evaluate(supervised):
    evalenv = gym.make('lidareval-v0')
    done = False
    reward_overall = 0
    _ = evalenv.reset()
    map = np.zeros((320, 320, 32))

    while not done:
        a = evalenv.action_space.sample()
        obv, reward, done, _ = evalenv.step({'map': map, 'rays': a['rays']})
        reward_overall += reward
        map = supervised.predict(obv['X'])
    print('EVALUATION DONE')
    return reward_overall


if __name__ == "__main__":

    LOAD = False
    # Create model on GPU
    agent = Supervised()

    home = expanduser("~")
    savedir = os.path.join(home, 'trained_models/my_keras_model_supervised.h5')

    if LOAD:
        loaddir = expanduser("~")
        loaddir = os.path.join(loaddir, 'Projekt/lidar-gym/trained_models/my_keras_model_supervised.h5')
        agent.load_weights(loaddir)

    env = gym.make('lidar-v2')
    episode = 1
    max_reward = -float('inf')
    env.seed(5)

    while True:
        done = False
        obv = env.reset()
        print('\n------------------- Drive number', episode, '-------------------------')

        while not done:
            print(obv)
            agent.append_to_buffer(obv)
            agent.train_model()
            obv, reward, done, info = env.step(obv['X'])

        episode += 1
        # Evaluate and save
        if episode % 5 == 0:
            rew = evaluate(agent)
            if rew > max_reward:
                print('new best agent - saving with reward:' + rew)
                max_reward = rew
                agent.save_weights(savedir)
