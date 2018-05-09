from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import tensorflow as tf
from os.path import expanduser
import os

from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
from lidar_gym.agent.models import create_toy_supervised_model


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
        # self._map_shape = (320, 320, 32)
        # self._map_shape = (160, 160, 16)
        self._map_shape = (80, 80, 8)
        self._batch_size = 4
        self._epochs_per_batch = 1
        self._learning_rate = 0.001
        self._l2reg = 0.001

        # buffer
        self._buffer_X, self._buffer_Y, self._buffer_size = self.init_buffer()

        # model
        self._model = create_toy_supervised_model(self._learning_rate, self._map_shape)
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

    def train_model(self):
        if self._buffer_size == self._batch_size:
            self._model.fit(x=self._buffer_X, y=self._buffer_Y, epochs=self._epochs_per_batch, shuffle=True,
                            batch_size=self._batch_size, callbacks=[self._tfboard], verbose=0)
            # clean buffer
            self._buffer_X, self._buffer_Y, self._buffer_size = self.init_buffer()

    def load_weights(self, weights_dir):
        self._model.load_weights(filepath=weights_dir)

    def save(self, save_dir):
        self._model.save(filepath=save_dir)

    def predict(self, input_X):
        input_X = np.expand_dims(input_X, axis=0)
        return self._model.predict(input_X)[0]


def evaluate(supervised):
    # evalenv = gym.make('lidareval-v0')
    # evalenv = gym.make('lidarsmalleval-v0')
    evalenv = gym.make('lidartoyeval-v0')
    done = False
    reward_overall = 0
    _ = evalenv.reset()
    # map = np.zeros((320, 320, 32))
    map = np.zeros(shape=supervised._map_shape)
    evalenv.seed(1)
    print('Evaluation started!')
    while not done:
        a = evalenv.action_space.sample()
        obv, reward, done, _ = evalenv.step({'map': map, 'rays': a['rays']})
        reward_overall += reward
        map = supervised.predict(obv['X'])
    print('Evaluation done with reward - ' + str(reward_overall))
    return reward_overall


if __name__ == "__main__":

    LOAD = False
    # Create model on GPU
    agent = Supervised()
    home = expanduser("~")
    savedir = os.path.join(home, 'trained_models/')

    if LOAD:
        loaddir = expanduser("~")
        loaddir = os.path.join(loaddir, 'trained_models/supervised_model_-205.62373544534486.h5')
        agent.load_weights(loaddir)

    env = gym.make('lidartoy-v2')
    episode = 0
    max_reward = -276.5
    env.seed(1)

    while True:
        done = False
        obv = env.reset()
        print('\n------------------- Drive number', episode, '-------------------------')

        while not done:
            agent.append_to_buffer(obv)
            agent.train_model()
            obv, reward, done, info = env.step(obv['X'])
            print('.', end='', flush=True)

        # Evaluate and save
        if episode % 10 == 0:
            rew = evaluate(agent)
            if rew > max_reward:
                print('new best agent - saving with reward:' + str(rew))
                max_reward = rew
                # agent.save_weights(savedir + 'supervised_model_' + str(max_reward) + '.h5')
                agent.save(savedir + 'supervised_toy_model_' + str(max_reward) + '.h5')
        episode += 1
