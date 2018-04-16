import gym
import numpy as np
import random
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Input, Lambda, Conv3D, MaxPool3D, Conv2D,\
                                                      MaxPool2D, Reshape
from tensorflow.contrib.keras.api.keras.backend import squeeze, expand_dims, reshape
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard

from os.path import expanduser
import os

import tensorflow as tf
from collections import deque
import lidar_gym
from lidar_gym.agent.supervised_agent import Supervised


class SARSA:

    def __init__(self, env):
        # setup environment
        self._env = env
        self._batch_size = 16
        self._map_shape = (320, 320, 32)
        self._max_rays = 200
        self._rays_shape = (160, 120)

        # setup consts
        self._gamma = 0.85
        self._epsilon = 1.0
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.995
        self._learning_rate = 0.005
        self._tau = .125

        # setup buffer
        self._buffer = deque(maxlen=self._batch_size)

        # double network
        self._model = self.create_model()
        self._target_model = self.create_model()

        # logger
        home = expanduser("~")
        logdir = os.path.join(home, 'DQN_logs/')
        self._tfboard = TensorBoard(log_dir=logdir, batch_size=self._batch_size, write_graph=False)

    def create_model(self):
        state_input = Input(shape=self._map_shape)
        r1 = Lambda(lambda x: expand_dims(x, -1))(state_input)

        c1 = Conv3D(2, 4, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(r1)
        p1 = MaxPool3D(pool_size=2)(c1)
        c2 = Conv3D(4, 4, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(p1)
        c3 = Conv3D(1, 8, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(c2)
        s1 = Lambda(lambda x: squeeze(x, 4))(c3)
        c4 = Conv2D(9, 3, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(s1)
        r2 = Reshape((480, 480, 1))(c4)
        p2 = MaxPool2D(pool_size=(3, 4))(r2)
        c5 = Conv2D(2, 4, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(p2)
        c6 = Conv2D(1, 4, padding='same', kernel_regularizer=l2(0.0001), activation='linear')(c5)
        output = Lambda(lambda x: squeeze(x, 3))(c6)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    def predict(self, state):
        return self._model.predict(state)[0]

    def act(self, state):
        # Exploration vs exploitation
        self._epsilon *= self._epsilon_decay
        self._epsilon = max(self._epsilon_min, self._epsilon)
        if np.random.random() < self._epsilon:
            return self._env.action_space.sample()['rays']
        state = np.expand_dims(state, axis=0)
        rays = self._model.predict(state)[0]
        # Q values to top n bools
        ret = np.zeros(shape=self._rays_shape, dtype=bool)
        ret[self._largest_indices(rays, self._max_rays)] = True
        return ret

    def replay(self):
        if len(self._buffer) < self._batch_size:
            return

        samples = random.sample(self._buffer, self._batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = np.expand_dims(state, axis=0)
            target = self._target_model.predict(state)
            if done:
                target[action == 1] = reward
            else:
                new_state = np.expand_dims(new_state, axis=0)
                Q_future = self._target_model.predict(new_state)
                target[0, action] = reward + Q_future * self._gamma
            self._model.fit(state, target, epochs=1, verbose=1)

    def target_train(self):
        weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self._tau + target_weights[i] * (1 - self._tau)

    def save_model(self, fn):
        self._model.save(fn)

    def append_to_buffer(self, state, action, reward, new_state, done):
        self._buffer.append([state, action, reward, new_state, done])

    def _n_best_Q(self, arr, n):
        """
        Returns the n largest indices from a numpy array.
        """
        indices = self._largest_indices(arr, n)
        return np.sum(arr[indices])/self._max_rays

    def _largest_indices(self, arr, n):
        """
        Returns the n largest indices from a numpy array.
        """
        flat = arr.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, arr.shape)

    def clear_buffer(self):
        self._buffer.clear()


def evaluate(supervised, dqn):
    evalenv = gym.make('lidareval-v0')
    done = False
    reward_overall = 0
    _ = evalenv.reset()
    map = np.zeros((320, 320, 32))
    while not done:
        rays = dqn.predict(map)
        obv, reward, done, _ = evalenv.step({'map': map, 'rays': rays})
        reward_overall += reward
        map = supervised.predict(obv)
    return reward_overall


if __name__ == "__main__":
    env = gym.make('lidar-v0')

    # updateTargetNetwork = 1000
    dqn_agent = SARSA(env=env)
    supervised = Supervised()

    loaddir = expanduser("~")
    loaddir = os.path.join(loaddir, 'Projekt/lidar-gym/trained_models/my_keras_model.h5')
    supervised.load_weights(loaddir)
    savedir = expanduser("~")
    savedir = os.path.join(loaddir, 'Projekt/lidar-gym/trained_models/my_keras_dqn_model.h5')

    shape = dqn_agent._map_shape

    episode = 0
    max_reward = -float('inf')

    while True:
        done = False
        curr_state = env.reset()
        curr_state = np.zeros((shape[0], shape[1], shape[2]))
        epoch = 1

        # training
        while not done:
            action = dqn_agent.act(curr_state)
            new_state, reward, done, _ = env.step({'rays': action, 'map': curr_state})

            new_state = supervised.predict(new_state['X'])
            dqn_agent.append_to_buffer(curr_state, action, reward, new_state, done)

            if epoch % dqn_agent._batch_size == 0:
                dqn_agent.replay()  # internally iterates default (prediction) model
                dqn_agent.target_train()  # iterates target model

            curr_state = new_state
            epoch += 1

        # evaluation and saving
        print('end of episode')
        episode += 1
        if episode % 5 == 0:
            rew = evaluate(supervised, dqn_agent)
            if rew > max_reward:
                print('new best agent - saving with reward:' + rew)
                max_reward = rew
                dqn_agent.save_model(savedir)

        dqn_agent.clear_buffer()
