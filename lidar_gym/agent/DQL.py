import gym
import numpy as np
import random
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Input, Lambda, Conv3D, MaxPool3D, Conv2D,\
                                                      MaxPool2D, Reshape, Add, Multiply
from tensorflow.contrib.keras.api.keras.backend import squeeze, expand_dims, reshape
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
import tensorflow.contrib.keras.api.keras.backend as K
from os.path import expanduser
import os

import tensorflow as tf
from collections import deque
import lidar_gym
from lidar_gym.agent.supervised_agent import Supervised
from lidar_gym.tools.sum_tree import Memory


class DQN:

    def __init__(self, env):
        # setup environment
        self._env = env
        # plus one in TD compute
        self._batch_size = 7
        # self._map_shape = (320, 320, 32)
        # self._max_rays = 200
        # self._rays_shape = (160, 120)
        self._map_shape = (160, 160, 16)
        self._max_rays = 100
        self._rays_shape = (120, 90)

        # setup consts
        self._gamma = 0.9
        self._epsilon = 1.0
        self._epsilon_min = 0.2
        self._epsilon_decay = 0.999
        self._learning_rate = 0.001
        self._tau = .1

        # setup buffer
        # self._buffer_size = 200
        self._buffer_size = 512
        self._buffer = Memory(self._buffer_size)

        # double network
        self._model = self.create_model()
        self._target_model = self.create_model()

        # logger
        home = expanduser("~")
        logdir = os.path.join(home, 'DQN_logs/')
        self._tfboard = TensorBoard(log_dir=logdir, batch_size=self._batch_size, write_graph=False)

    def create_model(self):
        # reconstructed input
        reconstructed_input = Input(shape=self._map_shape)
        r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
        c11 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r11)
        p11 = MaxPool3D(pool_size=2)(c11)
        c21 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p11)

        # sparse input
        sparse_input = Input(shape=self._map_shape)
        r12 = Lambda(lambda x: expand_dims(x, -1))(sparse_input)
        c12 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r12)
        p12 = MaxPool3D(pool_size=2)(c12)
        c22 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p12)

        '''
        # merge LARGE inputs
        c2 = Add()([c21, c22])
        c3 = Conv3D(1, 4, padding='same', activation='relu')(c2)
        s1 = Lambda(lambda x: squeeze(x, 4))(c3)
        c4 = Conv2D(9, 4, padding='same', activation='relu')(s1)
        r2 = Reshape((480, 480, 1))(c4)
        p2 = MaxPool2D(pool_size=(3, 4))(r2)
        c5 = Conv2D(2, 4, padding='same', activation='relu')(p2)
        c6 = Conv2D(1, 4, padding='same', activation='linear')(c5)
        output = Lambda(lambda x: squeeze(x, 3))(c6)
        '''

        # merge SMALL inputs
        a1 = Add()([c21, c22])
        c1 = Conv3D(1, 4, padding='same', activation='relu', kernel_regularizer='l2')(a1)
        s1 = Lambda(lambda x: squeeze(x, 4))(c1)
        c2 = Conv2D(8, 4, padding='same', activation='relu', kernel_regularizer='l2')(s1)
        p1 = MaxPool2D(pool_size=2)(c2)
        c3 = Conv2D(81, 4, padding='same', activation='relu', kernel_regularizer='l2')(p1)
        r2 = Reshape((360, 360, 1))(c3)
        p2 = MaxPool2D(pool_size=(3, 4))(r2)
        c5 = Conv2D(4, 4, padding='same', activation='relu')(p2)
        c6 = Conv2D(8, 4, padding='same', activation='linear')(c5)
        c7 = Conv2D(1, 4, padding='same', activation='linear')(c6)
        output = Lambda(lambda x: K.squeeze(x, 3))(c7)

        model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    def predict(self, state):
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        rays = self._model.predict(state)[0]
        ret = np.zeros(shape=self._rays_shape, dtype=bool)
        ret[self._largest_indices(rays, self._max_rays)] = True
        return ret

    def act(self, state):
        # Exploration vs exploitation
        self._epsilon = max(self._epsilon_min, self._epsilon)
        if np.random.random() < self._epsilon:
            return self._env.action_space.sample()['rays']
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        rays = self._model.predict(state)[0]
        # Q values to top n bools
        ret = np.zeros(shape=self._rays_shape, dtype=bool)
        ret[self._largest_indices(rays, self._max_rays)] = True
        return ret

    def replay(self):
        self._epsilon *= self._epsilon_decay

        if self._buffer.length < self._batch_size:
            return

        samples = self._buffer.sample(self._batch_size)
        for sample in samples:
            idx, data = sample
            state, action, reward, new_state, done = data
            state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
            Q = self._target_model.predict(state)
            target = np.copy(Q)
            if done:
                target[0, action] = reward
            else:
                new_state = [np.expand_dims(new_state[0], axis=0), np.expand_dims(new_state[1], axis=0)]
                # Q learning:
                # q_future = self._n_best_Q(self._target_model.predict(new_state), self._max_rays)
                # target[0, action] = reward + q_future * self._gamma
                # double Q learning
                online_max = self._largest_indices(self._model.predict(new_state), self._max_rays)
                q_future = np.sum(self._target_model.predict(new_state)[online_max])/self._max_rays
                target[0, action] = reward + q_future * self._gamma

            self._buffer.update(idx, np.abs(np.sum(Q - target)))
            self._model.fit(state, target, epochs=1, verbose=0)

    def TD_size(self, sample):
        # we already make a computation to fit nn here so why not to fit
        state, action, reward, new_state, done = sample
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        Q = self._target_model.predict(state)
        target = np.copy(Q)
        if done:
            target[0, action] = reward
        else:
            new_state = [np.expand_dims(new_state[0], axis=0), np.expand_dims(new_state[1], axis=0)]
            online_max = self._largest_indices(self._model.predict(new_state), self._max_rays)
            q_future = np.sum(self._target_model.predict(new_state)[online_max]) / self._max_rays
            target[0, action] = reward + q_future * self._gamma
        self._model.fit(state, target, epochs=1, verbose=0)
        ret = np.abs(np.sum(Q - target))
        return ret

    def target_train(self):
        weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self._tau + target_weights[i] * (1 - self._tau)

    def save_model(self, f):
        self._model.save(f)

    def load_model_weights(self, f):
        self._model.load_weights(filepath=f)
        self._target_model.load_weights(filepath=f)

    def load_model(self, f):
        self._model = None
        self._target_model = None
        self._model = tf.keras.models.load_model(f)
        self._target_model = tf.keras.models.load_model(f)

    def append_to_buffer(self, state, action, reward, new_state, done):
        sample = state, action, reward, new_state, done
        self._buffer.add(self.TD_size(sample), sample)

    def _n_best_Q(self, arr, n):
        """
        Returns the n largest indices from a numpy array.
        """
        indices = self._largest_indices(arr, n)
        return np.sum(arr[indices])/n

    def _largest_indices(self, arr, n):
        """
        Returns the n indices with largest values from a numpy array.
        """
        flat = arr.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, arr.shape)


def evaluate(supervised, dqn):
    # evalenv = gym.make('lidareval-v0')
    evalenv = gym.make('lidarsmalleval-v0')
    done = False
    reward_overall = 0
    obv = evalenv.reset()
    print('Evaluation started')
    reconstucted = np.zeros(dqn._map_shape)
    sparse = np.zeros(dqn._map_shape)
    step = 0
    while not done:
        rays = dqn.predict([reconstucted, sparse])
        obv, reward, done, _ = evalenv.step({'map': reconstucted, 'rays': rays})
        reward_overall += reward
        sparse = obv['X']
        reconstucted = supervised.predict(sparse)
        step += 1
        if step == 100:
            with open('train_log', 'a+') as f:
                f.write(ray_string(rays))
    with open('train_log', 'a+') as f:
        f.write(str(reward_overall))
    print('Evaluation ended with value: ' + str(reward_overall))
    return reward_overall


def ray_string(action_in):
    to_print = np.empty(action_in.shape, dtype=str)
    to_print[:] = ' '
    to_print[action_in] = '+'
    ret = '\n--------------------------------------------------------' \
          '----------------------------------------------------------------------------\n'
    for i in range(action_in.shape[1]):
        ret += '|'
        ret += ''.join(to_print[:, i])
        ret += '|\n'
    ret += '----------------------------------------------------------' \
           '--------------------------------------------------------------------------\n\n'
    return ret


if __name__ == "__main__":
    # env = gym.make('lidar-v0')
    env = gym.make('lidarsmall-v0')

    dql_agent = DQN(env=env)
    supervised = Supervised()

    home = expanduser("~")
    loaddir = os.path.join(home, 'trained_models/supervised_small_model_-242.64441054044056.h5')
    supervised.load_weights(loaddir)
    # dql_agent.load_model(os.path.join(home, 'trained_models/dqn_model_-250.39402455340868.h5'))
    savedir = os.path.join(home, 'Projekt/lidar-gym/trained_models/')

    shape = dql_agent._map_shape

    episode = 0
    max_reward = -float('inf')

    while True:

        done = False
        curr_state = env.reset()
        curr_state = [np.zeros((shape[0], shape[1], shape[2])), np.zeros((shape[0], shape[1], shape[2]))]
        print('\n------------------- Drive number', episode, '-------------------------')
        # training
        while not done:
            action = dql_agent.act(curr_state)
            new_state, reward, done, _ = env.step({'rays': action, 'map': curr_state[1]})

            new_state = [new_state['X'], supervised.predict(new_state['X'])]
            dql_agent.append_to_buffer(curr_state, action, reward, new_state, done)

            dql_agent.replay()        # internally iterates inside (prediction) model
            dql_agent.target_train()  # updates target model

            curr_state = new_state
            print('.', end='', flush=True)

        # evaluation and saving
        print('end of episode')

        episode += 1

        if episode % 25 == 0:
            rew = evaluate(supervised, dql_agent)
            if rew > max_reward:
                print('new best agent - saving with reward:' + str(rew))
                max_reward = rew
                dql_agent.save_model(savedir + 'dqn_model_' + str(max_reward) + '.h5')

