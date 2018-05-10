import gym
import numpy as np
import random
from lidar_gym.visualiser.printer import ray_string

from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
import tensorflow.contrib.keras.api.keras.backend as K
from os.path import expanduser
import os

import tensorflow as tf
from collections import deque
import lidar_gym
from lidar_gym.agent.supervised_agent import Supervised
from lidar_gym.tools.sum_tree import Memory
from lidar_gym.agent.models import create_toy_dqn_model


class DQN:

    def __init__(self, env):
        # setup environment
        self._env = env
        self._batch_size = 8

        '''
        LARGE
        self._map_shape = (320, 320, 32)
        self._max_rays = 200
        self._rays_shape = (160, 120)
        '''

        self._map_shape = (80, 80, 8)
        self._max_rays = 15
        self._lidar_shape = (40, 30)

        # setup consts
        self._gamma = 0.9
        self._epsilon = 1.0
        self._epsilon_min = 0.25
        self._epsilon_decay = 0.999
        self._learning_rate = 0.001
        self._tau = .025

        # setup buffer
        # self._buffer_size = 200
        self._buffer_size = 1024
        self._buffer = Memory(self._buffer_size)

        # double network
        self._model = create_toy_dqn_model(self._learning_rate, self._map_shape)
        self._target_model = create_toy_dqn_model(self._learning_rate, self._map_shape)

        # logger
        home = expanduser("~")
        logdir = os.path.join(home, 'DQN_logs/')
        self._tfboard = TensorBoard(log_dir=logdir, batch_size=self._batch_size, write_graph=False)

    def predict(self, state):
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        rays = self._model.predict(state)[0]
        ret = np.zeros(shape=self._lidar_shape, dtype=bool)
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
        ret = np.zeros(shape=self._lidar_shape, dtype=bool)
        ret[self._largest_indices(rays, self._max_rays)] = True
        return ret

    def replay(self):
        self._epsilon *= self._epsilon_decay
        if self._buffer.length < self._batch_size:
            return

        idxs, cur_states, actions, rewards, new_states, dones = self._get_batch()

        Q = self._target_model.predict([cur_states[:, 0], cur_states[:, 1]])
        online_predict = self._model.predict([new_states[:, 0], new_states[:, 1]])
        targets = np.copy(Q)

        for i in range(self._batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                online_max = self._largest_indices(online_predict[i], self._max_rays)
                q_future = np.mean(self._target_model.predict(np.expand_dims(new_states[i], axis=0))[online_max])
                targets[i, action] = reward + q_future * self._gamma
            self._buffer.update(idxs[i], np.abs(np.sum(Q[i] - targets[i])))

        self._model.fit(cur_states, targets, batch_size=self._batch_size, epochs=1, verbose=0)


        '''
        for idx, sample in enumerate(samples):
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
            state_batch[idx] = state
            target_batch[idx] = target
        self._model.fit(state_batch, target_batch, batch_size=self._batch_size, epochs=1, verbose=0)
        '''

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

    def _get_batch(self):
        samples = self._buffer.sample(self._batch_size)
        # data holders
        idxs = np.empty(self._batch_size, dtype=int)
        cur_states = np.empty((self._batch_size, 2,) + self._map_shape)
        actions = np.empty((self._batch_size, ) + self._lidar_shape)
        rewards = np.empty(self._batch_size)
        new_states = np.empty((self._batch_size, 2,) + self._map_shape)
        dones = np.empty(self._batch_size, dtype=bool)

        # fill data holders
        for i, sample in enumerate(samples):
            idx, data = sample
            cur_state, action, reward, new_state, done = data
            idxs[i] = idx
            cur_states[i][0] = cur_state[0]
            cur_states[i][1] = cur_state[1]
            actions[i] = action
            rewards[i] = reward
            new_states[i][0] = new_state[0]
            new_states[i][1] = new_state[1]
            dones[i] = done

        return idxs, cur_states, actions, rewards, new_states, dones


def evaluate(supervised, dqn):
    # evalenv = gym.make('lidareval-v0')
    # evalenv = gym.make('lidarsmalleval-v0')
    evalenv = gym.make('lidartoyeval-v0')
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


if __name__ == "__main__":
    # env = gym.make('lidar-v0')
    # env = gym.make('lidarsmall-v0')
    env = gym.make('lidartoy-v0')

    dql_agent = DQN(env=env)
    supervised = Supervised()

    home = expanduser("~")
    loaddir = os.path.join(home, 'trained_models/supervised_toy_model_-255.41430450850987.h5')
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

