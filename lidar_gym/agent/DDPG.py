"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Input, Lambda, Conv3D, MaxPool3D, Conv2D,\
                                                      MaxPool2D, Reshape, Flatten, Layer, Add, Multiply
from tensorflow.contrib.keras.api.keras.backend import squeeze, expand_dims, reshape
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.optimizers import Adam
import tensorflow.contrib.keras.api.keras.backend as K
import tensorflow as tf
from lidar_gym.agent.supervised_agent import Supervised
from lidar_gym.tools.sum_tree import Memory

import random
from collections import deque
from os.path import expanduser
import os


class ActorCritic:

    def __init__(self, env, sess):
        # self.map_shape = (320, 320, 32)
        # self.lidar_shape = (160, 120)
        # self.max_rays = 200
        self.map_shape = (160, 160, 16)
        self.lidar_shape = (120, 90)
        self.action_size = self.lidar_shape[0]*self.lidar_shape[1]
        self.max_rays = 100
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = .999
        self.min_epsilon = 0.2
        self.gamma = .95
        self.tau = .15
        self.batch_size = 7
        self.buffer_size = 1000
        self.num_proto_actions = 5

        self.pert_threshold_dist = 0.025
        self.pert_threshold_decay = 0.995
        self.pert_alpha = 1.01
        self.pert_variance = self.pert_threshold_dist

        self.buffer = Memory(self.buffer_size)
        self.actor_sparse_input, self.actor_reconstructed_input, self.actor_model = self._create_actor_model()
        _, _, self.target_actor_model = self._create_actor_model()
        _, _, self.perturbed_actor_model = self._create_actor_model()

        # where we will feed de/dC (from critic)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.lidar_shape[0], self.lidar_shape[1]])

        actor_model_weights = self.actor_model.trainable_weights
        # dC/dA (from actor)
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # critic model
        self.critic_sparse_input, self.critic_reconstructed_input,\
            self.critic_action_input, self.critic_model = self._create_critic_model()
        _, _, _, self.target_critic_model = self._create_critic_model()

        # where we calculate de/dC for feeding above
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # Model Definitions
    def _create_actor_model(self):
        reconstructed_input = Input(shape=self.map_shape)
        r11 = Lambda(lambda x: expand_dims(x, -1))(reconstructed_input)
        c11 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r11)
        p11 = MaxPool3D(pool_size=2)(c11)
        c21 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p11)

        sparse_input = Input(shape=self.map_shape)
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
        c5 = Conv2D(2, 4, padding='same', activation='linear')(p2)
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
        c5 = Conv2D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(p2)
        c6 = Conv2D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(c5)
        c7 = Conv2D(1, 4, padding='same', activation='linear', kernel_regularizer='l2')(c6)
        sigmoid = Lambda(lambda x: x/(tf.abs(x)+50))(c7)
        bias = Lambda(lambda x: (x+1)/2)(sigmoid)

        output = Lambda(lambda x: squeeze(x, 3))(bias)

        # print(alpha.shape, adda.shape, const.shape, output.shape)

        ret_model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
        adam = Adam(lr=0.001)
        ret_model.compile(loss='mse', optimizer=adam)
        return sparse_input, reconstructed_input, ret_model

    def _create_critic_model(self):

        reconstructed_input = Input(shape=self.map_shape)
        r11 = Lambda(lambda x: expand_dims(x, -1))(reconstructed_input)
        c11 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r11)
        p11 = MaxPool3D(pool_size=2)(c11)
        c21 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p11)

        sparse_input = Input(shape=self.map_shape)
        r12 = Lambda(lambda x: expand_dims(x, -1))(sparse_input)
        c12 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r12)
        p12 = MaxPool3D(pool_size=2)(c12)
        c22 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p12)

        action_input = Input(shape=self.lidar_shape)
        r13 = Lambda(lambda x: expand_dims(x, -1))(action_input)
        c13 = Conv2D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(r13)
        c23 = Conv2D(1, 4, padding='same', activation='relu', kernel_regularizer='l2')(c13)

        '''
        # merge LARGE action inputs and output reward
        a1 = Add()([c21, c22])
        c1 = Conv3D(1, 4, padding='same', activation='relu')(a1)
        s1 = Lambda(lambda x: squeeze(x, 4))(c1)
        c2 = Conv2D(9, 4, padding='same', activation='relu')(s1)
        r1 = Reshape((480, 480, 1))(c2)
        p1 = MaxPool2D(pool_size=(3, 4))(r1)
        a2 = Add()([p1, c23])
        c3 = Conv2D(2, 4, padding='same', activation='relu')(a2)
        p2 = MaxPool2D(pool_size=4, strides=4)(c3)
        c4 = Conv2D(4, 4, padding='same', activation='relu')(p2)
        p3 = MaxPool2D(pool_size=4, strides=4)(c4)
        c5 = Conv2D(1, 4, padding='same', activation='relu')(p3)
        f1 = Flatten()(c5)
        output = Dense(1, activation='linear')(f1)
        '''

        # merge SMALL action inputs and output action Q value
        a1 = Add()([c21, c22])
        c1 = Conv3D(1, 4, padding='same', activation='relu', kernel_regularizer='l2')(a1)
        s1 = Lambda(lambda x: squeeze(x, 4))(c1)
        c2 = Conv2D(8, 4, padding='same', activation='relu', kernel_regularizer='l2')(s1)
        p1 = MaxPool2D(pool_size=2)(c2)
        c3 = Conv2D(81, 4, padding='same', activation='relu', kernel_regularizer='l2')(p1)
        r2 = Reshape((360, 360, 1))(c3)
        p2 = MaxPool2D(pool_size=(3, 4))(r2)
        a2 = Multiply()([p2, c23])
        c4 = Conv2D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(a2)
        p2 = MaxPool2D(pool_size=4, strides=4)(c4)
        c5 = Conv2D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p2)
        p3 = MaxPool2D(pool_size=4, strides=4)(c5)
        c6 = Conv2D(1, 4, padding='same', activation='relu', kernel_regularizer='l2')(p3)
        f1 = Flatten()(c6)
        output = Dense(1, activation='linear')(f1)

        ret_model = Model(inputs=[sparse_input, reconstructed_input, action_input], outputs=output)

        adam = Adam(lr=0.001)
        ret_model.compile(loss="mse", optimizer=adam)
        return sparse_input, reconstructed_input, action_input, ret_model

    def append_to_buffer(self, state, action, reward, new_state, done):
        sample = state, action, reward, new_state, done
        self.buffer.add(self._TD_size(sample), sample)

    def _train_actor(self, samples):
        for sample in samples:
            idx, data = sample
            state, action, reward, new_state, _ = data
            state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
            pred = self.predict(new_state)
            predicted_action = np.expand_dims(pred, axis=0)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_sparse_input: state[0],
                self.critic_reconstructed_input: state[1],
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_sparse_input: state[0],
                self.actor_reconstructed_input: state[1],
                self.actor_critic_grad: grads
            })
        # self._soft_update(self.actor_model, self.target_actor_model, self.tau)

    def _train_critic(self, samples):
        for sample in samples:
            idx, data = sample
            cur_state, action, reward, new_state, done = data
            cur_state = [np.expand_dims(cur_state[0], axis=0), np.expand_dims(cur_state[1], axis=0)]
            action = np.expand_dims(action, axis=0)

            if not done:
                new_state = [np.expand_dims(new_state[0], axis=0), np.expand_dims(new_state[1], axis=0)]
                probs = self.target_actor_model.predict(new_state)
                target_action = np.expand_dims(self._probs_to_bestQ(probs[0], cur_state[0], cur_state[1]), axis=0)
                future_reward = self.target_critic_model.predict(
                    [new_state[0], new_state[1], target_action])[0][0]
                reward += self.gamma * future_reward

            reward = np.expand_dims(reward, axis=0)
            TD = np.abs(self.critic_model.predict([cur_state[0], cur_state[1], action])[0][0] - reward[0])
            self.buffer.update(idx, TD)
            self.critic_model.fit([cur_state[0], cur_state[1], action], reward, verbose=0)

    def train(self):
        self.epsilon *= self.epsilon_decay
        if self.buffer.length < self.batch_size:
            return

        samples = self.buffer.sample(self.batch_size)
        self._train_critic(samples)
        self._train_actor(samples)


    def _soft_update(self, target, model, tau):
        model_weights = model.get_weights()
        target_weights = target.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = model_weights[i] * tau + target_weights[i] * (1 - tau)
        target.set_weights(target_weights)

    def _update_perturbed(self):
        perturbed_weights = self.actor_model.get_weights()
        for i in range(len(perturbed_weights)):
            perturbed_weights[i] += np.random.normal(0, self.pert_variance, perturbed_weights[i].shape)
        self.perturbed_actor_model.set_weights(perturbed_weights)

    def update_target(self):
        self._soft_update(self.critic_model, self.target_critic_model, self.tau)
        self._soft_update(self.actor_model, self.target_actor_model, self.tau)

    # predictions
    def predict(self, state):
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        probs = self.actor_model.predict(state)
        return self._probs_to_bestQ(probs[0], state[0], state[1])

    def predict_perturbed(self, state):
        '''
        # parameter space perturbation
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        self._update_perturbed()
        probs = self.actor_model.predict(state)
        print(probs)
        probs_perturbed = self.perturbed_actor_model.predict(state)
        dist = np.sum(np.linalg.norm(probs - probs_perturbed))/self.action_size
        if dist > self.pert_threshold_dist:
            self.pert_variance /= self.pert_alpha
        else:
            self.pert_variance *= self.pert_alpha
            # print(dist)
        '''

        # action space perturbation
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        probs = self.actor_model.predict(state)
        print(probs)
        probs_perturbed = probs # + np.random.normal(0, self.pert_variance, probs.shape)
        dist = np.mean(np.abs(probs - probs_perturbed))
        if dist > self.pert_threshold_dist:
            self.pert_variance /= self.pert_alpha
        else:
            self.pert_variance *= self.pert_alpha
        return self._probs_to_bestQ(probs_perturbed[0], state[0], state[1])

    def perturbation_decay(self):
        self.pert_threshold_dist *= self.pert_threshold_decay

    def _probs_to_bestQ(self, probs, sparse_state, reconstructed_state):
        # pseudo - wolpetinger policy
        assert probs.ndim == 2, 'has shape: ' + str(probs.shape)
        proto_action = np.zeros(shape=(self.num_proto_actions, ) + self.lidar_shape, dtype=bool)
        # sample more rays and choose 5 actions randomly
        indexes = self._largest_indices(probs, self.max_rays*2)
        for i in range(self.num_proto_actions):
            samples = np.asarray(random.sample(list(np.transpose(indexes)), self.max_rays))
            proto_action[i, samples[:, 0], samples[:, 1]] = True
        q_values = self.critic_model.predict([sparse_state, reconstructed_state, proto_action], self.num_proto_actions)
        index_max = np.argmax(q_values)
        return proto_action[index_max]

    def _largest_indices(self, arr, n):
        """
        Returns the n largest indices from a numpy array.
        """
        flat = arr.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, arr.shape)

    def save_model(self, critic_path, actor_path):
        self.critic_model.save_weights(filepath=critic_path)
        self.actor_model.save_weights(filepath=actor_path)

    def load_model(self, f_actor, f_critic):
        self.actor_model.load_weights(filepath=f_actor)
        self.critic_model.load_weights(filepath=f_critic)

    def _TD_size(self, sample):
        cur_state, action, reward, new_state, done = sample
        cur_state = [np.expand_dims(cur_state[0], axis=0), np.expand_dims(cur_state[1], axis=0)]
        action = np.expand_dims(action, axis=0)

        if not done:
            new_state = [np.expand_dims(new_state[0], axis=0), np.expand_dims(new_state[1], axis=0)]

            probs = self.target_actor_model.predict(new_state)
            target_action = np.expand_dims(self._probs_to_bestQ(probs[0], cur_state[0], cur_state[1]), axis=0)

            future_reward = self.target_critic_model.predict(
                [new_state[0], new_state[1], target_action])[0][0]
            reward += self.gamma * future_reward

        reward = np.expand_dims(reward, axis=0)
        TD = np.abs(self.critic_model.predict([cur_state[0], cur_state[1], action])[0][0] - reward[0])
        self.critic_model.fit([cur_state[0], cur_state[1], action], reward, verbose=0)
        return TD


def evaluate(supervised, reinforce):
    # evalenv = gym.make('lidareval-v0')
    evalenv = gym.make('lidarsmalleval-v0')
    done = False
    reward_overall = 0
    _ = evalenv.reset()
    # print('Evaluation started')
    reconstucted = np.zeros(reinforce.map_shape)
    sparse = np.zeros(reinforce.map_shape)
    step = 0
    while not done:
        rays = reinforce.predict([reconstucted, sparse])
        obv, reward, done, _ = evalenv.step({'map': reconstucted, 'rays': rays})
        reward_overall += reward
        sparse = obv['X']
        reconstucted = supervised.predict(sparse)
        step += 1
        if step == 100:
            with open('train_log_DDPG', 'a+') as f:
                f.write(ray_string(rays))
    with open('train_log_DDPG', 'a+') as f:
        f.write(str(reward_overall))
    print('Evaluation ended with value: ' + str(reward_overall))
    return reward_overall


def ray_string(action_in):
    # create string to visualise action in console
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
    env.seed(1)

    sess = tf.Session()
    K.set_session(sess)

    model = ActorCritic(env, sess)
    supervised = Supervised()

    home = expanduser("~")
    loaddir = os.path.join(home, 'trained_models/supervised_small_model_-242.64441054044056.h5')
    supervised.load_weights(loaddir)
    savedir = os.path.join(home, 'Projekt/lidar-gym/trained_models/')

    load_actor = os.path.join(home, 'Projekt/lidar-gym/trained_models/actor_-272.92755734050354.h5')
    load_critic = os.path.join(home, 'Projekt/lidar-gym/trained_models/critic_-272.92755734050354.h5')
    model.load_model(load_actor, load_critic)
    shape = model.map_shape

    episode = 0
    max_reward = -float('inf')

    while True:
        done = False
        curr_state = env.reset()
        curr_state = [np.zeros((shape[0], shape[1], shape[2])), np.zeros((shape[0], shape[1], shape[2]))]
        epoch = 1
        print('\n------------------- Drive number', episode, '-------------------------')
        # training
        while not done:
            rays = model.predict_perturbed(curr_state)
            print(ray_string(rays))
            new_state, reward, done, _ = env.step({'rays': rays, 'map': curr_state[1]})

            new_state = [new_state['X'], supervised.predict(new_state['X'])]
            model.append_to_buffer(curr_state, rays, reward, new_state, done)

            model.train()
            model.update_target()

            curr_state = new_state
            epoch += 1
            print('.', end='', flush=True)
            # print(model.pert_variance)

        episode += 1
        # evaluation and saving
        print('\nend of episode')
        if episode % 25 == 0:
            model.perturbation_decay()
            rew = evaluate(supervised, model)
            if rew > max_reward:
                print('new best agent - saving with reward:' + str(rew))
                max_reward = rew
                critic_name = 'critic_' + str(max_reward) + '.h5'
                actor_name = 'actor_' + str(max_reward) + '.h5'
                model.save_model(savedir + critic_name, savedir + actor_name)

