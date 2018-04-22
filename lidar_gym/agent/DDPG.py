"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Input, Lambda, Conv3D, MaxPool3D, Conv2D,\
                                                      MaxPool2D, Reshape
from tensorflow.contrib.keras.api.keras.backend import squeeze, expand_dims, reshape
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.layers import Add, Multiply
from tensorflow.contrib.keras.api.keras.optimizers import Adam
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.contrib.keras.api.keras.activations import softmax
import tensorflow as tf
from lidar_gym.agent.supervised_agent import Supervised

import random
from collections import deque
from os.path import expanduser
import os


class ActorCritic:

    def __init__(self, env, sess):
        self.map_shape = (320, 320, 32)
        self.lidar_shape = (160, 120)
        self.max_rays = 200
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .25
        self.batch_size = 8
        self.buffer_size = 200

        self.buffer = deque(maxlen=self.buffer_size)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        # where we will feed de/dC (from critic)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.lidar_shape[0], self.lidar_shape[1]])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # critic model
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        # where we calculate de/dC for feeding above
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # Model Definitions
    def create_actor_model(self):
        reconstructed_input = Input(shape=self.map_shape)
        r11 = Lambda(lambda x: expand_dims(x, -1))(reconstructed_input)
        c11 = Conv3D(2, 4, padding='same', activation='relu')(r11)
        p11 = MaxPool3D(pool_size=2)(c11)
        c21 = Conv3D(4, 4, padding='same', activation='relu')(p11)

        sparse_input = Input(shape=self.map_shape)
        r12 = Lambda(lambda x: expand_dims(x, -1))(sparse_input)
        c12 = Conv3D(2, 4, padding='same', activation='relu')(r12)
        p12 = MaxPool3D(pool_size=2)(c12)
        c22 = Conv3D(4, 4, padding='same', activation='relu')(p12)

        # merge inputs
        c2 = Add()([c21, c22])
        c3 = Conv3D(1, 4, padding='same', activation='relu')(c2)
        s1 = Lambda(lambda x: squeeze(x, 4))(c3)
        c4 = Conv2D(9, 4, padding='same', activation='relu')(s1)
        r2 = Reshape((480, 480, 1))(c4)
        p2 = MaxPool2D(pool_size=(3, 4))(r2)
        c5 = Conv2D(2, 4, padding='same', activation='linear')(p2)
        c6 = Conv2D(1, 4, padding='same', activation='softmax')(c5)
        output = Lambda(lambda x: squeeze(x, 3))(c6)

        ret_model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
        adam = Adam(lr=0.001)
        ret_model.compile(loss='mse', optimizer=adam)
        return [sparse_input, reconstructed_input], ret_model

    def create_critic_model(self):

        reconstructed_input = Input(shape=self.map_shape)
        r11 = Lambda(lambda x: expand_dims(x, -1))(reconstructed_input)
        c11 = Conv3D(2, 4, padding='same', activation='relu')(r11)
        p11 = MaxPool3D(pool_size=2)(c11)
        c21 = Conv3D(4, 4, padding='same', activation='relu')(p11)

        sparse_input = Input(shape=self.map_shape)
        r12 = Lambda(lambda x: expand_dims(x, -1))(sparse_input)
        c12 = Conv3D(2, 4, padding='same', activation='relu')(r12)
        p12 = MaxPool3D(pool_size=2)(c12)
        c22 = Conv3D(4, 4, padding='same', activation='relu')(p12)

        action_input = Input(shape=self.lidar_shape)
        r13 = Lambda(lambda x: expand_dims(x, -1))(action_input)
        c13 = Conv2D(2, 4, padding='same', activation='relu')(r13)
        c23 = Conv2D(4, 4, padding='same', activation='relu')(c13)

        # merge action inputs and output reward
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
        output = Dense(1, activation='linear')(c5)

        ret_model = Model(inputs=[sparse_input, reconstructed_input, action_input], outputs=output)

        adam = Adam(lr=0.001)
        ret_model.compile(loss="mse", optimizer=adam)
        return [sparse_input, reconstructed_input], action_input, ret_model

    # training
    def append_to_buffer(self, state, action, reward, new_state, done):
        if len(self.buffer) > 0:
            _, _, _, state, _ = self.buffer[-1]
        self.buffer.append([state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            state, action, reward, new_state, _ = sample
            state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
            predicted_action = self.actor_model.predict(state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            cur_state = [np.expand_dims(cur_state[0], axis=0), np.expand_dims(cur_state[1], axis=0)]
            action = np.expand_dims(action, axis=0)

            if not done:
                new_state = [np.expand_dims(new_state[0], axis=0), np.expand_dims(new_state[1], axis=0)]
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state[0], new_state[1], target_action])[0][0]
                reward += self.gamma * future_reward
            print(reward)
            self.critic_model.fit([cur_state[0], cur_state[1], action], reward, verbose=0)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # updating target models
    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i] * self.tau + actor_target_weights[i] * (1 - self.tau)
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i] * self.tau + critic_target_weights[i] * (1 - self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # predictions
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return np.asarray(self.env.action_space.sample()['rays'], dtype=np.float)
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        rays = self.actor_model.predict(state)[0]
        return rays

    def predict(self, state):
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        return self.probs_to_bools(self.target_actor_model.predict(state)[0])

    def probs_to_bools(self, probs):
        assert probs.ndim == 2, 'has shape: ' + str(probs.shape)
        ret = np.zeros(shape=self.lidar_shape, dtype=bool)
        ret[self._largest_indices(probs, self.max_rays)] = True
        return ret

    def _largest_indices(self, arr, n):
        """
        Returns the n largest indices from a numpy array.
        """
        flat = arr.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, arr.shape)

    def save_model(self, critic_path, actor_path):
        self.target_critic_model.save_weights(filepath=critic_path)
        self.target_actor_model.save_weights(filepath=actor_path)


def evaluate(supervised, reinforce):
    evalenv = gym.make('lidareval-v0')
    done = False
    reward_overall = 0
    _ = evalenv.reset()
    print('Evaluation started')
    reconstucted = np.zeros(reinforce.map_shape)
    sparse = np.zeros(reinforce.map_shape)
    while not done:
        rays = reinforce.predict([reconstucted, sparse])
        obv, reward, done, _ = evalenv.step({'map': reconstucted, 'rays': rays})
        reward_overall += reward
        sparse = obv['X']
        reconstucted = supervised.predict(sparse)
    print('Evaluation ended with value: ' + str(reward_overall))
    return reward_overall


if __name__ == "__main__":
    env = gym.make('lidar-v0')
    env.seed(1)

    sess = tf.Session()
    K.set_session(sess)

    model = ActorCritic(env, sess)
    supervised = Supervised()

    home = expanduser("~")
    loaddir = os.path.join(home, 'trained_models/supervised_model_-209.51747300555627.h5')
    supervised.load_weights(loaddir)
    savedir = os.path.join(home, 'Projekt/lidar-gym/trained_models/')

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
            action_prob = model.act(curr_state)
            new_state, reward, done, _ = env.step({'rays': model.probs_to_bools(action_prob), 'map': curr_state[0]})

            new_state = [new_state['X'], supervised.predict(new_state['X'])]
            model.append_to_buffer(curr_state, action_prob, reward, new_state, done)

            model.train()

            curr_state = new_state
            epoch += 1

        # evaluation and saving
        print('end of episode')
        if episode % 5 == 0:
            rew = evaluate(supervised, model)
            if rew > max_reward:
                print('new best agent - saving with reward:' + str(rew))
                max_reward = rew
                critic_name = 'critic_' + str(max_reward) + '.h5'
                actor_name = 'actor_' + str(max_reward) + '.h5'
                model.save_model(savedir + critic_name, savedir + actor_name)

        episode += 1
