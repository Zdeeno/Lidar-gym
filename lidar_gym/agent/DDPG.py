"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np
from lidar_gym.agent.models import create_toy_actor_model, create_toy_critic_model
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
        '''
        # large map
        self.map_shape = (320, 320, 32)
        self.lidar_shape = (160, 120)
        self.max_rays = 200
        '''

        # toy map
        self.map_shape = (80, 80, 8)
        self.lidar_shape = (40, 30)
        self.max_rays = 15
        self.env = env
        self.sess = sess

        self.action_size = self.lidar_shape[0]*self.lidar_shape[1]
        self.learning_rate = 0.001
        self.gamma = .95
        self.tau = .01
        self.batch_size = 8
        self.buffer_size = 1024
        self.num_proto_actions = 5

        '''
        # action space perturbation consts
        self.pert_threshold_decay = 0.995
        self.pert_threshold_dist = 0.025*self.pert_threshold_decay
        self.pert_alpha = 1.01
        self.pert_variance = self.pert_threshold_dist
        '''

        # OU consts
        self.epsilon = 1
        self.epsilon_decay = 1/(1000*200)
        self.mean = 0.5
        self.theta = 0.75
        self.sigma = 0.1

        self.buffer = Memory(self.buffer_size)
        self.actor_sparse_input, self.actor_reconstructed_input, self.actor_model =\
            create_toy_actor_model(self.learning_rate, self.map_shape)
        _, _, self.target_actor_model = create_toy_actor_model(self.learning_rate, self.map_shape)
        _, _, self.perturbed_actor_model = create_toy_actor_model(self.learning_rate, self.map_shape)

        # where we will feed de/dC (from critic)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.lidar_shape[0], self.lidar_shape[1]])

        actor_model_weights = self.actor_model.trainable_weights
        # dC/dA (from actor)
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # critic model
        self.critic_sparse_input, self.critic_reconstructed_input,\
            self.critic_action_input, self.critic_model = create_toy_critic_model(self.learning_rate, self.map_shape,
                                                                                  self.lidar_shape)
        _, _, _, self.target_critic_model = create_toy_critic_model(self.learning_rate, self.map_shape,
                                                                    self.lidar_shape)

        # where we calculate de/dC for feeding above
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def append_to_buffer(self, state, action, reward, new_state, done):
        sample = state, action, reward, new_state, done
        self.buffer.add(self._TD_size(sample), sample)

    def _train_actor(self, samples):
        state_batch = np.asarray(samples[:][1][0])
        action_batch = np.asarray(samples[:][1][1])

        for i, sample in enumerate(samples):
            idx, data = sample
            state, action, reward, new_state, _ = data
            state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
            pred = self.predict(new_state)
            predicted_action = np.expand_dims(pred, axis=0)
            state_batch[i] = state
            action_batch[i] = predicted_action

        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_sparse_input: state_batch[:][0],
            self.critic_reconstructed_input: state_batch[:][1],
            self.critic_action_input: action_batch
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_sparse_input: state_batch[:][0],
            self.actor_reconstructed_input: state_batch[:][1],
            self.actor_critic_grad: grads
        })

    def _train_critic(self, samples):
        samples = np.asarray(samples)
        state_batch = samples[:][1][0]
        action_batch = samples[:][1][1]
        reward_batch = samples[:][1][2]
        new_state_batch = samples[:][1][3]
        done_batch = samples[:][1][4]

        probs = self.target_actor_model.predict(new_state_batch, batch_size=self.batch_size)

        for i in range(self.batch_size):
            if not done_batch[i]:
                target_action = np.expand_dims(self._probs_to_bestQ(probs[i], state_batch[i][0], state_batch[i][1]), axis=0)
                future_reward = self.target_critic_model.predict(
                    [np.expand_dims(new_state_batch[i][0], axis=0),
                     np.expand_dims(new_state_batch[i][1], axis=0), target_action])[0][0]
                reward_batch[i] += self.gamma * future_reward
        TDs = np.abs(self.critic_model.predict([state_batch[:][0], state_batch[:][1], action_batch],
                                               batch_size=self.batch_size)[0][0] - reward_batch)
        for i, td in enumerate(TDs):
            self.buffer.update(samples[i][1], td)
        self.critic_model.fit([state_batch[:][0], state_batch[:][1], action_batch], reward_batch,
                              verbose=0, batch_size=self.batch_size)

        '''
        print(samples[0][1][0])
        state_batch = np.empty((self.batch_size,) + np.shape(samples[0][1][0]))
        action_batch = np.empty((self.batch_size,) + samples[0][1][1].shape)
        reward_batch = np.empty((self.batch_size, ))

        for i, sample in enumerate(samples):
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
            state_batch[i] = np.asarray([np.squeeze(cur_state[0], 0), np.squeeze(cur_state[1], 0)])
            action_batch[i] = action
            reward_batch[i] = reward

        self.critic_model.fit([state_batch[:][0], state_batch[:][1], action_batch], reward_batch,
                              verbose=0, batch_size=self.batch_size)
        '''

    def train(self):
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

        '''
        # action space perturbation
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        probs = self.actor_model.predict(state)
        # print(probs)
        probs_perturbed = probs + np.random.normal(0, self.pert_variance, probs.shape)
        dist = np.mean(np.abs(probs - probs_perturbed))
        if dist > self.pert_threshold_dist:
            self.pert_variance /= self.pert_alpha
        else:
            self.pert_variance *= self.pert_alpha
        '''

        # Ornstein-Uhlenbeck policy
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        probs = self.actor_model.predict(state)
        noise = self.theta * (self.mean - probs) + self.sigma * np.random.standard_normal(probs.shape)
        probs_perturbed = probs + self.epsilon * noise
        self.epsilon -= self.epsilon_decay

        return self._probs_to_bestQ(probs_perturbed[0], state[0], state[1])

    '''
    def _update_perturbed(self):
        # for parameter space perturbations
        perturbed_weights = self.actor_model.get_weights()
        for i in range(len(perturbed_weights)):
            perturbed_weights[i] += np.random.normal(0, self.pert_variance, perturbed_weights[i].shape)
        self.perturbed_actor_model.set_weights(perturbed_weights)
    '''

    '''
    def perturbation_decay(self):
        self.pert_threshold_dist *= self.pert_threshold_decay
    '''

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
        self.critic_model.save(filepath=critic_path)
        self.actor_model.save(filepath=actor_path)

    def load_model_weights(self, f_actor, f_critic):
        self.actor_model.load_weights(filepath=f_actor)
        self.critic_model.load_weights(filepath=f_critic)

    def load_models(self, f_actor, f_critic):
        self.actor_model = tf.keras.models.load_model(filepath=f_actor)
        self.critic_model = tf.keras.models.load_model(filepath=f_critic)

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
        return TD


def evaluate(supervised, reinforce):
    # evalenv = gym.make('lidareval-v0')
    # evalenv = gym.make('lidarsmalleval-v0')
    evalenv = gym.make('lidartoyeval-v0')
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
    divider = np.empty(action_in.shape[0] + 2, dtype=str)
    divider[:] = '-'
    to_print[:] = ' '
    to_print[action_in] = '+'
    ret = '\n'
    ret += ''.join(divider)
    ret += '\n'
    for i in range(action_in.shape[1]):
        ret += '|'
        ret += ''.join(to_print[:, i])
        ret += '|\n'
    ret += ''.join(divider) + '\n\n'
    return ret


if __name__ == "__main__":
    # env = gym.make('lidar-v0')
    # env = gym.make('lidarsmall-v0')
    env = gym.make('lidartoy-v0')
    env.seed(1)

    sess = tf.Session()
    K.set_session(sess)

    model = ActorCritic(env, sess)
    supervised = Supervised()

    home = expanduser("~")
    loaddir = os.path.join(home, 'trained_models/supervised_toy_model_-255.41430450850987.h5')
    supervised.load_weights(loaddir)
    savedir = os.path.join(home, 'Projekt/lidar-gym/trained_models/')

    load_actor = os.path.join(home, 'Projekt/lidar-gym/trained_models/actor_-270.8374477013234.h5')
    load_critic = os.path.join(home, 'Projekt/lidar-gym/trained_models/critic_-270.8374477013234.h5')
    # model.load_model_weights(load_actor, load_critic)
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
            new_state, reward, done, _ = env.step({'rays': rays, 'map': curr_state[1]})

            new_state = [new_state['X'], supervised.predict(new_state['X'])]
            model.append_to_buffer(curr_state, rays, reward, new_state, done)

            model.train()
            model.update_target()

            curr_state = new_state
            epoch += 1
            print('.', end='', flush=True)
            # print(model.pert_variance)
            # print(ray_string(rays))

        episode += 1
        # evaluation and saving
        print('\nend of episode')
        if episode % 25 == 0:
            # model.perturbation_decay()
            rew = evaluate(supervised, model)
            if rew > max_reward:
                print('new best agent - saving with reward:' + str(rew))
                max_reward = rew
                critic_name = 'critic_' + str(max_reward) + '.h5'
                actor_name = 'actor_' + str(max_reward) + '.h5'
                model.save_model(savedir + critic_name, savedir + actor_name)

