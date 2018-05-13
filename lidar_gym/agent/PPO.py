"""
solving pendulum using actor-critic model with PPO optimization
"""

import gym
import numpy as np
from lidar_gym.agent.models import create_ppo_toy_actor_model, create_ppo_toy_critic_model
import tensorflow.contrib.keras.api.keras.backend as K
import tensorflow as tf
from lidar_gym.agent.supervised_agent import Supervised
from lidar_gym.tools.sum_tree import Memory

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
        self.max_rays = 15
        self.output_shape = (40, 30)
        self.lidar_shape = (2, self.max_rays)
        self.env = env
        self.sess = sess

        self.action_size = self.lidar_shape[0]*self.lidar_shape[1]
        self.learning_rate = 0.001
        self.batch_size = 8
        self.buffer_size = 1024

        # OU consts
        self.epsilon = 1
        self.epsilon_decay = 1/(1000*200)
        self.mean = 0
        self.theta = 0
        self.sigma = 0.2

        self.buffer = Memory(self.buffer_size)
        self.actor_model = create_ppo_toy_actor_model(self.learning_rate, self.map_shape)

        # critic model
        self.critic_model = create_ppo_toy_critic_model(self.learning_rate, self.map_shape,
                                                        self.lidar_shape)

    def append_to_buffer(self, state, action, reward, new_state, done):
        sample = state, action, reward, new_state, done
        self.buffer.add(self._TD_size(sample), sample)

    def _train_actor(self, batch):
        idxs, cur_states, actions, rewards, new_states, dones = batch
        predicted_actions = self.actor_model.predict([cur_states[:, 0], cur_states[:, 1],
                                                     np.zeros((self.batch_size, )),
                                                     np.zeros((self.batch_size, )),
                                                     np.zeros((self.batch_size, 2, self.max_rays))],
                                                     batch_size=self.batch_size)
        predicted_values = self.critic_model.predict([cur_states[:, 0], cur_states[:, 1]])

        self.actor_model.fit(x=[cur_states[:, 0], cur_states[:, 1], rewards, predicted_values, actions],
                             y=[predicted_actions], verbose=0)

    def _train_critic(self, batch):
        idxs, cur_states, actions, rewards, new_states, dones = batch

        self.critic_model.fit([cur_states[:, 0], cur_states[:, 1]], rewards,
                              verbose=0, batch_size=self.batch_size)

        # count TDs and update sum tree
        pred_Q = self.critic_model.predict([cur_states[:, 0], cur_states[:, 1]],
                                           batch_size=self.batch_size)[:, 0]
        for i in range(self.batch_size):
            td = np.abs(pred_Q[i] - rewards[i])
            self.buffer.update(idxs[i], td)

    def train(self):
        if self.buffer.length < self.batch_size:
            return

        self._train_critic(self._get_batch())
        self._train_actor(self._get_batch())

    # predictions
    def predict(self, state):
        state = [np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)]
        return self.actor_model.predict([state[0], state[1],
                                        np.zeros((1, )),
                                        np.zeros((1, )),
                                        np.zeros((1, 2, self.max_rays))])

    def predict_perturbed(self, state):
        # Ornstein-Uhlenbeck perturbations
        probs = self.predict(state)
        print(probs)
        noise = self.theta * (self.mean - probs) + self.sigma * np.random.standard_normal(probs.shape)
        probs_perturbed = probs + self.epsilon * noise
        self.epsilon -= self.epsilon_decay

        return probs_perturbed[0]

    def c2d(self, input):
        # continous action to 2D discrete array
        assert input.ndim == 2, 'has shape: ' + str(input.shape)
        half_shape = np.asarray(self.output_shape)/2
        azimuth = np.asarray(input[0, :] * half_shape[0] + half_shape[0], dtype=int)
        elevation = np.asarray(input[1, :] * half_shape[1] + half_shape[1], dtype=int)
        # avoid err outputs
        azimuth = np.clip(azimuth, 0, self.output_shape[0] - 1)
        elevation = np.clip(elevation, 0, self.output_shape[1] - 1)

        ret = np.zeros(self.output_shape, dtype=bool)
        ret[azimuth, elevation] = True
        return ret

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

        TD = np.abs(self.critic_model.predict([cur_state[0], cur_state[1]])[0][0] - reward)
        return TD

    def _get_batch(self):
        samples = self.buffer.sample(self.batch_size)
        # data holders
        idxs = np.empty(self.batch_size, dtype=int)
        cur_states = np.empty((self.batch_size, 2,) + self.map_shape)
        actions = np.empty((self.batch_size, ) + self.lidar_shape)
        rewards = np.empty(self.batch_size)
        new_states = np.empty((self.batch_size, 2,) + self.map_shape)
        dones = np.empty(self.batch_size, dtype=bool)

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
        a = reinforce.predict([reconstucted, sparse])[0]
        rays = reinforce.c2d(a)
        obv, reward, done, _ = evalenv.step({'map': reconstucted, 'rays': rays})
        reward_overall += reward
        sparse = obv['X']
        reconstucted = supervised.predict(sparse)
        step += 1
        evalenv.render(mode='ASCII')
    with open('train_log_PPO', 'a+') as f:
        f.write(str(reward_overall) + '@' + str(episode) + '\n')
    print('Evaluation after episode ' + str(episode) + ' ended with value: ' + str(reward_overall))
    return reward_overall


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

    load_actor = os.path.join(home, 'Projekt/lidar-gym/trained_models/actor_-260.2020874046176.h5')
    load_critic = os.path.join(home, 'Projekt/lidar-gym/trained_models/critic_-260.2020874046176.h5')
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
            action = model.c2d(rays)
            new_state, reward, done, _ = env.step({'rays': action, 'map': curr_state[1]})

            new_state = [new_state['X'], supervised.predict(new_state['X'])]
            model.append_to_buffer(curr_state, rays, reward, new_state, done)

            model.train()

            curr_state = new_state
            epoch += 1
            print('.', end='', flush=True)
            env.render(mode='ASCII')

        episode += 1
        # evaluation and saving
        print('\nend of episode')
        if episode % 25 == 0:
            # model.perturbation_decay()
            rew = evaluate(supervised, model)
            if rew > max_reward:
                print('new best agent - saving with reward:' + str(rew))
                max_reward = rew
                critic_name = 'critic_PPO' + str(max_reward) + '.h5'
                actor_name = 'actor_PPO' + str(max_reward) + '.h5'
                model.save_model(savedir + critic_name, savedir + actor_name)
