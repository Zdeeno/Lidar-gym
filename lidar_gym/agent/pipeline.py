import gym
import numpy as np
from tensorflow.contrib.keras.api.keras.models import Model
from lidar_gym.agent.DQL import DQN
from lidar_gym.agent.dDDPG import ActorCritic
from lidar_gym.agent.supervised_agent import Supervised
from lidar_gym.agent.DQL import DQN
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.contrib.keras.api.keras.activations import softmax
import tensorflow as tf
from lidar_gym.agent.supervised_agent import Supervised

import random
from collections import deque
from os.path import expanduser
import os


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
        obv = env.reset()
        curr_state = [np.zeros((shape[0], shape[1], shape[2])), np.zeros((shape[0], shape[1], shape[2]))]
        epoch = 1
        print('\n------------------- Drive number', episode, '-------------------------')
        # training
        while not done:
            if (episode % 10) < 5:
                    supervised.append_to_buffer(obv)
                    supervised.train_model()
                    if episode < 10:
                        print('fail')
                   #  obv, reward, done, info = env.step(obv['X'])
            else:
                # action_prob = model.act(curr_state)
                # new_state, reward, done, _ = env.step({'rays': model.probs_to_bools(action_prob), 'map': curr_state[0]})

                new_state = [new_state['X'], supervised.predict(new_state['X'])]
                # model.append_to_buffer(curr_state, action_prob, reward, new_state, done)

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
