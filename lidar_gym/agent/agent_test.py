from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import os
from os.path import expanduser
from lidar_gym.agent.supervised_agent import Supervised
from lidar_gym.agent.DDPG import ActorCritic
import tensorflow as tf
import tensorflow.contrib.keras.api.keras.backend as K


def evaluate_supervised(supervised):
    done = False
    reward_overall = 0
    _ = evalenv.reset()
    # map = np.zeros((320, 320, 32))
    map = np.zeros((160, 160, 16))
    evalenv.seed(1)
    episode = 0
    print('Evaluation started!')
    while not done:
        a = evalenv.action_space.sample()
        obv, reward, done, _ = evalenv.step({'map': map, 'rays': a['rays']})
        reward_overall += reward
        map = supervised.predict(obv['X'])
    print('Evaluation done with reward - ' + str(reward_overall))
    return reward_overall


def evaluate(supervised, reinforce):
    done = False
    reward_overall = 0
    _ = evalenv.reset()
    print('Evaluation started')
    reconstucted = np.zeros(reinforce.map_shape)
    sparse = np.zeros(reinforce.map_shape)
    episode = 0
    while not done:
        rays = reinforce.predict([reconstucted, sparse])
        print(rays)
        print(reconstucted)
        obv, reward, done, _ = evalenv.step({'map': reconstucted, 'rays': rays})
        print(reward)
        reward_overall += reward
        sparse = obv['X']
        reconstucted = supervised.predict(sparse)
        # if episode % 10 == 0:
            # evalenv.render()
        episode += 1
    print('Evaluation ended with value: ' + str(reward_overall))
    return reward_overall


if __name__ == "__main__":
    evalenv = gym.make('lidartoyroc-v0')

    # Create trained_models on GPU
    agent_map = Supervised()

    sess = tf.Session()
    K.set_session(sess)
    agent_rays = ActorCritic(evalenv, sess)

    home = expanduser("~")
    loaddir = os.path.join(home, 'Projekt/lidar-gym/lidar_gym/agent/trained_models/supervised_toy_model.h5')
    agent_map.load_weights(loaddir)

    actor = os.path.join(home, 'Projekt/lidar-gym/trained_models/actor_DDPG-254.9115130486834.h5')
    critic = os.path.join(home, 'Projekt/lidar-gym/trained_models/critic_DDPG-254.9115130486834.h5')
    agent_rays.load_models(actor, critic)
    evaluate(agent_map, agent_rays)
    # evaluate_supervised(agent_map)
