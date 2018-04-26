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
        obv, reward, done, _ = evalenv.step({'map': reconstucted, 'rays': rays})
        print(reward)
        reward_overall += reward
        sparse = obv['X']
        reconstucted = supervised.predict(sparse)
        if episode % 10 == 0:
            evalenv.render()
        episode += 1
    print('Evaluation ended with value: ' + str(reward_overall))
    return reward_overall


if __name__ == "__main__":
    evalenv = gym.make('lidareval-v0')

    # Create models on GPU
    agent = Supervised()

    sess = tf.Session()
    K.set_session(sess)
    agent_rays = ActorCritic(evalenv, sess)

    home = expanduser("~")

    loaddir = os.path.join(home, 'trained_models/supervised_model_-205.62373544534486.h5')
    agent.load_weights(loaddir)
    actor = os.path.join(home, 'Projekt/lidar-gym/trained_models/actor_-274.21080331962463.h5')
    critic = os.path.join(home, 'Projekt/lidar-gym/trained_models/critic_-274.21080331962463.h5')
    agent_rays.load_model(actor, critic)
    evaluate(agent, agent_rays)