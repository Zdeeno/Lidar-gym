from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import os
from os.path import expanduser
from lidar_gym.agent.supervised_agent import Supervised


def evaluate(supervised):
    evalenv = gym.make('lidareval-v0')
    done = False
    reward_overall = 0
    _ = evalenv.reset()
    map = np.zeros((320, 320, 32))
    print('Evaluation started!')
    while not done:
        a = evalenv.action_space.sample()
        obv, reward, done, _ = evalenv.step({'map': map, 'rays': a['rays']})
        reward_overall += reward
        map = supervised.predict(obv['X'])
    print('Evaluation done with reward - ' + str(reward_overall))
    return reward_overall


if __name__ == "__main__":

    LOAD = False
    # Create model on GPU
    agent = Supervised()

    home = expanduser("~")
    savedir = os.path.join(home, 'trained_models/')

    if LOAD:
        loaddir = expanduser("~")
        loaddir = os.path.join(loaddir, 'Projekt/lidar-gym/trained_models/my_keras_model_supervised.h5')
        agent.load_weights(loaddir)

    evaluate(agent)