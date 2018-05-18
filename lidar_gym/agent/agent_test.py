from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import os
from os.path import expanduser
from lidar_gym.agent.supervised_agent import Supervised
from lidar_gym.agent.DDPG import ActorCritic
from lidar_gym.envs.lidar_gym import LidarROCHelper, LidarToyROC
import tensorflow as tf
import tensorflow.contrib.keras.api.keras.backend as K
import matplotlib.pyplot as plt


# script which creates data for ROC curve


def evaluate(supervised, reinforce, eval, help):
    done = False
    reward_overall = 0
    _ = eval._reset()
    _ = help._reset()
    print('Evaluation started')
    reconstucted = np.zeros(reinforce.map_shape)
    sparse = np.zeros(reinforce.map_shape)
    dummy_map = np.zeros(reinforce.map_shape)
    all_rays = np.ones((40, 30), dtype=bool)
    episode = 0
    while not done:
        # rays = reinforce.predict([reconstucted, sparse])
        rays = eval.action_space.sample()['rays']
        obv, reward, done, _ = eval._step({'map': reconstucted, 'rays': rays})
        _, _, _, _ = help._step({'map': dummy_map, 'rays': all_rays})
        print(reward)
        reward_overall += reward
        sparse = obv['X']
        reconstucted = supervised.predict(sparse)
        episode += 1
    print('Evaluation ended with value: ' + str(reward_overall))
    return reward_overall


if __name__ == "__main__":
    evalenv = LidarToyROC()
    helpenv = LidarROCHelper()

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

    evaluate(agent_map, agent_rays, evalenv, helpenv)

    full_sparse = helpenv._obs_voxel_map
    ground_truth = evalenv._map
    reconstructed = evalenv._rec_voxel_map

    r_x, r_l, r_v = reconstructed.get_voxels()

    min_val = np.min(r_v)
    max_val = np.max(r_v)
    step_size = (max_val-min_val)/100
    steps = np.arange(min_val, max_val, step_size)

    f_s_v = np.nan_to_num(full_sparse.get_voxels(r_x, r_l))
    g_t_v = np.nan_to_num(ground_truth.get_voxels(r_x, r_l))

    f_s_true = np.zeros(np.shape(f_s_v))
    f_s_true[f_s_v > 0] = 1

    g_t_false = np.zeros(np.shape(g_t_v))
    g_t_false[g_t_v < 0] = 1

    false_positive = np.empty(np.shape(steps))
    true_positive = np.empty(np.shape(steps))

    for i, step in enumerate(steps):
        rec_true = np.zeros(np.shape(r_v))
        rec_true[(r_v + step) > 0] = 1
        false_positive[i] = np.nan_to_num(np.sum(g_t_false*rec_true)/np.sum(g_t_false))
        true_positive[i] = np.nan_to_num(np.sum(f_s_true*rec_true)/np.sum(f_s_true))

    print(false_positive)
    print(true_positive)

    plt.plot(false_positive, true_positive, '-', antialiased=True)
    plt.show()
