import numpy as np

import tensorforce.core.networks as networks
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import lidar_gym
import tflearn
import tensorflow as tf

# Create an OpenAIgym environment
env = OpenAIGym('lidar-v0', visualize=False)
# for some reason is env.state['min_value'] of wrong shape


class MyNetwork(networks.Network):

    def tf_apply(self, x, internals, update, return_internals=False):

        cnn_input = x['state']
        cnn_input = tf.expand_dims(cnn_input, -1)

        # CNN
        net = tflearn.conv_3d(cnn_input, 2, 2, strides=1, activation='relu', regularizer='L2')
        net = tflearn.conv_3d(net, 4, 4, strides=1, activation='relu', regularizer='L2')
        net = tflearn.max_pool_3d(net, 2, strides=2)
        net = tflearn.conv_3d(net, 8, 4, strides=1, activation='relu', regularizer='L2')
        net = tflearn.max_pool_3d(net, 2, strides=2)
        net = tflearn.conv_3d(net, 16, 4, strides=1, activation='relu', regularizer='L2')
        net = tflearn.conv_3d(net, 1, 8, strides=1, activation='relu', regularizer='L2')
        # net = tflearn.layers.conv.conv_3d_transpose(net, 1, 8, [320, 320, 32], strides=[1, 4, 4, 4, 1])
        # net = tf.squeeze(net, [4])
        # net = tf.reshape(net, [tf.shape(net)[0], 3276800])

        if return_internals:
            return net, dict()
        else:
            return net


# specify agent
update_mode = dict(unit='episodes', batch_size=1)
memory = dict(type='latest', include_next_states=False, capacity=1)


agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=MyNetwork,
    batching_capacity=1,
    update_mode=update_mode,
    memory=memory
)


# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=500, episode_finished=episode_finished)


# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
