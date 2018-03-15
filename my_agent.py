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
        print(cnn_input)

        # CNN
        net = tflearn.conv_3d(cnn_input, 2, 2, strides=1, activation='relu')
        net = tflearn.conv_3d(net, 4, 4, strides=1, activation='relu')
        net = tflearn.max_pool_3d(net, 2, strides=2)
        net = tflearn.conv_3d(net, 8, 4, strides=1, activation='relu')
        net = tflearn.max_pool_3d(net, 2, strides=2)
        net = tflearn.conv_3d(net, 16, 4, strides=1, activation='relu')
        net = tflearn.conv_3d(net, 32, 8, strides=1, activation='relu')
        net = tflearn.conv_3d(net, 1, 8, strides=1)
        net = tflearn.layers.conv.conv_3d_transpose(net, 1, 8, [320, 320, 32])

        net = tf.squeeze(net, [4])
        return net


agent = PPOAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=MyNetwork,
    batch_size=1,
    # BatchAgent
    keep_last_timestep=True,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=1,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    summary_spec=None,
    distributed_spec=None
)

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)


# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
