from __future__ import division, print_function, absolute_import

import threading
import random
import numpy as np
import time
from collections import deque
import lidar_gym.tools.math_processing as mp
import gym
import tensorflow as tf
import tflearn
import voxel_map as vm

# Fix for TF 0.12
try:
    writer_summary = tf.summary.FileWriter
    merge_all_summaries = tf.summary.merge_all
    histogram_summary = tf.summary.histogram
    scalar_summary = tf.summary.scalar
except Exception:
    writer_summary = tf.train.SummaryWriter
    merge_all_summaries = tf.merge_all_summaries
    histogram_summary = tf.histogram_summary
    scalar_summary = tf.scalar_summary

# =============================
#   Constants
# =============================

# Map constants
VOXEL_SIZE = 0.2
INPUT_SHAPE = (320, 320, 32)
# Change that value to test instead of train
testing = False
# Model path (to load when testing)
test_model_path = '/path/to/your/qlearning.tflearn.ckpt'
# Atari game to learn
# You can also try: 'Breakout-v0', 'Pong-v0', 'SpaceInvaders-v0', ...
game = 'lidar-v1'
# Learning threads
n_threads = 1

# =============================
#   Training Parameters
# =============================
# Max training steps
TMAX = 80000000
# Current training step
T = 0
# Timestep to reset the target network
I_target = 40000
# Learning rate
learning_rate = 0.001
# Reward discount rate
gamma = 0.99
# Number of timesteps to anneal epsilon
anneal_epsilon_timesteps = 400000

# =============================
#   Utils Parameters
# =============================
# Display or not gym evironment screens
show_training = True
# Directory for storing tensorboard summaries
summary_dir = '/tmp/tflearn_logs/'
summary_interval = 100
checkpoint_path = 'qlearning.tflearn.ckpt'
checkpoint_interval = 2000
# Number of episodes to run gym evaluation
num_eval_episodes = 100


# =============================
#   TFLearn Deep Q Network
# =============================


def build_dqn():
    """
    Building a DQN.
    """
    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    inputs = tf.placeholder(tf.float32, [None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], 1])
    net = inputs
    net = tflearn.conv_3d(net, 2, 2, strides=1, activation='relu')
    net = tflearn.conv_3d(net, 4, 4, strides=1, activation='relu')
    net = tflearn.max_pool_3d(net, 2, strides=2)
    net = tflearn.conv_3d(net, 8, 4, strides=1, activation='relu')
    net = tflearn.max_pool_3d(net, 2, strides=2)
    net = tflearn.conv_3d(net, 16, 4, strides=1, activation='relu')
    net = tflearn.conv_3d(net, 32, 8, strides=1, activation='relu')
    net = tflearn.conv_3d(net, 1, 8, strides=1)
    q_values = tflearn.layers.conv.conv_3d_transpose(net, 1, 8, [INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
    return inputs, q_values

# =============================
#   Environment Wrapper
# =============================


class MyEnvironment:
    """
    Small wrapper for gym environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size from which environment state is constructed.
    """
    def __init__(self, gym_env, voxel_size, input_shape):
        self.env = gym_env
        self.cuboid_getter = mp.CuboidGetter(voxel_size, input_shape)
        self.curr_T = np.eye(4)
        self.shift_T = np.eye(4, dtype=float)
        self.shift_T[0, 3] = -0.25 * input_shape[0] * voxel_size
        self.shift_T[1, 3] = -0.5 * input_shape[1] * voxel_size
        self.shift_T[2, 3] = -0.5 * input_shape[2] * voxel_size
        self.voxel_map_shape = input_shape
        self.voxel_size = voxel_size
        self.voxel_map = None

    def get_initial_state(self):
        """
        Resets the environment, clears the voxel map.
        """
        # Clear the state buffer
        self.voxel_map = vm.VoxelMap(self.voxel_size)
        self.voxel_map.voxel_size = self.voxel_size
        self.voxel_map.free_update = - 1.0
        self.voxel_map.hit_update = 1.0
        self.voxel_map.occupancy_threshold = 0.0
        obs = self.env.reset()
        x_t = self.get_preprocessed_nn_input(obs[0])
        return x_t

    def get_preprocessed_nn_input(self, obs):
        # update voxel map
        curr_T = obs['T'][0]
        new_points = np.transpose(obs['points'])
        new_vals = obs['values']
        last_vals = self.voxel_map.get_voxels(new_points)
        new_vals = last_vals + new_vals
        self.voxel_map.set_voxels(new_points, new_vals)

        # get CNN input
        points, values = self.cuboid_getter.get_map_cuboid(self.voxel_map, curr_T, self.shift_T)
        points = np.asarray(points/self.voxel_size, dtype=int)
        ret = np.zeros(shape=self.voxel_map_shape)
        ret[points[0], points[1], points[2]] = values
        return ret

    def step(self, action):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """

        obs, r_t, terminal, info = self.env.step(action)
        x_t1 = self.get_preprocessed_nn_input(obs[0])

        return x_t1, r_t, terminal, info


# =============================
#   1-step Q-Learning
# =============================
def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


def actor_learner_thread(thread_id, env, session, graph_ops, summary_ops, saver):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    global TMAX, T

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, assign_ops, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    env = MyEnvironment(gym_env=env)

    # Initialize network gradients
    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))

    time.sleep(3*thread_id)
    t = 0
    while T < TMAX:
        # Get initial game observation
        s_t = env.get_initial_state()

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        while True:
            # Forward the deep q network, get Q(s,a) values
            output = q_values.eval(session=session, feed_dict={s: [s_t]})

            # Choose next action based on e-greedy policy
            if random.random() <= epsilon:
                action = env.env.action_space.sample()
            else:
                action = {'map': output, 'rays': env.env.action_space.sample()['rays']}

            # Scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps

            # Gym excecutes action in game environment on behalf of actor-learner
            s_t1, r_t, terminal, info = env.step(action)
            # Accumulate gradients
            readout_j1 = target_q_values.eval(session=session,
                                              feed_dict={st: [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + gamma * np.max(readout_j1))

            a_batch.append(output)
            s_batch.append(s_t)

            # Update the state and counters
            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(output)

            # Optionally update target network
            if T % I_target == 0:
                session.run(reset_target_network_params)

            # Optionally update online network
            if s_batch:
                session.run(grad_update, feed_dict={y: y_batch,
                                                    a: a_batch,
                                                    s: s_batch})
            # Clear gradients
            s_batch = []
            a_batch = []
            y_batch = []

            # Save model progress
            if t % checkpoint_interval == 0:
                saver.save(session, "qlearning.ckpt", global_step=t)

            # Print end of episode stats
            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(assign_ops[i],
                                {summary_placeholders[i]: float(stats[i])})
                print("| Thread %.2i" % int(thread_id), "| Step", t,
                      "| Reward: %.2i" % int(ep_reward), " Qmax: %.4f" %
                      (episode_ave_max_q/float(ep_t)),
                      " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                      (t/float(anneal_epsilon_timesteps)))
                break


def build_graph():
    # Create shared deep q network
    s, q_network = build_dqn()
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network
    st, target_q_network = build_dqn()
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network

    # Op for periodically updating target network with online network weights
    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], 1])
    y = tf.placeholder("float", [None])
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grad_update = optimizer.minimize(y, var_list=network_params)

    graph_ops = {"s": s,
                 "q_values": q_values,
                 "st": st,
                 "target_q_values": target_q_values,
                 "reset_target_network_params": reset_target_network_params,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}

    return graph_ops


# Set up some episode summary ops to visualize on tensorboard.
def build_summaries():
    episode_reward = tf.Variable(0.)
    scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    scalar_summary("Qmax Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    scalar_summary("Epsilon", logged_epsilon)
    # Threads shouldn't modify the main graph, so we use placeholders
    # to assign the value of every summary (instead of using assign method
    # in every thread, that would keep creating new ops in the graph)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float")
                            for i in range(len(summary_vars))]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i])
                  for i in range(len(summary_vars))]
    summary_op = merge_all_summaries()
    return summary_placeholders, assign_ops, summary_op


def train(session, graph_ops, saver):
    """
    Train a model.
    """

    # Set up game environments (one per thread)
    envs = [gym.make(game) for i in range(n_threads)]

    summary_ops = build_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    writer = writer_summary(summary_dir + "/qlearning", session.graph)

    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    # Start n_threads actor-learner training threads
    actor_learner_threads = \
        [threading.Thread(target=actor_learner_thread,
                          args=(thread_id, envs[thread_id], session,
                                graph_ops,
                                summary_ops, saver))
         for thread_id in range(n_threads)]
    for t in actor_learner_threads:
        t.start()
        time.sleep(0.01)

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        if show_training:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now


def evaluation(session, graph_ops, saver):
    """
    Evaluate a model.
    """
    saver.restore(session, test_model_path)
    print("Restored model weights from ", test_model_path)
    monitor_env = gym.make(game)
    monitor_env.monitor.start("qlearning/eval")

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    # Wrap env with AtariEnvironment helper class
    env = MyEnvironment(monitor_env, VOXEL_SIZE, INPUT_SHAPE)

    for i_episode in range(num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session=session, feed_dict={s : [s_t]})
            action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print(ep_reward)
    monitor_env.monitor.close()


def main(_):
    with tf.Session() as session:
        graph_ops = build_graph()
        saver = tf.train.Saver(max_to_keep=5)

        if testing:
            evaluation(session, graph_ops, saver)
        else:
            train(session, graph_ops, saver)

if __name__ == "__main__":
    tf.app.run()