import gym
import numpy as np
import random
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Input, Lambda, Conv3D, MaxPool3D, Conv2D,\
                                                      MaxPool2D, Reshape
from tensorflow.contrib.keras.api.keras.backend import squeeze, expand_dims, reshape
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.optimizers import Adam

import tensorflow as tf
from collections import deque


def logistic_loss(y_true, y_pred):
    # own objective function
    bigger = tf.cast(tf.greater(y_true, 0.0), tf.float32)
    smaller = tf.cast(tf.greater(0.0, y_true), tf.float32)

    weights_positive = 0.5 / tf.reduce_sum(bigger)
    weights_negative = 0.5 / tf.reduce_sum(smaller)

    weights = bigger*weights_positive + smaller*weights_negative

    # Here often occurs numeric instability -> nan or inf
    # return tf.reduce_sum(weights * (tf.log(1 + tf.exp(-y_pred * y_true))))
    a = -y_pred*y_true
    b = tf.maximum(0.0, a)
    t = b + tf.log(tf.exp(-b) + tf.exp(a-b))
    return tf.reduce_sum(weights*t)

buffer_X, buffer_Y, buffer_size = None, None, 0


class DQN:
    def __init__(self, env):
        # setup environment
        self._env = env
        self._batch_size = 16
        self._map_shape = (320, 320, 32)
        self._max_rays = 200

        # setup consts
        self._gamma = 0.85
        self._epsilon = 1.0
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.995
        self._learning_rate = 0.005
        self._tau = .125

        # setup buffer
        self._buffer = deque(maxlen=self._batch_size)

        # double network
        self._model = self.create_model()
        self._target_model = self.create_model()

    def create_model(self):
        state_input = Input(shape=self._map_shape)
        r1 = Lambda(lambda x: expand_dims(x, -1))(state_input)

        c1 = Conv3D(2, 4, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(r1)
        p1 = MaxPool3D(pool_size=2)(c1)
        c2 = Conv3D(4, 4, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(p1)
        c3 = Conv3D(1, 8, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(c2)
        s1 = Lambda(lambda x: squeeze(x, 4))(c3)
        c4 = Conv2D(3, 4, padding='same', kernel_regularizer=l2(0.0001), activation='relu')(s1)
        r2 = Reshape((480, 480, 1))(c4)
        p2 = MaxPool2D(pool_size=(3, 4))(r2)
        c5 = Conv2D(2, 4, padding='same', kernel_regularizer=l2(0.0001), activation='linear')(p2)
        output = Conv2D(1, 4, padding='same', kernel_regularizer=l2(0.0001), activation='softmax')(c5)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss=logistic_loss, optimizer=adam)
        return model

    def act(self, state):
        # Exploration vs exploitation
        self._epsilon *= self._epsilon_decay
        self._epsilon = max(self._epsilon_min, self._epsilon)
        if np.random.random() < self._epsilon:
            return self._env.action_space['rays'].sample()
        return self._model.predict(state)

    def replay(self):
        if len(self._buffer) < self._batch_size:
            return

        samples = random.sample(self._buffer, self._batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self._target_model.predict(state)
            if done:
                target[action == 1] = reward
            else:
                Q_future = self._n_best_Q(self._target_model.predict(new_state), self._max_rays)
                target[action] = reward + Q_future * self._gamma
            self._model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self._tau + target_weights[i] * (1 - self._tau)

    def save_model(self, fn):
        self._model.save(fn)

    def append_to_buffer(self, state, action, reward, new_state, done):
        self._buffer.append([state, action, reward, new_state, done])

    def _n_best_Q(self, arr, n):
        """
        Returns the n largest indices from a numpy array.
        """
        flat = arr.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        indices = np.unravel_index(indices, arr.shape)
        return np.sum(arr[indices])/self._max_rays


if __name__ == "__main__":
    env = gym.make('lidar-v0')

    trials = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    episode = 0

    while True:
        done = False
        cur_state = env.reset()
        # TODO curr_state = Supervised.predict(curr_state)

        while not done:
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step({'rays': action, 'map': cur_state})

            # TODO new_state = Supervised.predict(new_state)
            dqn_agent.append_to_buffer(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break

        episode += 1
