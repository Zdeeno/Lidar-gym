import gym
import numpy as np
import random
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Input, Lambda, Conv3D, MaxPool3D, Conv2D,\
                                                      MaxPool2D, Reshape
from tensorflow.contrib.keras.api.keras.backend import squeeze, expand_dims, reshape
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.layers import Add, Multiply
from tensorflow.contrib.keras.api.keras.optimizers import Adam
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.contrib.keras.api.keras.activations import softmax

from collections import deque


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.map_shape = (320, 320, 32)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        state_input = Input(shape=self.map_shape)
        r1 = Lambda(lambda x: expand_dims(x, -1))(state_input)

        c1 = Conv3D(2, 4, padding='same', kernel_regularizer=l2(0.01), activation='relu')(r1)
        p1 = MaxPool3D(pool_size=2)(c1)
        c2 = Conv3D(4, 4, padding='same', kernel_regularizer=l2(0.01), activation='relu')(p1)
        c3 = Conv3D(1, 8, padding='same', kernel_regularizer=l2(0.01), activation='relu')(c2)
        s1 = Lambda(lambda x: squeeze(x, 4))(c3)
        c4 = Conv2D(3, 4, padding='same', kernel_regularizer=l2(0.01), activation='relu')(s1)
        r2 = Reshape((480, 480, 1))(c4)
        p2 = MaxPool2D(pool_size=(3, 4))(r2)
        c5 = Conv2D(2, 4, padding='same', kernel_regularizer=l2(0.01), activation='linear')(p2)
        output = Conv2D(1, 4, padding='same', kernel_regularizer=l2(0.01), activation='softmax')(c5)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    gamma = 0.9
    epsilon = .95

    trials = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape(1, 2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break