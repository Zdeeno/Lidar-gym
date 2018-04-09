from __future__ import division, print_function, absolute_import
import numpy as np
import gym
import lidar_gym
import tensorflow as tf
import tflearn
import lidar_gym
from tensorflow.contrib.keras import models
import tensorflow.contrib.keras as keras
import os
from tensorflow.contrib.keras.api.keras.layers import Conv3D, MaxPool3D, Input, Lambda, Conv3DTranspose
from tensorflow.contrib.keras.api.keras.backend import squeeze, expand_dims
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from os.path import expanduser

# Constants
MAP_SIZE = (320, 320, 32)
BATCH_SIZE = 1


def build_network():
    # 3D convolutional network building
    inputs = Input(shape=(MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))
    reshape = Lambda(lambda x: expand_dims(x, -1))(inputs)

    c1 = Conv3D(2, 4, padding='same', kernel_regularizer=l2(0.01), activation='relu')(reshape)
    c2 = Conv3D(4, 4, padding='same', kernel_regularizer=l2(0.01), activation='relu')(c1)
    p1 = MaxPool3D(pool_size=2)(c2)
    c3 = Conv3D(8, 4, padding='same', kernel_regularizer=l2(0.01), activation='relu')(p1)
    p2 = MaxPool3D(pool_size=2)(c3)
    c4 = Conv3D(16, 4, padding='same', kernel_regularizer=l2(0.01), activation='relu')(p2)
    c5 = Conv3D(32, 4, padding='same', kernel_regularizer=l2(0.01), activation='relu')(c4)
    c6 = Conv3D(1, 8, padding='same', kernel_regularizer=l2(0.01), activation='linear')(c5)
    out = Conv3DTranspose(1, 8, strides=[4, 4, 4], padding='same', activation='linear', kernel_regularizer=l2(0.01))(c6)
    outputs = Lambda(lambda x: squeeze(x, 4))(out)

    return Model(inputs, outputs)


model = build_network()
mydir = expanduser("~")
mydir = os.path.join(mydir, 'Projekt/lidar-gym/trained_models/my_keras_model.h5')
model.load_weights(filepath=mydir)
env = gym.make('lidar-v0')
done = False
random_action = env.action_space.sample()
episode = 1
my_input = np.empty((BATCH_SIZE, MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))
env.seed(14)

while True:
    obv = env.reset()
    print('\n------------------- Drive number', episode, '-------------------------')
    epoch = 0
    while not done:

        my_input[0] = obv['X']
        action = model.predict(my_input)
        obv, reward, done, info = env.step(action[0])
        print('reward: ' + str(reward))
        if epoch % 5 == 0:
            env.render()
        epoch = epoch + 1

    episode += 1
    done = False