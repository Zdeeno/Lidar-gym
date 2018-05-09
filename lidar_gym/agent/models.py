from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Input, Lambda, Conv3D, MaxPool3D, Conv2D,\
                                                      MaxPool2D, Reshape, Flatten, Layer, Add, Conv3DTranspose
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.optimizers import Adam
import tensorflow.contrib.keras.api.keras.backend as K
import tensorflow as tf

import random
from collections import deque
from os.path import expanduser
import os


def logistic_loss(y_true, y_pred):
    # own objective function
    bigger = tf.cast(tf.greater(y_true, 0.0), tf.float32)
    smaller = tf.cast(tf.greater(0.0, y_true), tf.float32)

    weights_positive = 0.5 / tf.reduce_sum(bigger)
    weights_negative = 0.5 / tf.reduce_sum(smaller)

    weights = bigger * weights_positive + smaller * weights_negative

    # Here often occurs numeric instability -> nan or inf
    # return tf.reduce_sum(weights * (tf.log(1 + tf.exp(-y_pred * y_true))))
    a = -y_pred * y_true
    b = tf.maximum(0.0, a)
    t = b + tf.log(tf.exp(-b) + tf.exp(a - b))
    return tf.reduce_sum(weights * t)


# -------------------------------- TOY SIZE MODELS ------------------------------


def create_toy_actor_model(lr, map_shape):
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu')(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu')(p11)
    c31 = Conv3D(8, 4, padding='same', activation='linear')(c21)

    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu')(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 4, padding='same', activation='relu')(p12)
    c32 = Conv3D(8, 4, padding='same', activation='linear')(c22)

    # merge SMALL inputs
    a1 = Add()([c31, c32])
    c1 = Conv3D(1, 4, padding='same', activation='relu')(a1)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c1)
    c2 = Conv2D(8, 4, padding='same', activation='relu')(s1)
    p1 = MaxPool2D(pool_size=2)(c2)
    c3 = Conv2D(3, 4, padding='same', activation='relu')(p1)
    r2 = Reshape((40, 30, 1))(c3)
    c5 = Conv2D(4, 4, padding='same', activation='relu')(r2)
    c6 = Conv2D(16, 4, padding='same', activation='linear', kernel_regularizer='l2')(c5)
    c7 = Conv2D(1, 4, padding='same', activation='linear', kernel_regularizer='l2')(c6)
    sigmoid = Lambda(lambda x: x / (tf.abs(x) + 50))(c7)
    bias = Lambda(lambda x: (x + 1) / 2)(sigmoid)

    output = Lambda(lambda x: K.squeeze(x, 3))(bias)

    ret_model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
    adam = Adam(lr=lr)
    ret_model.compile(loss='mse', optimizer=adam)
    return sparse_input, reconstructed_input, ret_model


def create_toy_critic_model(lr, map_shape, lidar_shape):
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu')(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu')(p11)
    c31 = Conv3D(8, 4, padding='same', activation='linear')(c21)

    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu')(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 4, padding='same', activation='relu')(p12)
    c32 = Conv3D(8, 4, padding='same', activation='linear')(c22)

    action_input = Input(shape=lidar_shape)
    r13 = Lambda(lambda x: K.expand_dims(x, -1))(action_input)
    c13 = Conv2D(4, 4, padding='same', activation='relu')(r13)
    c23 = Conv2D(8, 4, padding='same', activation='relu')(c13)
    c33 = Conv2D(1, 4, padding='same', activation='linear')(c23)

    # merge SMALL action inputs and output action Q value
    a1 = Add()([c31, c32])
    c1 = Conv3D(1, 4, padding='same', activation='relu')(a1)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c1)
    c2 = Conv2D(8, 4, padding='same', activation='relu')(s1)
    p1 = MaxPool2D(pool_size=2)(c2)
    c3 = Conv2D(3, 4, padding='same', activation='linear')(p1)
    r2 = Reshape((40, 30, 1))(c3)
    a2 = Add()([r2, c33])
    c4 = Conv2D(2, 4, padding='same', activation='relu')(a2)
    p2 = MaxPool2D(pool_size=2)(c4)
    c5 = Conv2D(4, 4, padding='same', activation='relu')(p2)
    p3 = MaxPool2D(pool_size=2)(c5)
    c6 = Conv2D(1, 4, padding='same', activation='relu')(p3)
    f1 = Flatten()(c6)
    d1 = Dense(5, activation='linear')(f1)
    output = Dense(1, activation='linear')(d1)

    ret_model = Model(inputs=[sparse_input, reconstructed_input, action_input], outputs=output)

    adam = Adam(lr=lr)
    ret_model.compile(loss="mse", optimizer=adam)
    return sparse_input, reconstructed_input, action_input, ret_model


def create_toy_dqn_model(lr, map_shape):
    # TOY model version
    # reconstructed input
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p11)

    # sparse input
    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 2, padding='same', activation='relu', kernel_regularizer='l2')(p12)

    # merge SMALL inputs
    a1 = Add()([c21, c22])
    c1 = Conv3D(1, 4, padding='same', activation='relu', kernel_regularizer='l2')(a1)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c1)
    c2 = Conv2D(8, 4, padding='same', activation='relu', kernel_regularizer='l2')(s1)
    p1 = MaxPool2D(pool_size=2)(c2)
    c3 = Conv2D(3, 4, padding='same', activation='relu', kernel_regularizer='l2')(p1)
    r1 = Reshape((40, 30, 1))(c3)
    c5 = Conv2D(4, 4, padding='same', activation='relu')(r1)
    c6 = Conv2D(8, 4, padding='same', activation='linear')(c5)
    c7 = Conv2D(1, 4, padding='same', activation='linear')(c6)
    output = Lambda(lambda x: K.squeeze(x, 3))(c7)

    model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
    adam = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model


def create_toy_supervised_model(lr, map_shape):
    # 3D TOY-convolutional network building
    inputs = Input(shape=(map_shape[0], map_shape[1], map_shape[2]))
    reshape = Lambda(lambda x: K.expand_dims(x, -1))(inputs)

    c1 = Conv3D(2, 2, padding='same', activation='relu')(reshape)
    c2 = Conv3D(4, 4, padding='same', activation='relu')(c1)
    p1 = MaxPool3D(pool_size=2)(c2)
    c3 = Conv3D(8, 4, padding='same', activation='relu')(p1)
    p2 = MaxPool3D(pool_size=2)(c3)
    c4 = Conv3D(16, 2, padding='same', activation='relu')(p2)
    c5 = Conv3D(32, 2, padding='same', activation='relu')(c4)
    c6 = Conv3D(1, 2, padding='same', activation='linear')(c5)
    out = Conv3DTranspose(1, 4, strides=[4, 4, 4], padding='same', activation='linear')(c6)
    outputs = Lambda(lambda x: K.squeeze(x, 4))(out)
    opt = Adam(lr=lr)
    model = Model(inputs, outputs)
    model.compile(optimizer=opt, loss=logistic_loss)
    return model


# -------------------------------- SMALL SIZE MODELS ----------------------------


def create_small_actor_model(lr, map_shape):
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu', )(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu', )(p11)
    c31 = Conv3D(8, 4, padding='same', activation='linear', )(c21)

    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu', )(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 4, padding='same', activation='relu', )(p12)
    c32 = Conv3D(8, 4, padding='same', activation='linear', )(c22)

    # merge SMALL inputs
    a1 = Add()([c31, c32])
    c1 = Conv3D(1, 4, padding='same', activation='relu')(a1)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c1)
    c2 = Conv2D(8, 4, padding='same', activation='relu')(s1)
    p1 = MaxPool2D(pool_size=2)(c2)
    c3 = Conv2D(81, 4, padding='same', activation='relu')(p1)
    r2 = Reshape((360, 360, 1))(c3)
    p2 = MaxPool2D(pool_size=(3, 4))(r2)
    c5 = Conv2D(4, 4, padding='same', activation='relu')(p2)
    c6 = Conv2D(16, 4, padding='same', activation='linear')(c5)
    c7 = Conv2D(1, 4, padding='same', activation='linear')(c6)
    sigmoid = Lambda(lambda x: x/(tf.abs(x)+50))(c7)
    bias = Lambda(lambda x: (x+1)/2)(sigmoid)

    output = Lambda(lambda x: K.squeeze(x, 3))(bias)

    ret_model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
    adam = Adam(lr=lr)
    ret_model.compile(loss='mse', optimizer=adam)
    return sparse_input, reconstructed_input, ret_model


def create_small_critic_model(lr, map_shape, lidar_shape):
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu', )(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu', )(p11)
    c31 = Conv3D(8, 4, padding='same', activation='linear', )(c21)

    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu', )(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 4, padding='same', activation='relu', )(p12)
    c32 = Conv3D(8, 4, padding='same', activation='linear', )(c22)

    action_input = Input(shape=lidar_shape)
    r13 = Lambda(lambda x: K.expand_dims(x, -1))(action_input)
    c13 = Conv2D(4, 4, padding='same', activation='relu', )(r13)
    c23 = Conv2D(8, 4, padding='same', activation='relu', )(c13)
    c33 = Conv2D(8, 4, padding='same', activation='linear', )(c23)

    # merge SMALL action inputs and output action Q value
    a1 = Add()([c31, c32])
    c1 = Conv3D(1, 4, padding='same', activation='relu', )(a1)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c1)
    c2 = Conv2D(8, 4, padding='same', activation='relu', )(s1)
    p1 = MaxPool2D(pool_size=2)(c2)
    c3 = Conv2D(81, 4, padding='same', activation='relu', )(p1)
    r2 = Reshape((360, 360, 1))(c3)
    p2 = MaxPool2D(pool_size=(3, 4))(r2)
    a2 = Add()([p2, c33])
    c4 = Conv2D(2, 4, padding='same', activation='relu', )(a2)
    p2 = MaxPool2D(pool_size=4, strides=4)(c4)
    c5 = Conv2D(4, 4, padding='same', activation='relu', )(p2)
    p3 = MaxPool2D(pool_size=4, strides=4)(c5)
    c6 = Conv2D(1, 4, padding='same', activation='relu', )(p3)
    f1 = Flatten()(c6)
    d1 = Dense(5, activation='linear')(f1)
    output = Dense(1, activation='linear')(d1)

    ret_model = Model(inputs=[sparse_input, reconstructed_input, action_input], outputs=output)

    adam = Adam(lr=lr)
    ret_model.compile(loss="mse", optimizer=adam)
    return sparse_input, reconstructed_input, action_input, ret_model


def create_small_dqn_model(lr, map_shape):
    # reconstructed input
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p11)

    # sparse input
    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p12)

    # merge SMALL inputs
    a1 = Add()([c21, c22])
    c1 = Conv3D(1, 4, padding='same', activation='relu', kernel_regularizer='l2')(a1)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c1)
    c2 = Conv2D(8, 4, padding='same', activation='relu', kernel_regularizer='l2')(s1)
    p1 = MaxPool2D(pool_size=2)(c2)
    c3 = Conv2D(81, 4, padding='same', activation='relu', kernel_regularizer='l2')(p1)
    r2 = Reshape((360, 360, 1))(c3)
    p2 = MaxPool2D(pool_size=(3, 4))(r2)
    c5 = Conv2D(4, 4, padding='same', activation='relu')(p2)
    c6 = Conv2D(8, 4, padding='same', activation='linear')(c5)
    c7 = Conv2D(1, 4, padding='same', activation='linear')(c6)
    output = Lambda(lambda x: K.squeeze(x, 3))(c7)

    model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
    adam = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model


def create_supervised_model(lr, map_shape):
    # suitable for large and small environment
    inputs = Input(shape=(map_shape[0], map_shape[1], map_shape[2]))
    reshape = Lambda(lambda x: K.expand_dims(x, -1))(inputs)

    c1 = Conv3D(2, 4, padding='same', kernel_regularizer='l2', activation='relu')(reshape)
    c2 = Conv3D(4, 4, padding='same', kernel_regularizer='l2', activation='relu')(c1)
    p1 = MaxPool3D(pool_size=2)(c2)
    c3 = Conv3D(8, 4, padding='same', kernel_regularizer='l2', activation='relu')(p1)
    p2 = MaxPool3D(pool_size=2)(c3)
    c4 = Conv3D(16, 4, padding='same', kernel_regularizer='l2', activation='relu')(p2)
    c5 = Conv3D(32, 4, padding='same', kernel_regularizer='l2', activation='relu')(c4)
    c6 = Conv3D(1, 4, padding='same', kernel_regularizer='l2', activation='linear')(c5)
    out = Conv3DTranspose(1, 4, strides=[4, 4, 4], padding='same', activation='linear',
                          kernel_regularizer='l2')(c6)
    outputs = Lambda(lambda x: K.squeeze(x, 4))(out)
    opt = Adam(lr=lr)
    model = Model(inputs, outputs)
    model.compile(optimizer=opt, loss=logistic_loss)
    return model


# ------------------------------------ LARGE SIZE MODELS ----------------------------------------


def create_large_actor_model(lr, map_shape):
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu', )(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu', )(p11)
    c31 = Conv3D(8, 4, padding='same', activation='linear', )(c21)

    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu', )(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 4, padding='same', activation='relu', )(p12)
    c32 = Conv3D(8, 4, padding='same', activation='linear', )(c22)

    # merge LARGE inputs
    c2 = Add()([c31, c32])
    c3 = Conv3D(1, 4, padding='same', activation='relu')(c2)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c3)
    c4 = Conv2D(9, 4, padding='same', activation='relu')(s1)
    r2 = Reshape((480, 480, 1))(c4)
    p2 = MaxPool2D(pool_size=(3, 4))(r2)
    c5 = Conv2D(2, 4, padding='same', activation='linear')(p2)
    c6 = Conv2D(1, 4, padding='same', activation='linear')(c5)
    c7 = Conv2D(1, 4, padding='same', activation='linear')(c6)
    sigmoid = Lambda(lambda x: x / (tf.abs(x) + 50))(c7)
    bias = Lambda(lambda x: (x + 1) / 2)(sigmoid)
    output = Lambda(lambda x: K.squeeze(x, 3))(bias)

    ret_model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
    adam = Adam(lr=lr)
    ret_model.compile(loss='mse', optimizer=adam)
    return sparse_input, reconstructed_input, ret_model


def create_large_critic_model(lr, map_shape, lidar_shape):
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu', )(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu', )(p11)
    c31 = Conv3D(8, 4, padding='same', activation='linear', )(c21)

    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu', )(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 4, padding='same', activation='relu', )(p12)
    c32 = Conv3D(8, 4, padding='same', activation='linear', )(c22)

    action_input = Input(shape=lidar_shape)
    r13 = Lambda(lambda x: K.expand_dims(x, -1))(action_input)
    c13 = Conv2D(4, 4, padding='same', activation='relu', )(r13)
    c23 = Conv2D(8, 4, padding='same', activation='relu', )(c13)
    c33 = Conv2D(8, 4, padding='same', activation='linear', )(c23)

    # merge LARGE action inputs and output reward
    a1 = Add()([c31, c32])
    c1 = Conv3D(1, 4, padding='same', activation='relu')(a1)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c1)
    c2 = Conv2D(9, 4, padding='same', activation='relu')(s1)
    r1 = Reshape((480, 480, 1))(c2)
    p1 = MaxPool2D(pool_size=(3, 4))(r1)
    a2 = Add()([p1, c33])
    c3 = Conv2D(2, 4, padding='same', activation='relu')(a2)
    p2 = MaxPool2D(pool_size=4, strides=4)(c3)
    c4 = Conv2D(4, 4, padding='same', activation='relu')(p2)
    p3 = MaxPool2D(pool_size=4, strides=4)(c4)
    c5 = Conv2D(1, 4, padding='same', activation='relu')(p3)
    f1 = Flatten()(c5)
    output = Dense(1, activation='linear')(f1)

    ret_model = Model(inputs=[sparse_input, reconstructed_input, action_input], outputs=output)

    adam = Adam(lr=lr)
    ret_model.compile(loss="mse", optimizer=adam)
    return sparse_input, reconstructed_input, action_input, ret_model


def create_large_dqn_model(lr, map_shape):
    # reconstructed input
    reconstructed_input = Input(shape=map_shape)
    r11 = Lambda(lambda x: K.expand_dims(x, -1))(reconstructed_input)
    c11 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r11)
    p11 = MaxPool3D(pool_size=2)(c11)
    c21 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p11)

    # sparse input
    sparse_input = Input(shape=map_shape)
    r12 = Lambda(lambda x: K.expand_dims(x, -1))(sparse_input)
    c12 = Conv3D(2, 4, padding='same', activation='relu', kernel_regularizer='l2')(r12)
    p12 = MaxPool3D(pool_size=2)(c12)
    c22 = Conv3D(4, 4, padding='same', activation='relu', kernel_regularizer='l2')(p12)

    # merge LARGE inputs
    c2 = Add()([c21, c22])
    c3 = Conv3D(1, 4, padding='same', activation='relu')(c2)
    s1 = Lambda(lambda x: K.squeeze(x, 4))(c3)
    c4 = Conv2D(9, 4, padding='same', activation='relu')(s1)
    r2 = Reshape((480, 480, 1))(c4)
    p2 = MaxPool2D(pool_size=(3, 4))(r2)
    c5 = Conv2D(2, 4, padding='same', activation='relu')(p2)
    c6 = Conv2D(1, 4, padding='same', activation='linear')(c5)
    output = Lambda(lambda x: K.squeeze(x, 3))(c6)

    model = Model(inputs=[sparse_input, reconstructed_input], outputs=output)
    adam = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model