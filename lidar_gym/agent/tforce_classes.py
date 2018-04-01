from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from math import sqrt
import numpy as np
from math import log

import tensorforce.core.networks as net
import tensorflow as tf
from tensorforce import TensorForceError, util
import tensorforce.core.distributions as dists


class MyBeta(dists.Distribution):
    """
    Beta distribution, for bounded continuous actions.
    """

    def __init__(self, shape, min_value, max_value, alpha=0.0, beta=0.0, scope='mybeta', summary_labels=()):
        """
        Beta distribution.
        Args:
            shape: Action shape.
            min_value: Minimum value of continuous actions.
            max_value: Maximum value of continuous actions.
            alpha: Optional distribution bias for the alpha value.
            beta: Optional distribution bias for the beta value.
        """

        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
        action_size = util.prod(self.shape)

        self.alpha = Conv3dTranspose(size=1, stride=4, window=8, activation='none', bias=alpha, scope='alpha')
        self.beta = Conv3dTranspose(size=1, stride=4, window=8, activation='none', bias=beta, scope='beta')

        super(MyBeta, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_parameterize(self, x):
        # Softplus to ensure alpha and beta >= 1
        # epsilon < 1.0, hence negative
        log_eps = log(util.epsilon)

        alpha = self.alpha.apply(x=x)
        alpha = tf.clip_by_value(t=alpha, clip_value_min=log_eps, clip_value_max=-log_eps)
        alpha = tf.log(x=(tf.exp(x=alpha) + 1.0)) + 1.0

        beta = self.beta.apply(x=x)
        beta = tf.clip_by_value(t=beta, clip_value_min=log_eps, clip_value_max=-log_eps)
        beta = tf.log(x=(tf.exp(x=beta) + 1.0)) + 1.0

        shape = (-1,) + self.shape
        alpha = tf.reshape(tensor=alpha, shape=shape)
        beta = tf.reshape(tensor=beta, shape=shape)

        alpha_beta = tf.maximum(x=(alpha + beta), y=util.epsilon)
        log_norm = tf.lgamma(x=alpha) + tf.lgamma(x=beta) - tf.lgamma(x=alpha_beta)

        return alpha, beta, alpha_beta, log_norm

    def tf_sample(self, distr_params, deterministic):
        alpha, beta, alpha_beta, _ = distr_params

        # Deterministic: mean as action
        definite = beta / alpha_beta

        # Non-deterministic: sample action using gamma distribution
        alpha_sample = tf.random_gamma(shape=(), alpha=alpha)
        beta_sample = tf.random_gamma(shape=(), alpha=beta)

        sampled = beta_sample / tf.maximum(x=(alpha_sample + beta_sample), y=util.epsilon)

        return self.min_value + (self.max_value - self.min_value) * \
            tf.where(condition=deterministic, x=definite, y=sampled)

    def tf_log_probability(self, distr_params, action):
        alpha, beta, _, log_norm = distr_params
        action = (action - self.min_value) / (self.max_value - self.min_value)
        action = tf.minimum(x=action, y=(1.0 - util.epsilon))
        return (beta - 1.0) * tf.log(x=tf.maximum(x=action, y=util.epsilon)) + \
            (alpha - 1.0) * tf.log1p(x=-action) - log_norm

    def tf_entropy(self, distr_params):
        alpha, beta, alpha_beta, log_norm = distr_params
        return log_norm - (beta - 1.0) * tf.digamma(x=beta) - (alpha - 1.0) * tf.digamma(x=alpha) + \
            (alpha_beta - 2.0) * tf.digamma(x=alpha_beta)

    def tf_kl_divergence(self, distr_params1, distr_params2):
        alpha1, beta1, alpha_beta1, log_norm1 = distr_params1
        alpha2, beta2, alpha_beta2, log_norm2 = distr_params2
        return log_norm2 - log_norm1 - tf.digamma(x=beta1) * (beta2 - beta1) - \
            tf.digamma(x=alpha1) * (alpha2 - alpha1) + tf.digamma(x=alpha_beta1) * (alpha_beta2 - alpha_beta1)

    def tf_regularization_loss(self):
        regularization_loss = super(MyBeta, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.alpha.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        regularization_loss = self.beta.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        distribution_variables = super(MyBeta, self).get_variables(include_nontrainable=include_nontrainable)
        alpha_variables = self.alpha.get_variables(include_nontrainable=include_nontrainable)
        beta_variables = self.beta.get_variables(include_nontrainable=include_nontrainable)

        return distribution_variables + alpha_variables + beta_variables

    def get_summaries(self):
        distribution_summaries = super(MyBeta, self).get_summaries()
        alpha_summaries = self.alpha.get_summaries()
        beta_summaries = self.beta.get_summaries()

        return distribution_summaries + alpha_summaries + beta_summaries


class Conv3d(net.Layer):
    """
    3-dimensional convolutional layer.
    """

    def __init__(
        self,
        size,
        window=3,
        stride=1,
        padding='SAME',
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='conv3d',
        summary_labels=()
    ):
        """
        3D convolutional layer.
        Args:
            size: Number of filters
            window: Convolution window size, either an integer or pair of integers.
            stride: Convolution stride, either an integer or pair of integers.
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity, or dict with name & arguments
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = size
        if isinstance(window, int):
            self.window = (window, window, window)
        elif len(window) == 3:
            self.window = tuple(window)
        else:
            raise TensorForceError('Invalid window {} for conv2d layer, must be of size 2'.format(window))
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = net.Nonlinearity(summary_labels=summary_labels, **util.prepare_kwargs(activation))
        super(Conv3d, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update=False):
        if util.rank(x) != 5:
            raise TensorForceError('Invalid input rank for conv2d layer: {}, must be 5'.format(util.rank(x)))

        filters_shape = self.window + (x.shape[4].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        stride_h, stride_w, stride_d = self.stride if type(self.stride) is tuple else (self.stride, self.stride, self.stride)
        x = tf.nn.conv3d(input=x, filter=self.filters, strides=(1, stride_h, stride_w, stride_d, 1), padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Conv3d, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        layer_variables = super(Conv3d, self).get_variables(include_nontrainable=include_nontrainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_nontrainable=include_nontrainable)

        return layer_variables + nonlinearity_variables

    def get_summaries(self):
        layer_summaries = super(Conv3d, self).get_summaries()
        nonlinearity_summaries = self.nonlinearity.get_summaries()

        return layer_summaries + nonlinearity_summaries


class Pool3d(net.Layer):
    """
    3-dimensional pooling layer.
    """

    def __init__(
        self,
        pooling_type='max',
        window=2,
        stride=2,
        padding='SAME',
        scope='pool3d',
        summary_labels=()
    ):
        """
        2-dimensional pooling layer.
        Args:
            pooling_type: Either 'max' or 'average'.
            window: Pooling window size, either an integer or pair of integers.
            stride: Pooling stride, either an integer or pair of integers.
            padding: Pooling padding, one of 'VALID' or 'SAME'.
        """
        self.pooling_type = pooling_type
        if isinstance(window, int):
            self.window = (1, window, window, window, 1)
        elif len(window) == 3:
            self.window = (1, window[0], window[1], window[2], 1)
        else:
            raise TensorForceError('Invalid window {} for pool2d layer, must be of size 2'.format(window))
        if isinstance(stride, int):
            self.stride = (1, stride, stride, stride, 1)
        elif len(window) == 3:
            self.stride = (1, stride[0], stride[1], stride[2], 1)
        else:
            raise TensorForceError('Invalid stride {} for pool2d layer, must be of size 2'.format(stride))
        self.padding = padding
        super(Pool3d, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update=False):
        if self.pooling_type == 'average':
            x = tf.nn.avg_pool3d(value=x, ksize=self.window, strides=self.stride, padding=self.padding)

        elif self.pooling_type == 'max':
            x = tf.nn.max_pool3d(value=x, ksize=self.window, strides=self.stride, padding=self.padding)

        else:
            raise TensorForceError('Invalid pooling type: {}'.format(self.name))

        return x


class Conv3dTranspose(net.Layer):
    """
    3-dimensional convolutional layer.
    """

    def __init__(
        self,
        size,
        window=3,
        stride=1,
        padding='SAME',
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='conv3d_transpose',
        summary_labels=()
    ):
        """
        3D deconvolutional layer.
        Args:
            size: Number of filters
            window: Convolution window size, either an integer or pair of integers.
            stride: Convolution stride, either an integer or pair of integers.
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity, or dict with name & arguments
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = size
        if isinstance(window, int):
            self.window = (window, window, window)
        elif len(window) == 3:
            self.window = tuple(window)
        else:
            raise TensorForceError('Invalid window {} for conv2d layer, must be of size 2'.format(window))
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = net.Nonlinearity(summary_labels=summary_labels, **util.prepare_kwargs(activation))
        super(Conv3dTranspose, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update=False):
        if util.rank(x) != 5:
            raise TensorForceError('Invalid input rank for conv2d layer: {}, must be 5'.format(util.rank(x)))

        filters_shape = self.window + (x.shape[4].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        stride_h, stride_w, stride_d = self.stride if type(self.stride) is tuple else (self.stride, self.stride, self.stride)
        out_shape = (1, x.shape[1].value*stride_h, x.shape[2].value*stride_w, x.shape[3].value*stride_d, 1)
        x = tf.nn.conv3d_transpose(value=x, filter=self.filters, strides=(1, stride_h, stride_w, stride_d, 1),
                                   output_shape=out_shape, padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Conv3dTranspose, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        layer_variables = super(Conv3dTranspose, self).get_variables(include_nontrainable=include_nontrainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_nontrainable=include_nontrainable)

        return layer_variables + nonlinearity_variables

    def get_summaries(self):
        layer_summaries = super(Conv3dTranspose, self).get_summaries()
        nonlinearity_summaries = self.nonlinearity.get_summaries()

        return layer_summaries + nonlinearity_summaries