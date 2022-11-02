import numpy as np
import tensorflow as tf

BN_EPSILON = 0.001

def Conv2d(x, filters, k_size, stride, name, dilation_rate=(1, 1), padding='same'):
    stddev = tf.sqrt(2 / k_size**2 / filters)
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    #regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    output = tf.layers.conv2d(inputs=x,
                                filters=filters,
                                kernel_size=k_size,
                                strides=stride,
                                dilation_rate=(1, 1),
                                padding=padding,
                                kernel_initializer=initializer,
                                name=name,
                                use_bias=False)
    return output

def BN(x, phase_train, name):
    x_shape = x.get_shape().as_list()
    param_shape = x_shape[-1:]
    batch_mean, batch_var = tf.nn.moments(x, axes=[0 ,1, 2])
    beta = tf.Variable(tf.zeros(param_shape))
    gamma = tf.Variable(tf.ones(param_shape))
    ema = tf.train.ExponentialMovingAverage(decay=0.99)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, BN_EPSILON, name=name)

def Relu(inputs, name):
    return tf.nn.relu(inputs, name=name)

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def max_pooling(inputs, k_size, stride, name, padding='valid'):
    return tf.layers.max_pooling2d(inputs=inputs,
                                    pool_size=k_size,
                                    strides=stride,
                                    padding=padding,
                                    name=name)


def Flatten(inputs, name='flatten'):
    return tf.layers.flatten(inputs, name=name)


def fc_layer(inputs, unit_num, name, use_bias=False):
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.layers.dense(inputs=inputs,
                            units=unit_num,
                            use_bias=use_bias,
                            kernel_initializer=initializer,
                            name=name)


def LRN(x, radius, alpha, beta, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius = radius,
                                                 alpha = alpha, beta = beta,
                                                 bias = bias)
