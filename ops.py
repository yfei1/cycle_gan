import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_norm(x, name="batch_norm", ):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


KERNEL_SIZE = (3, 3)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                biases_initializer=None)


def resize_deconvblock(input_, output_dim, name="resize_deconvblock", norm=batch_norm, activation=tf.nn.leaky_relu):
    _, h, w, c = input_.get_shape().as_list()

    with tf.variable_scope(name):
        # upsample the image
        input_ = tf.image.resize_bilinear(input_, (2 * h, 2 * w))
        input_ = norm(input_, name=name + "norm")
        input_ = activation(input_, name=name + "actv")
        input_ = conv2d(input_, output_dim, ks=3, s=1, name=name + "conv")

        return input_


def convblock(input_, output_dim, ks=3, s=2, name="convblock", norm=batch_norm, activation=tf.nn.leaky_relu):
    with tf.variable_scope(name):
        input_ = norm(input_, name=name + "norm")
        input_ = activation(input_, name=name + "actv")
        input_ = conv2d(input_, output_dim, ks=ks, s=s, name=name + "conv")
        return input_


def group_norm():
    pass


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)


def nonlocalblock(input_, compression=2):
    # input -> (batch_size, h, w, c)
    _, h, w, c = input_.get_shape().as_list()
    intermediate_dim = c // 2

    # theata -> (batch_size, h*w, inter_dim)
    theta = tf.layers.conv2d(input_, intermediate_dim, KERNEL_SIZE, strides=(1, 1), padding='same',
                             kernel_initializer='truncated_normal')
    theta = tf.reshape(theta, shape=(-1, h * w, intermediate_dim))

    # phi -> (batch_size, h*w/2, inter_dim)
    phi = tf.layers.conv2d(input_, intermediate_dim, KERNEL_SIZE, strides=(1, 1), padding='same',
                           kernel_initializer='truncated_normal')
    phi = tf.reshape(phi, shape=(-1, h * w, intermediate_dim))
    phi = tf.layers.max_pooling1d(phi, pool_size=compression, strides=compression)

    # f -> (batch_size, h*w, h*w/2)
    f = tf.matmul(theta, phi, transpose_b=True)
    f = tf.nn.softmax(f)

    # g -> (batch_size, h*w/2, inter_dim)
    g = tf.layers.conv2d(input_, intermediate_dim, KERNEL_SIZE, strides=(1, 1), padding='same',
                         kernel_initializer='truncated_normal')
    g = tf.reshape(g, shape=(-1, h * w, intermediate_dim))
    g = tf.layers.max_pooling1d(g, pool_size=compression, strides=compression)

    # out -> (batch_size, h*w, inter_dim)
    out = tf.matmul(f, g)
    out = tf.reshape(out, shape=(-1, h, w, intermediate_dim))
    # out -> (batch_size, h*w, c)
    out = tf.layers.conv2d(out, c, KERNEL_SIZE, strides=(1, 1), padding='same', kernel_initializer='truncated_normal')

    # residual connection
    return tf.add(input_, out)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
