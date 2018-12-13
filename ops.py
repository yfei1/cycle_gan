import tensorflow as tf

KERNEL_SIZE = (3, 3)


def batch_norm(x, name="batch_norm"):
    return tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, name=name)


def instance_norm(input, name="instance_norm"):
    return tf.contrib.layers.instance_norm(input, param_initializer=tf.truncated_normal_initializer(stddev=.02), scope=name)


def group_norm(x, gamma=1, beta=0, G=4, eps=1e-5, name="group_norm"):
    with tf.variable_scope(name):
        N , H, W, C = x.get_shape().as_list()
        # Group Norm is C is divisible by 2, otherwise using layer norm
        G = 2 if C % G == 0 else 1
        x = tf.reshape(x, [-1, H, W, C // G, G])
        mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, shape=[-1, H, W, C])
        return x * gamma + beta


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return tf.layers.conv2d(
                input_, output_dim, ks, strides=s, padding=padding,
                kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                bias_initializer=None
                )


def res_block(input_, output_dim, y=None, name="res_block_", norm=batch_norm, activation=tf.nn.leaky_relu, scaling=None, down_conv=True):
    _, h, w, c = input_.get_shape().as_list()
    
    with tf.variable_scope(name):
        # Residual
        t = input_
        if y is not None:
            _, h1, w1, c1 = t.get_shape().as_list()
            y = tf.image.resize_bilinear(y, (h1, w1))
            t = tf.concat([t, y], axis=-1, name=name + "_cc1")
            
        t = norm(t, name=name+"norm1")
        t = activation(t, name=name+"actv1")

        if scaling == "upsample":
            t = tf.image.resize_bilinear(t, (2 * h, 2 * w))
            input_ = tf.image.resize_bilinear(input_, (2 * h, 2 * w))
        input_ = convblock(input_, output_dim, ks=1, s=1)

        t = conv2d(t, output_dim, ks=3, s=1, name=name+"conv1")
        
        if y is not None:
            _, h1, w1, c1 = t.get_shape().as_list()
            y = tf.image.resize_bilinear(y, (h1, w1))
            t = tf.concat([t, y], axis=-1, name=name+ "_cc2")
            
        t = norm(t, name=name+"norm2")
        t = activation(t, name=name+"actv2")

        if scaling == "downsample" and down_conv:
            t = conv2d(t, output_dim, ks=3, s=2, name=name+"conv2")
            input_ = tf.layers.average_pooling2d(input_, pool_size=2, strides=2, padding='same', name=name+"ap_in1")
        else:
            t = conv2d(t, output_dim, ks=3, s=1, name=name+"conv2")

        if scaling == "downsample" and not down_conv:
            t = tf.layers.average_pooling2d(t, pool_size=2, strides=2, padding='same', name=name+"ap_r1")
            input_ = tf.layers.average_pooling2d(input_, pool_size=2, strides=2, padding='same', name=name+"ap_in1")

        # Shortcut
        return t + input_
    

def resize_deconvblock(input_, output_dim, name="resize_deconvblock", norm=batch_norm, activation=tf.nn.leaky_relu):
    _, h, w, c = input_.get_shape().as_list()

    with tf.variable_scope(name):
        # upsample the image
        input_ = tf.image.resize_bilinear(input_, (2 * h, 2 * w))
        input_ = norm(input_, name=name + "norm")
        input_ = activation(input_, name=name + "actv")
        input_ = conv2d(input_, output_dim, ks=3, s=1, name=name + "conv")

        return input_


def spectral_normalizer(W, u, name="sn"):
    with tf.variable_scope(name):
        v = tf.nn.l2_normalize(tf.matmul(u, W))
        _u = tf.nn.l2_normalize(tf.matmul(v, W, transpose_b=True))
        sigma = tf.matmul(tf.matmul(_u, W), v, transpose_b=True)
        sigma = tf.reduce_sum(sigma)
        return sigma, _u    


def convblock(input_, output_dim, ks=3, s=2, name="convblock", norm=batch_norm, activation=tf.nn.leaky_relu):
    input_ = norm(input_, name=name + "norm")
    input_ = activation(input_, name=name + "actv")
    input_ = conv2d(input_, output_dim, ks=ks, s=s, name=name + "conv")
    return input_


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(
            input_, output_dim, ks,
            strides=s, padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev)
        )


def nonlocalblock(input_, name='nonlocal_', compression=2):
    # input -> (batch_size, h, w, c)
    _, h, w, c = input_.get_shape().as_list()
    intermediate_dim = c // 2

    # theata -> (batch_size, h*w, inter_dim)
    theta = tf.layers.conv2d(input_, intermediate_dim, KERNEL_SIZE, strides=(1, 1), padding='same',
                             kernel_initializer='truncated_normal', name=name+'theta_conv1')
    theta = tf.reshape(theta, shape=(-1, h * w, intermediate_dim))

    # phi -> (batch_size, h*w/2, inter_dim)
    phi = tf.layers.conv2d(input_, intermediate_dim, KERNEL_SIZE, strides=(1, 1), padding='same',
                           kernel_initializer='truncated_normal', name=name+'phi_conv1')
    phi = tf.reshape(phi, shape=(-1, h * w, intermediate_dim))
    phi = tf.layers.max_pooling1d(phi, pool_size=compression, strides=compression, name=name+'phi_maxpool1')

    # f -> (batch_size, h*w, h*w/2)
    f = tf.matmul(theta, phi, transpose_b=True)
    f = tf.nn.softmax(f)

    # g -> (batch_size, h*w/2, inter_dim)
    g = tf.layers.conv2d(input_, intermediate_dim, KERNEL_SIZE, strides=(1, 1), padding='same',
                         kernel_initializer='truncated_normal', name=name+'g_conv1')
    g = tf.reshape(g, shape=(-1, h * w, intermediate_dim))
    g = tf.layers.max_pooling1d(g, pool_size=compression, strides=compression, name=name+'g_maxpool1')

    # out -> (batch_size, h*w, inter_dim)
    out = tf.matmul(f, g)
    out = tf.reshape(out, shape=(-1, h, w, intermediate_dim))
    # out -> (batch_size, h*w, c)
    out = tf.layers.conv2d(out, c, KERNEL_SIZE, strides=(1, 1), padding='same', kernel_initializer='truncated_normal', name=name+'out_conv1')

    # residual connection
    return tf.add(input_, out)


def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)
