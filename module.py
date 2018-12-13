from ops import *

gf_dim = 64
df_dim = 64
output_c_dim = 3
is_training = True


def discriminator(image, labels, reuse=False, name="discriminator", norm=batch_norm):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        n1 = nonlocalblock(h0, name='d_nonlocal1_')
        h1 = lrelu(norm(conv2d(n1, df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(norm(conv2d(h1, df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(norm(conv2d(h2, df_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        return h4


def discriminator_condnet(image_A, labels, reuse=False, name="discriminator", norm=batch_norm):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        h1 = res_block(image_A, df_dim, norm=norm, name='d_h1', scaling='downsample')
        n1 = nonlocalblock(h1, name='d_nonlocal1')
        h2 = res_block(n1, df_dim * 2, norm=norm, name='d_h2', scaling='downsample')
        h3 = res_block(h2, df_dim * 4, norm=norm, name='d_h3', scaling='downsample')
        h4 = res_block(h3, df_dim * 8, norm=norm, name='d_h4', scaling='downsample')
        
        h5 = lrelu(h4)
        h6 = tf.reduce_sum(h5, [1, 2])
            
        # initialize variables
        dense_u = tf.get_variable(name="dense_u", shape=(1, df_dim * 8), initializer=tf.initializers.random_normal(), trainable=False)
        dense = tf.layers.Dense(1, use_bias=False, activation=None, kernel_initializer=tf.initializers.random_normal())
        embed_y = tf.get_variable('embeddings', shape=[1, df_dim * 8], initializer=tf.initializers.random_normal(), trainable=True)
        embed_u = tf.get_variable("embed_u", shape=(1, 1), initializer=tf.initializers.random_normal(), trainable=False)    
        # dense
        if not dense.built:
            dense.build(h6.shape)
        sigma, new_u = spectral_normalizer(dense.kernel, dense_u)
            
        with tf.control_dependencies([dense.kernel.assign(dense.kernel / sigma), dense_u.assign(new_u)]):
            output = dense(h6)
            sigma, new_u = spectral_normalizer(embed_y, embed_u)
        with tf.control_dependencies([embed_y.assign(embed_y / sigma), embed_u.assign(new_u)]):
            w_y = tf.nn.embedding_lookup(embed_y, labels)
            w_y = tf.reshape(w_y, (-1, df_dim * 8))
            output += tf.reduce_sum(w_y * h6, axis=1, keepdims=True)
            
        return tf.nn.sigmoid(output)


def generator_condnet(image_A, labels, reuse=False, name="generator", norm=batch_norm):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        c1 = conv2d(image_A, gf_dim, ks=7, s=1, name="g_c1_c1")
        c2 = convblock(c1, gf_dim * 2, ks=3, s=2, name="g_c2_cb1", norm=norm)
        c3 = convblock(c2, gf_dim * 4, ks=3, s=2, name="g_c3_cb2", norm=norm)
            
        r1 = res_block(c3, gf_dim * 4, norm=norm, name='g_r1')
        r2 = res_block(r1, gf_dim * 4, norm=norm, name='g_r2')
        r3 = res_block(r2, gf_dim * 4, norm=norm, name='g_r3')
        r4 = res_block(r3, gf_dim * 4, norm=norm, name='g_r4')
        r5 = res_block(r4, gf_dim * 4, norm=norm, name='g_r5')
        r6 = res_block(r5, gf_dim * 4, norm=norm, name='g_r6')
        r7 = res_block(r6, gf_dim * 4, norm=norm, name='g_r7')
        r8 = res_block(r7, gf_dim * 4, norm=norm, name='g_r8')
        r9 = res_block(r8, gf_dim * 4, norm=norm, name='g_r9')

        n1 = nonlocalblock(r9, name='g_nonlocal1_')
        d1 = res_block(n1, gf_dim * 2, scaling='upsample', norm=norm, name='g_d1_')
        d2 = res_block(d1, gf_dim, scaling='upsample', norm=norm, name='g_d2_')
        d3 = convblock(d2, output_c_dim, ks=7, s=1, name='g_d3_', norm=norm)

        return tf.nn.tanh(d3)


def generator_unet(image, Y, reuse=False, name="generator", norm=batch_norm):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        e1 = norm(conv2d(image, gf_dim, name='g_e1_conv'))
        e2 = norm(conv2d(lrelu(e1), gf_dim * 2, name='g_e2_conv'), 'g_bn_e2')
        e3 = norm(conv2d(lrelu(e2), gf_dim * 4, name='g_e3_conv'), 'g_bn_e3')
        e4 = norm(conv2d(lrelu(e3), gf_dim * 8, name='g_e4_conv'), 'g_bn_e4')
        e5 = norm(conv2d(lrelu(e4), gf_dim * 8, name='g_e5_conv'), 'g_bn_e5')
        e6 = norm(conv2d(lrelu(e5), gf_dim * 8, name='g_e6_conv'), 'g_bn_e6')
        e7 = norm(conv2d(lrelu(e6), gf_dim * 8, name='g_e7_conv'), 'g_bn_e7')

        d1 = deconv2d(lrelu(e7), gf_dim * 8, name='g_d1')
        d1 = tf.nn.dropout(d1, .5)
        d1 = tf.concat([norm(d1, 'g_d1_n'), e6], 3)

        d2 = deconv2d(lrelu(d1), gf_dim * 8, name='g_d2')
        d2 = tf.nn.dropout(d2, .5)
        d2 = tf.concat([norm(d2, 'g_d2_n'), e5], 3)

        d3 = deconv2d(lrelu(d2), gf_dim * 8, name='g_d3')
        d3 = tf.nn.dropout(d3, .5)
        d3 = tf.concat([norm(d3, 'g_d3_n'), e4], 3)

        d4 = deconv2d(lrelu(d3), gf_dim * 8, name='g_d4')
        d4 = tf.concat([norm(d4, 'g_d4_n'), e3], 3)

        d5 = deconv2d(lrelu(d4), gf_dim * 4, name='g_d5')
        d5 = tf.concat([norm(d5, 'g_d5_n'), e2], 3)

        d6 = deconv2d(lrelu(d5), gf_dim * 2, name='g_d6')
        d6 = tf.concat([norm(d6, 'g_d6_n'), e1], 3)

        d7 = deconv2d(lrelu(d6), output_c_dim, name='g_d7')

        return tf.nn.tanh(d7)


def generator_resnet(image, Y, reuse=False, name="generator", norm=batch_norm):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            y = norm(conv2d(x, dim, ks, s, padding='SAME', name=name + '_c1'), name + '_bn1')
            y = norm(conv2d(y, dim, ks, s, padding='SAME', name=name + '_c2'), name + '_bn2')
            return y + x

        c1 = lrelu(norm(conv2d(image, gf_dim    , 7, 1, name='g_c1'), 'g_c1_n'))
        c2 = lrelu(norm(conv2d(c1, gf_dim * 2, 3, 2, name='g_c2'), 'g_c2_n'))
        c3 = lrelu(norm(conv2d(c2, gf_dim * 4, 3, 2, name='g_c3'), 'g_c3_n'))

        r1 = residule_block(c3, gf_dim * 4, name='g_r1')
        r2 = residule_block(r1, gf_dim * 4, name='g_r2')
        r3 = residule_block(r2, gf_dim * 4, name='g_r3')
        r4 = residule_block(r3, gf_dim * 4, name='g_r4')
        r5 = residule_block(r4, gf_dim * 4, name='g_r5')
        r6 = residule_block(r5, gf_dim * 4, name='g_r6')
        r7 = residule_block(r6, gf_dim * 4, name='g_r7')
        r8 = residule_block(r7, gf_dim * 4, name='g_r8')
        r9 = residule_block(r8, gf_dim * 4, name='g_r9')

        d1 = deconv2d(r9, gf_dim * 2, 3, 2, name='g_d1')
        d1 = lrelu(norm(d1, 'g_d1_n'))
        d2 = deconv2d(d1, gf_dim, 3, 2, name='g_d2')
        d2 = lrelu(norm(d2, 'g_d2_n'))
        pred = conv2d(d2, output_c_dim, 7, 1, padding='SAME', name='g_pred')

        return tf.nn.tanh(pred)
