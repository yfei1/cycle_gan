from ops import *

gf_dim = 16
df_dim = 16
output_c_dim = 3
is_training = True

def discriminator(image, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, df_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def generator_condnet(image_A, image_B, reuse=False, name="generator", resize_deconv=True):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
            return y + x

        def cond_residule_block(x, y, name="cond_res"):
            _, h, w, c = x.get_shape().as_list()
            y = tf.image.resize_bilinear(y, (h, w))

            tmp = x
            tmp = tf.concat([tmp, y], axis=-1, name=name + "_cc1")
            tmp = convblock(tmp, c, ks=3, s=1, norm=batch_norm, activation=tf.nn.relu, name=name + "_cb1")
            tmp = tf.concat([tmp, y], axis=-1, name=name + "_cc2")
            tmp = convblock(tmp, c, ks=3, s=1, norm=batch_norm, activation=tf.nn.relu, name=name + "_cb2")
            return x + tmp


        c1 = conv2d(image_A, gf_dim, ks=7, s=1, name="g_c1_c1")
        c2 = convblock(c1, gf_dim * 2, ks=3, s=2, name="g_c2_cb1")
        c3 = convblock(c2, gf_dim * 4, ks=3, s=2, name="g_c3_cb2")

        r1 = residule_block(c3, gf_dim * 4, name='g_r1')
        r2 = residule_block(r1, gf_dim * 4, name='g_r2')
        r3 = residule_block(r2, gf_dim * 4, name='g_r3')
        r4 = residule_block(r3, gf_dim * 4, name='g_r4')
        r5 = residule_block(r4, gf_dim * 4, name='g_r5')
        r6 = residule_block(r5, gf_dim * 4, name='g_r6')
        r7 = residule_block(r6, gf_dim * 4, name='g_r7')
        r8 = residule_block(r7, gf_dim * 4, name='g_r8')
        r9 = residule_block(r8, gf_dim * 4, name='g_r9')

        if resize_deconv:
            d1 = resize_deconvblock(r9, gf_dim * 2, name="g_d1_")
            x1 = cond_residule_block(d1, image_B, name="g_x1_")
            d2 = resize_deconvblock(x1, gf_dim, name="g_d2_")
            x2 = cond_residule_block(d2, image_B, name="g_x2_")
            d3 = convblock(x2, output_c_dim, ks=7, s=1, name="g_d3_")
        else:
            d1 = deconv2d(r9, gf_dim * 2, 3, 2, name='g_d1_')
            x1 = cond_residule_block(d1, image_B, name="g_x1_")
            d2 = deconv2d(x1, gf_dim, 3, 2, name='g_d2_')
            x2 = cond_residule_block(d2, image_B, name="g_x2_")
            d3 = convblock(x2, output_c_dim, ks=7, s=1, name="g_d3_")

        return tf.nn.tanh(d3)


def generator_unet(image, reuse=False, name="generator"):
    dropout_rate = 0.5 if is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), gf_dim * 2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), gf_dim * 4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), gf_dim * 8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), gf_dim * 8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), gf_dim * 8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), gf_dim * 8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), gf_dim * 8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), gf_dim * 8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), gf_dim * 8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), gf_dim * 8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), gf_dim * 8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), gf_dim * 4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), gf_dim * 2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def generator_resnet(image, Y, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, gf_dim * 4, name='g_r1')
        r2 = residule_block(r1, gf_dim * 4, name='g_r2')
        r3 = residule_block(r2, gf_dim * 4, name='g_r3')
        r4 = residule_block(r3, gf_dim * 4, name='g_r4')
        r5 = residule_block(r4, gf_dim * 4, name='g_r5')
        r6 = residule_block(r5, gf_dim * 4, name='g_r6')
        r7 = residule_block(r6, gf_dim * 4, name='g_r7')
        r8 = residule_block(r7, gf_dim * 4, name='g_r8')
        r9 = residule_block(r8, gf_dim * 4, name='g_r9')

        d1 = deconv2d(r9, gf_dim * 2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
