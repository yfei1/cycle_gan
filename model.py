import tensorflow as tf
import numpy as 
import os
from scipy.misc import imsave
import tensorflow.contrib.gan as gan

# from __future__ import division
# import time
# from glob import glob
# from collections import namedtuple

from module import *
from utils import *


BATCH_SIZE = 1
INPUT_WIDTH = 256
INPUT_DIM = 3
X_PATH = './datasets/monet2photo/trainA'
Y_PATH = './datasets/monet2photo/trainB'
log_every = 10
save_every = 10
L1_lambda = 10


class Model(object):
    def __init__(self):
        self.discriminator = discriminator
        self.generator = generator
        # self.criterionGAN = mae_criterion

        self.X = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.Y = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.X2Y_sample = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.Y2X_sample = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])

        self.X2Y = self.generator(self.X, self.options, False, name="generatorX2Y")
        self.X2Y2X = self.generator(self.X2Y, self.options, False, name="generatorY2X")
        self.Y2X = self.generator(self.Y, self.options, True, name="generatorY2X")
        self.Y2X2Y = self.generator(self.Y2X, self.options, True, name="generatorX2Y")

        with tf.variable_scope('discriminatorY'):
            self.d_X2Y = self.discriminator(self.X2Y, self.options, reuse=False)
        with tf.variable_scope('discriminatorX'):
            self.d_Y2X = self.discriminator(self.Y2X, self.options, reuse=False)

        with tf.variable_scope('discriminatorY'):
            self.d_Y = self.discriminator(self.Y, self.options, reuse=True)
        with tf.variable_scope('discriminatorX'):
            self.d_X = self.discriminator(self.X, self.options, reuse=True)
        with tf.variable_scope('discriminatorY'):
            self.d_Y2X_sample = self.discriminator(self.Y2X_sample, self.options, reuse=True)
        with tf.variable_scope('discriminatorX'):
            self.d_X2Y_sample = self.discriminator(self.X2Y_sample, self.options, reuse=True)

        # Declare losses, optimizers(trainers) and fid for evaluation
        self.g_loss = self.g_loss_function()
        self.d_loss = self.d_loss_function()
        self.g_train = self.g_trainer()
        self.d_train = self.d_trainer()
        # self.fid = self.fid_function()

    def g_loss_function(self):
        g_loss_X2Y = tf.losses.mean_squared_error(self.d_X2Y, tf.ones_like(self.d_X2Y))
        g_loss_Y2X = tf.losses.mean_squared_error(self.d_Y2X, tf.ones_like(self.d_Y2X))
        cyc_loss_X = tf.losses.absolute_difference(self.X, self.X2Y2X)
        cyc_loss_Y = tf.losses.absolute_difference(self.Y, self.Y2X2Y)
        g_loss = g_loss_X2Y + g_loss_Y2X + cyc_loss_X * L1_lambda + cyc_loss_Y * L1_lambda
        return g_loss

    def d_loss_function(self):
        d_Y_loss_real = tf.losses.mean_squared_error(self.d_Y, tf.ones_like(self.d_Y))
        d_Y_loss_fake = tf.losses.mean_squared_error(self.d_X2Y_sample, tf.zeros_like(self.d_X2Y_sample))
        d_Y_loss = (d_Y_loss_real + d_Y_loss_fake) / 2
        d_X_loss_real = tf.losses.mean_squared_error(self.d_X, tf.ones_like(self.d_X))
        d_X_loss_fake = tf.losses.mean_squared_error(self.d_Y2X_sample, tf.zeros_like(self.d_Y2X_sample))
        d_X_loss = (da_loss_real + da_loss_fake) / 2
        d_loss = d_Y_loss + d_X_loss
        return d_loss

    def g_trainer(self):
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        g_solver = tf.train.AdamOptimizer(args.learn_rate, args.beta1).minimize(self.g_loss, var_list=g_vars)
        return g_solver

    def d_trainer(self):
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        d_solver = tf.train.AdamOptimizer(args.learn_rate, args.beta1).minimize(self.d_loss, var_list=d_vars)
        return d_solver

    # def fid_function(self):
    #     INCEPTION_IMAGE_SIZE = (299, 299)
    #     real_resized = tf.image.resize_images(self.image_batch, INCEPTION_IMAGE_SIZE)
    #     fake_resized = tf.image.resize_images(self.g_output, INCEPTION_IMAGE_SIZE)
    #     return gan.eval.frechet_classifier_distance(real_resized, fake_resized, gan.eval.run_inception)

def buildDataset(self, x_path = X_PATH, y_path = Y_PATH):        
    x_Dataset = tf.data.Dataset.list_files( x_path + '/*.jpg')
    y_Dataset = tf.data.Dataset.list_files( y_path + '/*.jpg')

    x_images = x_Dataset.map(lambda x: tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x), channels = INPUT_SHAPE[2]), [INPUT_SHAPE[0], INPUT_SHAPE[1]]))
    y_images = y_Dataset.map(lambda x: tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x), channels = INPUT_SHAPE[2]), [INPUT_SHAPE[0], INPUT_SHAPE[1]]))

    xy_images = tf.data.Dataset.zip((x_images, y_images))
    xy_Dataset = xy_images.batch(BATCH_SIZE)
    return xy_Dataset

xy_Dataset = buildDataset()
# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# For saving/loading models
saver = tf.train.Saver()

# Load the last saved checkpoint during training or used by test
def load_last_checkpoint():
    saver.restore(sess, tf.train.latest_checkpoint('./'))

def train():
    for epoch in range(args.epoch):
        iterator = self.xy_Dataset.make_initializable_iterator()
        (x_next, y_next) = iterator.get_next()
        sess.run(iterator.initializer)

        # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
        
        iteration = 0
        while True:
            try:
                X, Y = self.session.run([x_next, y_next])

                if x_train.shape[0] != BATCH_SIZE:
                    break

                # Update G network and record fake outputs
                X2Y, Y2X, _ = self.sess.run([self.X2Y, self.Y2X, self.g_train], feed_dict={self.X: X, self.Y: Y})

                # Update D network
                _ = self.sess.run(self.d_train,feed_dict={self.X: batch_imagesX, self.Y: batch_imagesY, self.X2Y_sample: X2Y, self.X2Y_sample: Y2X})

                # Print losses
                if iteration % log_every == 0:
                    print('Iteration %d: Gen loss = %g | Discrim loss = %g' % (iteration, loss_g, loss_d))
                # Save
                if iteration % save_every == 0:
                    saver.save(sess, './dcgan_saved_model')
                iteration += 1

            except tf.errors.OutOfRangeError:
                print('epoch ' + str( epoch) + ' end.')
                break


def test():
    pass


# Ensure the output directory exists
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

if args.restore_checkpoint or args.mode == 'test':
    load_last_checkpoint()

if args.mode == 'train':
    train()
if args.mode == 'test':
    test()
       