import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave
import tensorflow.contrib.gan as gan

from module import *
# from utils import *


BATCH_SIZE = 1
INPUT_WIDTH = 128
INPUT_DIM = 3
X_PATH = './datasets/horse2zebra/trainA'
Y_PATH = './datasets/horse2zebra/trainB'
MODE = 'train'
OUT = './output'
log_every = 10
save_every = 10
L1_lambda = 10
RESTORE = False
epochs = 2
learn_rate = 2e-4
beta1 = 0.5
MODE = 'test'

class Model(object):
    def __init__(self):
        self.discriminator = discriminator
        self.generator = generator_condnet
        # self.criterionGAN = mae_criterion

        self.X = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.Y = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.X2Y_sample = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.Y2X_sample = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])

        self.X2Y = self.generator(self.X, self.Y, False, name="generatorX2Y")
        self.X2Y2X = self.generator(self.X2Y, self.X, False, name="generatorY2X")
        self.Y2X = self.generator(self.Y, self.X, True, name="generatorY2X")
        self.Y2X2Y = self.generator(self.Y2X, self.Y, True, name="generatorX2Y")

        self.d_X2Y = self.discriminator(self.X2Y, reuse=False, name="discriminatorY")
        self.d_Y2X = self.discriminator(self.Y2X, reuse=False, name="discriminatorX")

        self.d_Y = self.discriminator(self.Y, reuse=True, name="discriminatorY")
        self.d_X = self.discriminator(self.X, reuse=True, name="discriminatorX")
        self.d_Y2X_sample = self.discriminator(self.Y2X_sample, reuse=True, name="discriminatorY")
        self.d_X2Y_sample = self.discriminator(self.X2Y_sample, reuse=True, name="discriminatorX")

        # Declare losses, optimizers(trainers) and fid for evaluation
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.g_loss = self.g_loss_function()
        self.d_loss = self.d_loss_function()
        self.g_train = self.g_trainer()
        self.d_train = self.d_trainer()
        self.fid = self.fid_function()

    def g_loss_function(self):
        g_loss_X2Y = tf.reduce_mean(tf.losses.mean_squared_error(self.d_X2Y, tf.ones_like(self.d_X2Y)))
        g_loss_Y2X = tf.reduce_mean(tf.losses.mean_squared_error(self.d_Y2X, tf.ones_like(self.d_Y2X)))
        cyc_loss_X = tf.reduce_mean(tf.losses.absolute_difference(self.X, self.X2Y2X))
        cyc_loss_Y = tf.reduce_mean(tf.losses.absolute_difference(self.Y, self.Y2X2Y))
        g_loss = g_loss_X2Y + g_loss_Y2X + cyc_loss_X * L1_lambda + cyc_loss_Y * L1_lambda
        return g_loss

    def d_loss_function(self):
        d_Y_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(self.d_Y, tf.ones_like(self.d_Y)))
        d_Y_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(self.d_X2Y_sample, tf.zeros_like(self.d_X2Y_sample)))
        d_Y_loss = (d_Y_loss_real + d_Y_loss_fake) / 2
        d_X_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(self.d_X, tf.ones_like(self.d_X)))
        d_X_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(self.d_Y2X_sample, tf.zeros_like(self.d_Y2X_sample)))
        d_X_loss = (d_X_loss_real + d_X_loss_fake) / 2
        d_loss = d_Y_loss + d_X_loss
        return d_loss

    def g_trainer(self):
        g_solver = tf.train.AdamOptimizer(learn_rate, beta1).minimize(self.g_loss, var_list=self.g_vars)
        return g_solver

    def d_trainer(self):
        d_solver = tf.train.AdamOptimizer(learn_rate, beta1).minimize(self.d_loss, var_list=self.d_vars)
        return d_solver

    def fid_function(self):
        INCEPTION_IMAGE_SIZE = (299, 299)
        real_resized1 = tf.image.resize_images(self.X, INCEPTION_IMAGE_SIZE)
        fake_resized1 = tf.image.resize_images(self.Y2X, INCEPTION_IMAGE_SIZE)

        real_resized2 = tf.image.resize_images(self.Y, INCEPTION_IMAGE_SIZE)
        fake_resized2 = tf.image.resize_images(self.X2Y, INCEPTION_IMAGE_SIZE)

        d1 = gan.eval.frechet_classifier_distance(real_resized1, fake_resized1, gan.eval.run_inception)
        d2 = gan.eval.frechet_classifier_distance(real_resized2, fake_resized2, gan.eval.run_inception)
        return d1

def buildDataset(x_path = X_PATH, y_path = Y_PATH):        
    x_Dataset = tf.data.Dataset.list_files( x_path + '/*.jpg')
    y_Dataset = tf.data.Dataset.list_files( y_path + '/*.jpg')

    x_images = x_Dataset.map(lambda x: tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x), channels = INPUT_DIM), [INPUT_WIDTH, INPUT_WIDTH]))
    y_images = y_Dataset.map(lambda x: tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x), channels = INPUT_DIM), [INPUT_WIDTH, INPUT_WIDTH]))

    xy_images = tf.data.Dataset.zip((x_images, y_images))
    xy_Dataset = xy_images.batch(BATCH_SIZE)
    return xy_Dataset


if not os.path.exists(OUT):
    os.makedirs(OUT)
    os.makedirs(OUT+'/X2Y_OUT')
    os.makedirs(OUT+'/Y2X_OUT')
    
xy_Dataset = buildDataset()
model = Model()
# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# For saving/loading models
saver = tf.train.Saver()

# Load the last saved checkpoint during training or used by test
def load_last_checkpoint():
    saver.restore(sess, tf.train.latest_checkpoint('./'))

def train():
    for epoch in range(epochs):
        print('========================== EPOCH %d  ==========================' % epoch)
        iterator = xy_Dataset.make_initializable_iterator()
        (x_next, y_next) = iterator.get_next()
        sess.run(iterator.initializer)

        # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)        
        iteration = 0
        while True:
            try:
                X, Y = sess.run([x_next/127.5-1, y_next/127.5-1])

                if X.shape[0] != BATCH_SIZE:
                    break

                # Update G network and record fake outputs
                X2Y, Y2X, _ = sess.run([model.X2Y, model.Y2X, model.g_train], feed_dict={model.X: X, model.Y: Y})
                
                # Update D network
                d_loss, g_loss, _ = sess.run([model.d_loss, model.g_loss, model.d_train],feed_dict={model.X: X, model.Y: Y, model.X2Y_sample: X2Y, model.Y2X_sample: Y2X})
                
                # Print losses
                if iteration % log_every == 0:
                    print('Iteration %d: Gen loss = %g | Discrim loss = %g' % (iteration, g_loss, d_loss))
                # Save
                if iteration % save_every == 0:
                    saver.save(sess, './dcgan_saved_model')
                iteration += 1

            except tf.errors.OutOfRangeError:
                print('epoch ' + str( epoch) + ' end.')
                break
        
        saver.save(sess, './dcgan_saved_model')
        # sess.run(iterator.initializer)
        # X, Y = sess.run([x_next/127.5-1, y_next/127.5-1])

        # fid_ = sess.run(model.fid, feed_dict={model.X: X, model.Y: Y})
        # print('**** INCEPTION DISTANCE: %g ****' % fid_)


def test():
    iterator = xy_Dataset.make_initializable_iterator()
    (x_next, y_next) = iterator.get_next()
    sess.run(iterator.initializer)
    X, Y = sess.run([x_next/127.5-1, y_next/127.5-1])
    
    gen_y, gen_x = sess.run([model.X2Y, model.Y2X], feed_dict={model.X: X, model.Y: Y})    
    
    # Rescale the image from (-1, 1) to (0, 255)
    gen_y = ((gen_y / 2) - 0.5) * 255
    gen_x = ((gen_x / 2) - 0.5) * 255
    X = ((X / 2) - 0.5) * 255
    Y = ((Y / 2) - 0.5) * 255
    
    # Convert to uint8
    gen_y = gen_y.astype(np.uint8)
    gen_x = gen_x.astype(np.uint8)
    
    X = X.astype(np.uint8)
    Y = Y.astype(np.uint8)
    
    # Save images to disk
    for i in range(0, BATCH_SIZE):
        img_Xi = X[i]
        s = OUT+'/X2Y_OUT/X_'+str(i)+'.jpg'
        imsave(s, img_Xi)
        
        img_yi = gen_y[i]
        s = OUT+'/X2Y_OUT/Y_'+str(i)+'.jpg'
        imsave(s, img_yi)
        
        img_Yi = Y[i]
        s = OUT+'/Y2X_OUT/Y_'+str(i)+'.jpg'
        imsave(s, img_Yi)
        
        img_xi = gen_x[i]
        s = OUT+'/Y2X_OUT/X_'+str(i)+'.jpg'
        imsave(s, img_xi)
  

if MODE == 'train':
    train()

if MODE == 'test':
    load_last_checkpoint()
    test()
       