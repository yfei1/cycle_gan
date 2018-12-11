import tensorflow as tf
import numpy as np
import glob
import os
import time
from scipy.misc import imsave, imread, imresize
from random import shuffle
import tensorflow.contrib.gan as gan
from tqdm import tqdm
from module import *
import matplotlib.pyplot as plt
 

# from utils import *


BATCH_SIZE = 2
FID_BATCH_SIZE = 30
INPUT_WIDTH = 256
INPUT_DIM = 3
train_x_path = './datasets/horse2zebra/trainA'
train_y_path = './datasets/horse2zebra/trainB'
test_x_path = './datasets/horse2zebra/testA'
test_y_path = './datasets/horse2zebra/testB'
OUT = './output/resnet_in'
log_every = 20
save_every = 200
L1_lambda = 10
RESTORE = False
epochs = 10
learn_rate = 2e-4
beta1 = 0.5
MODE = 'train'
MODEL_PATH = './cycleGAN_resnet_in'

class Model(object):
    def __init__(self):
        self.discriminator = discriminator
        self.generator = generator_resnet
        # self.criterionGAN = mae_criterion

        self.X = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.Y = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.X2Y_sample = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        self.Y2X_sample = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DIM])
        
        self.true_labels = tf.placeholder(tf.int32, [None, 1])
        self.fake_labels = tf.placeholder(tf.int32, [None, 1])

        self.X2Y = self.generator(self.X, self.Y, False, name="generatorX2Y")
        self.X2Y2X = self.generator(self.X2Y, self.X, False, name="generatorY2X")
        self.Y2X = self.generator(self.Y, self.X, True, name="generatorY2X")
        self.Y2X2Y = self.generator(self.Y2X, self.Y, True, name="generatorX2Y")

        # TODO: true is 0, fake is 1
        self.d_X2Y = self.discriminator(self.X2Y, self.fake_labels, reuse=False, name="discriminatorY")
        self.d_Y2X = self.discriminator(self.Y2X, self.fake_labels, reuse=False, name="discriminatorX")

        self.d_Y = self.discriminator(self.Y, self.true_labels, reuse=True, name="discriminatorY")
        self.d_X = self.discriminator(self.X, self.true_labels, reuse=True, name="discriminatorX")
        self.d_Y2X_sample = self.discriminator(self.Y2X_sample, self.fake_labels, reuse=True, name="discriminatorY")
        self.d_X2Y_sample = self.discriminator(self.X2Y_sample, self.fake_labels, reuse=True, name="discriminatorX")

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
        d_solver = tf.train.AdamOptimizer(learn_rate/4, beta1).minimize(self.d_loss, var_list=self.d_vars)
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

def buildDataset(x_path, y_path, BATCH_SIZE, weShuffle = True):        
    x_files = glob.glob(x_path + "/*.jpg")  
    y_files = glob.glob(y_path + "/*.jpg")
    num_of_files = min(len(x_files), len(y_files))
    
    
    if weShuffle:
        shuffle(x_files)
        shuffle(y_files)

    x_images = [imresize(imread(x), (INPUT_WIDTH,INPUT_WIDTH)) for x in x_files]
    x_images = x_images[0:num_of_files]
    x_images = split(x_images,BATCH_SIZE)
    
    y_images = [imresize(imread(x), (INPUT_WIDTH,INPUT_WIDTH)) for x in y_files]
    y_images = y_images[0:num_of_files]    
    y_images = split(y_images,BATCH_SIZE)
    return np.array(x_images), np.array(y_images)
    
    
def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

if not os.path.exists(OUT):
    os.makedirs(OUT)
    os.makedirs(OUT+'/X2Y_OUT')
    os.makedirs(OUT+'/Y2X_OUT')
    
train_X, train_Y = buildDataset(train_x_path, train_y_path, BATCH_SIZE)
test_X, test_Y = buildDataset(test_x_path, test_y_path, BATCH_SIZE, weShuffle = False)
#fid_X, fid_Y = buildDataset(test_x_path, test_y_path, FID_BATCH_SIZE, weShuffle = False)

model = Model()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Start session
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# For saving/loading models
saver = tf.train.Saver()

# Load the last saved checkpoint during training or used by test
def load_last_checkpoint():
    saver.restore(sess, tf.train.latest_checkpoint('./'))

def train():
    # writer = tf.summary.FileWriter("./logs", sess.graph)
    print("Model will be saved at %s", MODEL_PATH)
    tqdm_epochs = tqdm(range(epochs))

    d_loss_last, g_loss_last = None, None

    epochs_iteration = []
    epochs_d_loss = []
    epochs_g_loss = []
    epochs_fid = []

    for epoch in tqdm_epochs:
        BATCH_SIZE = 1

        iteration = 0
        for i in range(len(train_X)):
            X = train_X[i]/127.5-1
            Y = train_Y[i]/127.5-1

            if X.shape[0] != BATCH_SIZE:
                break
            # Update G network and record fake outputs
            X2Y, Y2X, _ = sess.run([
                model.X2Y, model.Y2X, model.g_train], 
                feed_dict={
                    model.X: X, 
                    model.Y: Y,
                    model.true_labels: np.ones((BATCH_SIZE, 1), dtype=np.int32),
                    model.fake_labels: np.ones((BATCH_SIZE, 1), dtype=np.int32)
                    }
                )
            
            # Update G network and record fake outputs
            X2Y, Y2X, _ = sess.run([
                model.X2Y, model.Y2X, model.g_train], 
                feed_dict={
                    model.X: X, 
                    model.Y: Y,
                    model.true_labels: np.ones((BATCH_SIZE, 1), dtype=np.int32),
                    model.fake_labels: np.ones((BATCH_SIZE, 1), dtype=np.int32)
                    }
                )


            # Update D network
            d_loss, g_loss, _ = sess.run([
                model.d_loss, model.g_loss, model.d_train],
                feed_dict={
                    model.X: X, 
                    model.Y: Y, 
                    model.X2Y_sample: X2Y, 
                    model.Y2X_sample: Y2X,
                    model.true_labels: np.ones((BATCH_SIZE, 1), dtype=np.int32),
                    model.fake_labels: np.zeros((BATCH_SIZE, 1), dtype=np.int32)
                    }
                )
            # d_loss_sum_epoch, g_loss_sum_epoch = d_loss_summary, g_loss_summary
            tqdm_epochs.set_description('Iteration %d: Gen loss = %g | Discrim loss = %g' % (iteration, g_loss, d_loss))
            d_loss_last, g_loss_last = d_loss, g_loss

            # Save
            if iteration % save_every == 0:
                saver.save(sess, MODEL_PATH)
            iteration += 1
       
        saver.save(sess, MODEL_PATH)

        with tf.device('/cpu:0'):            
            fid_X, fid_Y = buildDataset(test_x_path, test_y_path, FID_BATCH_SIZE, weShuffle = False)
            fid_X = fid_X[0]/127.5-1
            fid_Y = fid_Y[0]/127.5-1
            fid = sess.run(model.fid, feed_dict={model.X: fid_X, model.Y: fid_Y})
            print('**** INCEPTION DISTANCE: %g ****' % fid)

        epochs_iteration.append(epoch)
        epochs_d_loss.append(d_loss_last)
        epochs_g_loss.append(g_loss_last)
        epochs_fid.append(fid)

    # end of all epochs
    plt.grid()
    plt.plot(epochs_iteration, epochs_d_loss, label='change of d_loss in all epochs')
    plt.savefig("plots/epochs_d_loss.jpg")
    plt.clf()
    plt.grid()
    plt.plot(epochs_iteration, epochs_g_loss, label='change of g_loss in all epochs')
    plt.savefig("plots/epochs_g_loss.jpg")
    plt.clf()
    plt.grid()
    plt.plot(epochs_iteration, epochs_fid,'r', label='change of fid in all epochs')
    plt.savefig("plots/epochs_fid.jpg")


def test(): 
    for idx in range(len(test_X)):
        X = test_X[idx]/127.5-1
        Y = test_Y[idx]/127.5-1
    
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
            s = OUT+'/X2Y_OUT/X_'+str(idx)+str(i)+'.jpg'
            imsave(s, img_Xi)

            img_yi = gen_y[i]
            s = OUT+'/X2Y_OUT/Y_'+str(idx)+str(i)+'.jpg'
            imsave(s, img_yi)

            img_Yi = Y[i]
            s = OUT+'/Y2X_OUT/Y_'+str(idx)+str(i)+'.jpg'
            imsave(s, img_Yi)

            img_xi = gen_x[i]
            s = OUT+'/Y2X_OUT/X_'+str(idx)+str(i)+'.jpg'
            imsave(s, img_xi)
  

if MODE == 'train':
    train()

if MODE == 'test':
    load_last_checkpoint()
    test()
       
