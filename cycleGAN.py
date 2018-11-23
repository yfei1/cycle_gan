import keras
from keras.layers import (
        Dense,
        Activation,
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        Input,
        Add,
        Reshape
)
from keras.models import Sequential
from keras.activations import (
        softmax,
        tanh,
        sigmoid,
        relu
)
from keras import optimizers
import numpy as np


""" Return an activation function as needed"""
def activation_func(name):

    def leakyReLu(input_):
        return relu(input_, alpha=.3)

    return {
            'ReLU': relu,
            'tanh': tanh,
            'leakyReLU': leakyReLu,
            'sigmoid': sigmoid,
            'softmax': softmax
    }[name]


INPUT_SHAPE = (128, 128, 3)

# Upsampling parameters
NUM_CONV_LAYERS = 1
INIT_FILTER = 8
KERNEL_SIZE = 3
CONV_STRIDES = (2, 2)

# Residual Block parameters
NUM_REPETITIONS = 2
NUM_RES_BLOCKS = 1
RES_STRIDES = (1, 1)

class CycleGAN:
    def __init__(self):
        self.generator_xy = self.generator()
        self.generator_yx = self.generator()
        self.discriminator_x = self.discriminator()
        self.discriminator_y = self.discriminator()

        X, Y     = Input(INPUT_SHAPE)   , Input(INPUT_SHAPE)
        X_, Y_   = self.generator_yx(Y) , self.generator_xy(X)
        X__, Y__ = self.generator_yx(Y_), self.generator_xy(X_)
        X_identity, Y_identity = self.generator_yx(X), self.generator_xy(Y)

        adam = optimizers.Adam(lr=5e-2, decay=.9)
 
        self.discriminator_x.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
        self.discriminator_y.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

        self.discriminator_x.trainable = False
        self.discriminator_y.trainable = False

        X_valid, Y_valid = self.discriminator_x(X_), self.discriminator_y(Y_)

        # TODO: Figure out the weights of the losses
        self.generators = Model(inputs=[X, Y], outputs=[X_valid, Y_valid, X__, Y__, X_identity, Y_identity])
        
        # The paper suggests using L1 norm for the last four loss functions, try out different settings if it doesn't work
        self.generators.compile(loss=['mse']*2 + ['mae']*4, optimizer=adam)


    def discriminator(self):
        # Input is an image
        model = Sequential()
        filter_out = self.conv(model, 'leakyReLU')
        # Not sure if a binary discriminator needs residual blocks
        self.residuals(model, 'leakyReLU', filter_out)
        self._addConvBlock(model, 'leakyReLU', filters=1, kernel_size=KERNEL_SIZE, strides=CONV_STRIDES)
        model.add(Reshape((-1,)))
        model.add(Dense(1))

        return model

        
    def generator(self):
        model = Sequential()
        filter_out = self.conv(model, 'leakyReLU')
        self.residuals(model, 'leakyReLU', filter_out)
        self.deconv(model, 'leakyReLU', filter_out)

        return model
        
    # Downsampling - return the final filter size
    def conv(self, model, activations, init_filter=INIT_FILTER, kernel_size=KERNEL_SIZE, strides=CONV_STRIDES):
        self._addConvBlock(model, activations, init_filter, kernel_size, strides, True)

        for i in range(NUM_CONV_LAYERS-1):
            init_filter *= 2
            self._addConvBlock(model, activations, init_filter, kernel_size, strides)

        return init_filter
    
    def deconv(self, model, activations, filters, kernel_size=KERNEL_SIZE, strides=CONV_STRIDES):
        for i in range(NUM_CONV_LAYERS-1):
            filters /= 2
            self._addDeconvBlock(model, activations, filters, kernel_size, strides)

        self._addDeconvBlock(model, activations, INPUT_SHAPE[2], kernel_size, strides)

    def residuals(self, model, activations, filters, kernel_size=KERNEL_SIZE, strides=RES_STRIDES, repetitions=NUM_REPETITIONS):
        for i in range(NUM_RES_BLOCKS):
            self._addResBlock(model, activations, filters, kernel_size, strides, repetitions)

    def _addResBlock(self, model, activations, filters, kernel_size, strides, repetitions):
        for i in range(repetitions):
            self._addConvBlock(model, activations, filters, kernel_size, strides)

    def _addDeconvBlock(self, model, activations, filters, kernel_size=KERNEL_SIZE, strides=CONV_STRIDES):
        model.add(BatchNormalization(axis=3))
        model.add(Activation(activation_func(activations)))
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same'))

    def _addConvBlock(self, model, activations, filters, kernel_size, strides, input_layer=False):
        if not input_layer:
            model.add(BatchNormalization(axis=3))
            model.add(Activation(activation_func(activations)))
            model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same'))
        else:
            model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', input_shape=INPUT_SHAPE))

    def train(self, x_train, y_train):
        # TODO: implements training process
        pass

    def test(self, x_test, y_test):
        # TODO: implements evaluation
        pass

# TODO Preprocess input images
x_train = np.random.normal(size=[2560, 128, 128, 3])
# y_train = x_train # train tensor for the generator
y_train = np.random.normal(size=[2560, 1]) # train tensor for the discriminator

x_test = np.random.normal(size=[2560, 128, 128, 3])
# y_test = x_test # test tensor for the generator
y_test = np.random.normal(size=[2560, 1]) # test tensor for the discriminator

cycleGAN = CycleGAN()
#model = cycleGAN.generator()
model = cycleGAN.discriminator()

cycleGAN.build(model)

cycleGAN.train(x_train, y_train, model)
print(cycleGAN.test(x_test, y_test, model))



