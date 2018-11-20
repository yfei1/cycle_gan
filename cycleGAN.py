import keras
from keras.layers import (
        Dense,
        Activation,
        BatchNormalization,
        Conv2D,
        Input,
        Add
)
from keras.models import Sequential
from keras.activations import (
        softmax,
        tanh,
        sigmoid,
        relu
)
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


class CycleGAN:
    def __init__(self):
        self.model = Sequential()

    def discriminator(self):
        # TODO bianry classifier
        pass
        
    def generator(self):
        # TODO add more downsampling layers
        self.conv()
        # TODO add more res blocks
        self.addResBlock(2, 64, 3, (1, 1), 'leakyReLU')
        # TODO add more upsampling layers
        self.deconv()
        
    # Downsampling
    def conv(self):
        # TODO
        pass

    def deconv(self):
        # TODO
        pass

    def addResBlock(self, repetitions, filters, kernel_size, strides, activations):
        for i in range(repetitions):
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Activation(activation_func(activations)))
            self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same'))

    def build(self):
        self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
                )
        print(self.model.summary())

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=5, batch_size=32)

    def test(self, x_test, y_test):
        self.model.evaluate(x_test, y_test, batch_size=32)


# TODO Preprocess input images
x_train = np.random.normal(size=[64, 100, 100, 3])
y_train = np.random.normal(size=[64, 100, 100, 16])
x_test, y_test = x_train, y_train

cycleGAN = CycleGAN()
cycleGAN.model.add(Conv2D(16, 3, padding='same', input_shape=(100, 100, 3)))
cycleGAN.addResBlock(repetitions=2, filters=16, kernel_size=3, strides=(1,1), activations='ReLU')
cycleGAN.addResBlock(repetitions=2, filters=16, kernel_size=3, strides=(1,1), activations='ReLU')
cycleGAN.addResBlock(repetitions=2, filters=16, kernel_size=3, strides=(1,1), activations='ReLU')


cycleGAN.build()

cycleGAN.train(x_train, y_train)




