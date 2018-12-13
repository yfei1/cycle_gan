import keras.backend as K
import keras.initializers as initializers
from keras.engine.topology import Layer
from keras.layers import Reshape

class GroupNormalization(Layer):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """
    def __init__(
        self,
        epsilon = 1e-5,
        G = 2,
        scale_initializer='ones', 
        bias_initializer='zeros', 
        **kwargs
    ):
        super(GroupNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-5
        self.G = G
        self.scale_initializer = initializers.get(scale_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.scale = self.add_weight(shape=(1, 1, input_shape[3]), 
                                     initializer=self.scale_initializer,
                                     trainable=True,
                                     name='{}_scale'.format(self.name))
        self.bias = self.add_weight(shape=(1, 1, input_shape[3]),
                                    initializer=self.bias_initializer,
                                    trainable=True,
                                    name='{}_bias'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        G = self.G
        _, H, W, C = K.int_shape(x)
        x = Reshape((H, W, C // G, G))(x)
        mean = K.mean(x, axis=[2, 3, 4], keepdims=True)
        std = K.std(x, axis=[2, 3, 4], keepdims=True)
        x = (x - mean) * (1 / (std + self.epsilon))
        x = Reshape((H, W, C))(x)
        return x * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape