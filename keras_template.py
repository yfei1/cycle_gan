import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np


model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10))
model.add(keras.activations.tanh)

model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd', 
        metrics=['accuracy']
        )

print(model.summary())

x_train = np.random.normal(size=[64, 100])
y_train = np.random.normal(size=[64, 10])

x_test, y_test = x_train, y_train

model.fit(x_train, y_train, epochs=5, batch_size=32)

print(model.evaluate(x_test, y_test, batch_size=128))


