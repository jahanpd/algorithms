import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

tf.enable_eager_execution()

class autoencoder:
    def __init__(self,x):
        print(tf.executing_eagerly())
        self.x = x.astype(np.float32) # value of inputs

        self.m, self.n = x.shape # m is rows (number of samples) and n is cols (features)


    def build_model(self, layers, dim_red):
        inputs = Input(shape=(self.n,))
        f = Dense(self.n, activation='relu')(inputs)
        steps = (self.n - dim_red)/layers
        for n in np.arange(self.n - steps, dim_red, -steps).round().astype(int):
            f = Dense(n, activation='relu')(f)
        middle = Dense(dim_red, activation='relu')(f)
        f = Dense(dim_red + steps, activation='relu')(middle)
        for n in np.arange(dim_red + 2*steps, self.n, steps).round().astype(int):
            f = Dense(n, activation='relu')(f)
        output = Dense(self.n, activation='relu')(f)

        self.autoencode = Model(inputs=inputs, outputs=[output])
        self.autoencode.compile(optimizer="Adam", loss="mse")

        self.reduce_dims = Model(inputs=inputs, outputs=[middle])
        self.reduce_dims.compile(optimizer="Adam", loss="mse")
        self.autoencode.summary()

    def train(self, epoch):
        self.autoencode.fit(x=self.x, y=self.x, epochs=epoch)
        self.reduce_dims.set_weights(self.autoencode.get_weights())

    def predict(self, x, reduce = None):
        if reduce is None:
            return self.autoencode.predict(x.astype(np.float32))
        else:
            return self.reduce_dims.predict(x.astype(np.float32))
