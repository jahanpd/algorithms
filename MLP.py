import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Softmax, Dropout
from tensorflow.keras.models import Model
tf.enable_eager_execution()

class MLP:
    def __init__(self,x,y):
        print(tf.executing_eagerly())
        self.x = x.astype(np.float32) # value of inputs
        self.y = y.astype(np.float32) # value of labels
        self.m, self.n = x.shape # m is rows (number of samples) and n is cols (features)
        self.s, self.t = y.shape # s is rows (number of samples) and n is lables (number of 'labels')

        self.W = tf.Variable(tf.zeros([self.n,self.t])) # define variable for weights
        self.b = tf.Variable(tf.zeros([self.t])) # define the variable for bias

        self.optimizer = tf.keras.optimizers.Adam() # use the Adam optimizer
        self.loss = tf.keras.losses.CategoricalCrossentropy() # Use Cat Cross Entropy loss

    def build(self, layers, nodes):
        inputs = Input(shape=(self.n,))
        f = Dense(nodes, activation=tf.nn.leaky_relu)(inputs)
        for n in np.arange(layers):
            f = Dense(nodes, activation=tf.nn.leaky_relu)(f)
            f = Dropout(0.5)(f)
        output = Dense(10, activation='softmax')(f)
        self.mlp = Model(inputs=inputs, outputs=[output])
        self.mlp.compile(optimizer=self.optimizer, loss=self.loss, metrics=[tf.keras.metrics.categorical_accuracy])
        self.mlp.summary()

    def train(self, epoch, path=None): # define a training function
        self.history = self.mlp.fit(x=self.x,
                                    y=self.y,
                                    validation_split=0.1,
                                    epochs=epoch)

    def evaluate(self, x, y): # have the option to predict on new data alone, or test set
        return self.mlp.evaluate(x=x.astype(np.float32), y=y.astype(np.float32))[1]
