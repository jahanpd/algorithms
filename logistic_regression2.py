import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class logistic_regression:
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

    @tf.function # call as TF function for efficiency
    def train_step(self): # define training function with a custom weight updating

        with tf.GradientTape() as tape: # this 'watches' trainable variables to automatically differentiate and find gradients
            Y_h = tf.nn.softmax((tf.matmul(self.x, self.W) + self.b)) # the softmax function that outputs a simple logistic_regression
            loss = self.loss(y_true=self.y,y_pred=Y_h) # find the loss of the predictions

        grads = tape.gradient(loss,[self.W,self.b]) # Computes the gradient using operations recorded in context of this tape.
        self.optimizer.apply_gradients(zip(grads, [self.W,self.b])) # applies the gradients to the variable
        return loss, Y_h

    def train(self, epochs, path=None): # define a training function
        self.cost_history, self.accuracy_history = [], [] # to store training progress metrics

        for epoch in range(epochs):
            c, Y_h = self.train_step() # train across full dataset (cf w a batch training method)

            accuracy = tf.keras.metrics.AUC() # calculate accuracy metric as AUC
            accuracy.update_state(self.y,Y_h)

            # store history
            self.cost_history.append(c.numpy())
            self.accuracy_history.append(accuracy.result().numpy())
            print("Epoch:", epoch,
                  " Cost:", self.cost_history[-1],
                  " AUC:",self.accuracy_history[-1])

        if path is not None:
            pass

        return self.cost_history, self.accuracy_history
