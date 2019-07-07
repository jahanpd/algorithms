import numpy as np
import tensorflow as tf

class logistic_regression:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.m, self.n = x.shape # m is rows and n is cols
        self.s, self.t = y.shape # m is rows and n is cols
        self.alpha = 0.01

        # create placeholders for training
        self.X = tf.placeholder(tf.float32, [None, self.n])
        self.Y = tf.placeholder(tf.float32, [None, self.t])

        # define trainable weights
        self.W = tf.Variable(tf.zeros([self.n,self.t]))

        # define the bias Variable
        self.b = tf.Variable(tf.zeros([self.t]))

        self.Y_h = tf.nn.softmax((tf.matmul(self.X, self.W) + self.b))

        self.cost = tf.nn.softmax_cross_entropy_with_logits(logits = self.Y_h,
                                                            labels = self.Y)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.alpha
                                                        ).minimize(self.cost)

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

    def train(self, epochs, path=None):
        with tf.Session() as sess:

            sess.run(self.init) # initialise variables

            self.cost_history, self.accuracy_history = [], [] # to store training progress

            for epoch in range(epochs):
                cost_per_epoch = 0
                sess.run(self.optimizer, feed_dict = {self.X: self.x, self.Y: self.y}) # run the optimizer

                c = sess.run(self.cost, feed_dict = {self.X: self.x, self.Y: self.y}) # calculate cost

                # calculate accuracy
                correct_pred = tf.equal(tf.argmax(self.Y_h,1), tf.argmax(self.Y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                # store history
                self.cost_history.append(sum(c))
                self.accuracy_history.append(accuracy.eval({self.X:self.x,self.Y:self.y})*100)

            correct_pred = tf.equal(tf.argmax(self.Y_h,1), tf.argmax(self.Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            if path is not None:
                save_path = self.saver.save(sess, path + "logisticRegression.ckpt")

            return self.cost_history, self.accuracy_history
