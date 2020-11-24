# Copyright (c) 2020
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# ABN 41 687 119 230
#
# Author: Ahmadreza Ahmadi

# This file includes the class that includes the deep learning model for semi-supervised terrrian classification: the prediction part

import tensorflow as tf
from tensorflow import nn
from utils import lazy_property

# The class does the one step prediction and it includes RNNs: 
class Prediction:

    def __init__(self, data, target, length, learning_rate, num_RNN, ext):
        self.data = data
        self.target = target
        length = tf.cast(length, tf.int32)
        self.length = length  
        self.learning_rate = learning_rate     
        self._num_RNN = num_RNN
        self.ext = ext
        self.prediction
        self.error
        self.optimize

    # returns the output prediction of the predictor RNNs
    @lazy_property
    def prediction(self):
        # Recurrent network.
        num_units = [self._num_RNN]
        cells = [nn.rnn_cell.GRUCell(n) for n in num_units]
        stacked_rnn = tf.contrib.rnn.MultiRNNCell(cells)
        output, _ = nn.dynamic_rnn(
            stacked_rnn,
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        ) 
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        target_size = int(self.target.get_shape()[2])
        
        output_reshape = tf.reshape(output, [-1, output_size])         
        
        weight, bias = self._weight_and_bias(
            self._num_RNN, target_size)
        # Tanh layer.
        prediction = self.ext * tf.nn.tanh(tf.matmul(output_reshape, weight) + bias)
        prediction_reshape = tf.reshape(prediction, [batch_size, max_length, target_size])              
        return prediction_reshape

    # returns the prediction cost 
    @lazy_property
    def cost(self):
        MSE = tf.reduce_mean(tf.squared_difference(self.prediction, self.target))
        tf.summary.scalar('MSE', MSE)
        return MSE    

    # performs adam optimizer
    @lazy_property
    def optimize(self):        
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)

     # returns the prediction error
    @lazy_property
    def error(self):
        MSE = tf.reduce_mean(tf.squared_difference(self.prediction, self.target))
        tf.summary.scalar('MSE', MSE)
        return (MSE / self.ext)    

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)