# Copyright (c) 2020
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# ABN 41 687 119 230
#
# Author: Ahmadreza Ahmadi

# This file includes the class that includes the deep learning model for supervised terrrian classification with attention mechanism

import tensorflow as tf
from tensorflow import nn
from utils import lazy_property

# The class does the terrain classification and it includes RNNs with attention
class Classification:

    def __init__(self, data, target, length, learning_rate, num_RNNs):
        self.data = data
        self.target = target
        length = tf.cast(length, tf.int32)
        self.length = length  
        self.learning_rate = learning_rate     
        self._num_RNNs = num_RNNs        
        self.prediction
        self.error
        self.optimize

    # returns the output prediction of the classifier 
    @lazy_property
    def prediction(self):
        # Recurrent network.
        output_RNN, _ = nn.dynamic_rnn(
            nn.rnn_cell.GRUCell(self._num_RNNs),
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        )      
        last = self._attention(output_RNN)        
        weight, bias = self._weight_and_bias(
            self._num_RNNs, int(self.target.get_shape()[1]))
        # Softmax layer.
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    # returns the classification cost
    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))       
        tf.summary.scalar('cross_entropy', cross_entropy)
        vars   = tf.trainable_variables() 
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name ]) * 0.005
        return (cross_entropy+lossL2)    
    
    # performs adam optimizer
    @lazy_property
    def optimize(self):        
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)

    # returns the classification error 
    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        tf.summary.scalar('error', tf.reduce_mean(tf.cast(mistakes, tf.float32)))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))    

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    # returns the values of the last step (step T) of RNNs
    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
    
    # The attention mechanism is from https://github.com/ilivans/tf-rnn-attention
    @staticmethod
    def _attention(inputs, attention_size=50, time_major=False, return_alphas=False):

        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        initializer = tf.random_normal_initializer(stddev=0.1)

        # Trainable parameters
        w_omega = tf.get_variable(name="w_omega", shape=[hidden_size, attention_size], initializer=initializer)
        b_omega = tf.get_variable(name="b_omega", shape=[attention_size], initializer=initializer)
        u_omega = tf.get_variable(name="u_omega", shape=[attention_size], initializer=initializer)

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas