import tensorflow as tf
from tensorflow import nn
from utils import lazy_property

# The RNN implementation is based on http://danijar.com/variable-sequence-lengths-in-tensorflow/
class Classification:
    def __init__(self, data, target, length, learning_rate, stage, num_RNN, num_FCN):
        self.data = data
        self.target = target
        self.layer1 = num_FCN
        self.stage = stage
        length = tf.cast(length, tf.int32)
        self.length = length  
        self.learning_rate = learning_rate     
        self._num_RNN = num_RNN
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        num_units = [self._num_RNN, self._num_RNN]
        cells = [nn.rnn_cell.GRUCell(n) for n in num_units]
        stacked_rnn = tf.contrib.rnn.MultiRNNCell(cells)
        # Recurrent network.    
        output_RNN, _ = nn.dynamic_rnn(
            stacked_rnn,
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        ) 

        batch_size = tf.shape(output_RNN)[0]
        max_length = int(output_RNN.get_shape()[1])
        output_size = int(output_RNN.get_shape()[2])
        
        output_reshape = tf.reshape(output_RNN, [-1, output_size])     
        weight_l1, bias_l1 = self._weight_and_bias(self._num_RNN, self.layer1)
        output_l1 = tf.nn.tanh(tf.matmul(output_reshape, weight_l1) + bias_l1)
        output_drop_out_l1 = tf.nn.dropout(output_l1, 0.2, seed=47)        
        output_drop_out_reshape_l1 = tf.reshape(output_drop_out_l1, [batch_size, max_length, self.layer1])
        
        last = self._last_relevant(output_drop_out_reshape_l1, self.length)

        weight_class, bias_class = self._weight_and_bias(
            self.layer1, int(self.target.get_shape()[1]))
        # Softmax layer.
        prediction = tf.nn.softmax(tf.matmul(last, weight_class) + bias_class)
        return prediction, weight_l1, bias_l1, weight_class, bias_class   

    @lazy_property
    def cost(self):
        prediction,_,_,_,_ = self.prediction
        cross_entropy = -tf.reduce_sum(self.target * tf.log(prediction))
        tf.summary.scalar('cross_entropy', cross_entropy)
        vars = tf.trainable_variables() 
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name ]) * 0.005
        return (cross_entropy+lossL2)

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        variables = tf.trainable_variables()
        # Optimizing only second layer RNNs
        variables_to_optimize = [v for v in variables if v.name.split('/')[0]=='rnn' 
                                      and v.name.split('/')[2]=='cell_1']               
        _,var1,var2, var3, var4 = self.prediction
        opt_cond = tf.cond(self.stage < 1, lambda: optimizer.minimize(self.cost, var_list=[variables_to_optimize, var1, var2, var3, var4]), lambda: optimizer.minimize(self.cost))
        return opt_cond


    @lazy_property
    def error(self):
        prediction, _,_,_,_ = self.prediction
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(prediction, 1))
        tf.summary.scalar('error', tf.reduce_mean(tf.cast(mistakes, tf.float32)))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant