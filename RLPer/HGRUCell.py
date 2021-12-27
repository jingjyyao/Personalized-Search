import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

class HGRUCell(core_rnn_cell.RNNCell):

    '''
    input_shape = [batch_size, num_query, features+2]
    state_shape = [batch_size, num_query, num_units]
    num_units = short_state_size + long_state_size
    '''

    def __init__(self, num_units, 
                short_state_size,
                long_state_size,
                activation=None,
                kernel_initializer=None,
                bias_initializer=None):
        self._num_units = num_units
        self._short_state_size = short_state_size
        self._long_state_size = long_state_size
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        assert(self._short_state_size+self._long_state_size==self._num_units)
        with tf.variable_scope(scope or type(self).__name__):  # "Hierachical-GRU-Cell"
            query_input, b, e = array_ops.split(
                value=inputs, num_or_size_splits=[int(inputs.get_shape()[1]-2), 1, 1], axis=1)
            session_state, user_state = array_ops.split(
                value=state, num_or_size_splits=[self._short_state_size, self._long_state_size], axis=1)

            #if session begins:
            session_state = constant_op.constant(0, dtype=np.float64, shape=session_state.get_shape()) + (1-b)*session_state
            state = array_ops.concat([session_state, user_state], 1)

            #Or:
            # Lambda = tf.get_variable("Lambda",
            #         shape = [2048, 1024])
            # session_state = b*tf.matmul(user_state, Lambda) + (1-b)*session_state
            # state = array_ops.concat([session_state, user_state], 2)

            total_input = array_ops.concat([query_input, session_state], 1)

            with tf.variable_scope("Gates1"):  # Reset gate and update gate.
                short_value = math_ops.sigmoid(
                  core_rnn_cell._linear([query_input, session_state], 2 * self._short_state_size, True, self._bias_initializer,
                          self._kernel_initializer))
                r1, u1 = array_ops.split(value=short_value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("Gates2"):  # Reset gate and update gate.
                long_value = math_ops.sigmoid(
                  core_rnn_cell._linear([session_state, user_state], 2 * self._long_state_size, True, self._bias_initializer,
                          self._kernel_initializer))
                r2, u2 = array_ops.split(value=long_value, num_or_size_splits=2, axis=1)

            with tf.variable_scope("Candidate1"):
                c1 = self._activation(
                  core_rnn_cell._linear([query_input, r1 * session_state], self._short_state_size, True,
                          self._bias_initializer, self._kernel_initializer))
            with tf.variable_scope("Candidate2"):
                c2 = self._activation(
                  core_rnn_cell._linear([session_state, r2 * user_state], self._long_state_size, True,
                          self._bias_initializer, self._kernel_initializer))
            c = array_ops.concat([c1, c2], 1)

            u2 = e*u2 + (1-e)*constant_op.constant(1, dtype=np.float64, shape=u2.get_shape())
            u = array_ops.concat([u1, u2], 1)
            new_h = u * state + (1 - u) * c
            
        return new_h, new_h