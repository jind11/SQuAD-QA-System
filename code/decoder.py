from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from util import variable_summaries


class Decoder():
    def __init__(self, hidden_size, summary_flag):
        self.hidden_size = hidden_size
        self.summary_flag = summary_flag

    def BiLSTM(self, inputs, masks, scope_name, dropout):
        with tf.variable_scope(scope_name):
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size), output_keep_prob = dropout)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size), output_keep_prob = dropout)
            seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inputs = inputs, sequence_length = seq_len, dtype=tf.float32
            )
            hidden_outputs = tf.concat(outputs, 2)

        return hidden_outputs

    def output_layer(self, G, M, context_mask,dropout, scope_name):
        # M (?, m, 2h)
        # the softmax part is implemented together with loss function
        with tf.variable_scope(scope_name):
            w_1 = tf.get_variable('w_start', shape=(10 * self.hidden_size, 1),
                initializer=tf.contrib.layers.xavier_initializer())
            w_2 = tf.get_variable('w_end', shape=(10 * self.hidden_size, 1),
                initializer=tf.contrib.layers.xavier_initializer())

            if self.summary_flag:
                variable_summaries(w_1, "output_w_1")
                variable_summaries(w_2, "output_w_2")

            self.batch_size = tf.shape(M)[0]

            # M2 = self.BiLSTM(M, context_mask, scope_name, dropout)

            temp1 = tf.concat([G, M], 2)  # (?, m, 10h)
            temp2 = tf.concat([G, M], 2)  # (?, m, 10h)
            temp_1_o = tf.nn.dropout(temp1, dropout)
            temp_2_o = tf.nn.dropout(temp2, dropout)

            w_1_tiled = tf.tile(tf.expand_dims(w_1, 0), [self.batch_size, 1, 1])
            w_2_tiled = tf.tile(tf.expand_dims(w_2, 0), [self.batch_size, 1, 1])

            pred_s = tf.squeeze(tf.einsum('aij,ajk->aik',temp_1_o, w_1_tiled)) # (?, m, 10h) * (?, 10h, 1) -> (?, m, 1)
            pred_e = tf.squeeze(tf.einsum('aij,ajk->aik',temp_2_o, w_2_tiled)) # (?, m, 10h) * (?, 10h, 1) -> (?, m, 1)
            return pred_s, pred_e

    def decode(self, G, context_mask, dropout):
        M = self.BiLSTM(G, context_mask, 'model_layer',dropout)
        pred_s, pred_e = self.output_layer(G, M, context_mask, dropout, 'output_layer')
        return pred_s, pred_e