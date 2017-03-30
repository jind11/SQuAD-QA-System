from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

class Encoder(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def BiLSTM(self, inputs, masks, scope_name, dropout):
        with tf.variable_scope(scope_name):
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size), output_keep_prob = dropout)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size), output_keep_prob = dropout)
            seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inputs = inputs, sequence_length = seq_len, dtype=tf.float32
            )
            hidden_outputs = tf.concat(outputs, 2)

        return hidden_outputs

    def similarity(self, question_rep, context_rep):
        return tf.matmul(question_rep, tf.transpose(context_rep, perm=(0,2,1))) # (?, n, 2h) * (?, 2h, m) = (?, n, m)

    def Qcode(self, align_mat, question_rep):
        # align_mat: (?, n, m)
        # question_rep: (?, n, 2h)
        a = tf.nn.softmax(align_mat, dim=1)
        return tf.matmul(tf.transpose(a, perm=(0,2,1)), question_rep) # (?, m, n) * (?, n, 2h) = (?, m, 2h)

    def Ques_filtering(self, align_mat, context_rep):
        b = tf.nn.softmax(align_mat, dim=-1) # (?, n, m)
        bf_max_pool = tf.reduce_max(b, axis=1) # (?, m)
        bf_max = bf_max_pool / tf.reduce_sum(bf_max_pool, axis=1, keep_dims=True) # (?, m)
        bf_max = tf.expand_dims(bf_max, axis=2) # (?, m, 1)
        bf_mean_pool = tf.reduce_mean(b, axis=1) # (?, m)
        bf_mean = bf_mean_pool / tf.reduce_sum(bf_mean_pool, axis=1, keep_dims=True) # (?, m)
        bf_mean = tf.expand_dims(bf_mean, axis=2) # (?, m, 1)

        Df_max = tf.multiply(bf_max, context_rep) # (?, m, 2h)
        Df_mean = tf.multiply(bf_mean, context_rep) # (?, m, 2h)
        Df = tf.concat([Df_max, Df_mean], 2) # (?, m, 4h)

        return Df, bf_max, bf_mean

    def alignment(self, context_rep, question_rep):
        align_mat = self.similarity(question_rep, context_rep)
        Qw = self.Qcode(align_mat, question_rep)
        Df, bf_max, bf_mean = self.Ques_filtering(align_mat, context_rep)

        return tf.concat([context_rep, Qw, tf.multiply(context_rep, Qw), context_rep - Qw, Df, bf_max, bf_mean], 2) # (?, m, 12h+2)

    def encode(self, context, context_mask, question, question_mask, dropout):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        context_rep = self.BiLSTM(context, context_mask, 'context-BiLSTM', dropout)  # (?, m, 2h)
        question_rep = self.BiLSTM(question, question_mask, 'quesition-BiLSTM', dropout) # (?, n, 2h)
        I = self.alignment(context_rep, question_rep)

        return I
