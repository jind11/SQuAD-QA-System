from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

import numpy as np
import tensorflow as tf

from util import variable_summaries

logging.basicConfig(level=logging.INFO)

class Encoder():
    def __init__(self, hidden_size, summary_flag, max_context_len, max_question_len):
        self.hidden_size = hidden_size
        self.summary_flag = summary_flag
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len

    def BiLSTM(self, inputs, masks, scope_name, dropout):
        with tf.variable_scope(scope_name):
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size), output_keep_prob = dropout)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size), output_keep_prob = dropout)
            seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inputs = inputs, sequence_length = seq_len, dtype=tf.float32
            )
            hidden_outputs = tf.transpose(tf.concat(outputs, 2), perm=[0, 2, 1])
        return hidden_outputs

    def bi_attention(self, y_q, y_c):
        # y_q: (?, 2h, n)
        # y_c: (?, 2h, m)
        # need to compute S first
        # S: (?, m, n)
        with tf.variable_scope('bi_attention') as scope:
            S = self.bilinear_similarity(y_q, y_c)
            H = self.Q2C_attention(y_q, y_c, S)  # H = (?, 2h, m)
            U = self.C2Q_attention(y_q, y_c, S)  # U = (?, 2h, m)
            # need to compute G
            G = tf.concat([y_c, H, y_c * U, y_c * H], 1)  # G = (?, 8h, m)
            G = tf.transpose(G, perm=[0, 2, 1]) # (?, m, 8h)
        return G

    def bilinear_similarity(self, y_q, y_c):
        # y_q: (?, 2h, n)
        # y_c: (?, 2h, m)
        # S : (?, n, m)
        with tf.variable_scope('similarity') as scope:
            self.batch_size = tf.shape(y_c)[0]
            w_alpha = tf.get_variable('w_alpha', shape=(2 * self.hidden_size, 2 * self.hidden_size),
                initializer=tf.contrib.layers.xavier_initializer())

            if self.summary_flag:
                variable_summaries(w_alpha, "bilinear_w_alpha")
            w_alpha_tiled = tf.tile(tf.expand_dims(w_alpha, 0), [self.batch_size, 1, 1])
            y_q_T = tf.transpose(y_q, perm=[0, 2, 1]) # U_T: (?, n, 2h)
            bi_S_temp = tf.einsum('aij,ajk->aik', y_q_T, w_alpha_tiled) # (?, n, 2h) * (2h, 2h) = (?, n, 2h)
            S = tf.einsum('aij,ajk->aik', bi_S_temp, y_c)  # (?, n, 2h) * (?, 2h, m) = (?, n, m)
        return S

    def C2Q_attention(self, y_q, y_c, S):
        # y_q: (?, 2h, n)
        # y_c: (?, 2h, m)
        # S: (?, n, m)
        a = tf.nn.softmax(S, dim=1)   # (?, n, m)
        U = tf.matmul(y_q, a)    # (?, 2h, n) * (?, n, m) = (?, 2h, m)
        return U

    def Q2C_attention(self, y_q, y_c, S):
        # y_q: (?, 2h, n)
        # y_c: (?, 2h, m)
        # S: (?, n, m)
        b = tf.nn.softmax(tf.reduce_max(S, axis=1, keep_dims=True)) # b = (?, 1, m)
        h = tf.einsum('aij,akj->aik', y_c, b) # (?, 2h, m) * (?, 1, m) => (?, 2h, 1)
        H = tf.tile(h, [1, 1, self.max_context_len])
        return H

    def _cosine_similarity(self, Q, C):
        # Q = (?, m, h)
        # C = (?, n, h)
        normed_Q = tf.nn.l2_normalize(Q, dim=-1)  # (?, n, h)
        normed_C = tf.nn.l2_normalize(C, dim=-1)  # (?, m, h)
        cosine_sim = tf.matmul(normed_Q, tf.transpose(normed_C, perm=[0, 2, 1]))
        return cosine_sim  # (?, n, m)

    def filter_layer(self, question, context):
        with tf.variable_scope('filter') as scope:
            w_f = tf.get_variable('w_filter', shape=(self.max_question_len, 1),
                initializer=tf.contrib.layers.xavier_initializer())

            if self.summary_flag:
                variable_summaries(w_f, "filter_layer_weights")

            self.batch_size = tf.shape(question)[0]
            w_f_tiled = tf.tile(tf.expand_dims(w_f, 0), [self.batch_size, 1, 1])
            cosine_sim = self._cosine_similarity(question, context)       # (?, n, m)
            relevence = tf.einsum('aij,aik->ajk', cosine_sim, w_f_tiled)  # (?, n, m) * (?, n, 1) => (?, m, 1)

        return context * relevence

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
        context_filt = self.filter_layer(question, context)
        yc = self.BiLSTM(context_filt, context_mask, 'context_BiLSTM', dropout) # (?, 2h, m)
        yq = self.BiLSTM(question, question_mask, 'question_BiLSTM', dropout) # (?, 2h, n)

        return self.bi_attention(yq, yc)
