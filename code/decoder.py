from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

class Decoder(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def BiLSTM_2layer(self, inputs, masks, scope_name, dropout):
        with tf.variable_scope(scope_name):
            lstm_fw_cell0 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size), output_keep_prob = dropout)
            lstm_bw_cell0 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size), output_keep_prob = dropout)
            lstm_fw_cell1 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size), output_keep_prob = dropout)
            lstm_bw_cell1 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size), output_keep_prob = dropout)
            lstm_fw_cells = [lstm_fw_cell0] + [lstm_fw_cell1]
            lstm_bw_cells = [lstm_bw_cell0] + [lstm_bw_cell1]
            lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(lstm_fw_cells, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(lstm_bw_cells, state_is_tuple=True)
            seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inputs = inputs, sequence_length = seq_len, dtype=tf.float32
            )
            hidden_outputs = tf.concat(outputs, 2)

        return hidden_outputs

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

    # def prediction(self, I3):
    #     xavier_initializer = tf.contrib.layers.xavier_initializer()
    #     with tf.variable_scope('output_layer'):
    #         w_s_f = tf.get_variable('w_start_f', shape=(2 * self.hidden_size, 1),
    #                                 initializer=xavier_initializer)
    #         w_e_f = tf.get_variable('w_end_f', shape=(2 * self.hidden_size, 1),
    #                                 initializer=xavier_initializer)
    #         w_s_b = tf.get_variable('w_start_b', shape=(2 * self.hidden_size, 1),
    #                                 initializer=xavier_initializer)
    #         w_e_b = tf.get_variable('w_end_b', shape=(2 * self.hidden_size, 1),
    #                                 initializer=xavier_initializer)
    #         w_h_f = tf.get_variable('w_hidden_f', shape=(self.hidden_size, 1),
    #                                 initializer=xavier_initializer)
    #         w_h_b = tf.get_variable('w_hidden_b', shape=(self.hidden_size, 1),
    #                                 initializer=xavier_initializer)
    #         self.batch_size = tf.shape(I3)[0]
    #         w_s_f = tf.tile(tf.expand_dims(w_s_f, 0), [self.batch_size, 1, 1]) # (?, 2h, 1)
    #         w_e_f = tf.tile(tf.expand_dims(w_e_f, 0), [self.batch_size, 1, 1])
    #         w_s_b = tf.tile(tf.expand_dims(w_s_b, 0), [self.batch_size, 1, 1])
    #         w_e_b = tf.tile(tf.expand_dims(w_e_b, 0), [self.batch_size, 1, 1])
    #         w_h_f = tf.tile(tf.expand_dims(w_h_f, 0), [self.batch_size, 1, 1])
    #         w_h_b = tf.tile(tf.expand_dims(w_h_b, 0), [self.batch_size, 1, 1]) # (?, h, 1)

    #         pred_s_f = tf.nn.softmax(tf.squeeze(tf.matmul(I3, w_s_f), 2)) # (?, m)
    #         pred_e_b = tf.nn.softmax(tf.squeeze(tf.matmul(I3, w_e_b), 2)) # (?, m)

    #         with tf.variable_scope('forward'):
    #             lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
    #             lstm_fw_input = tf.squeeze(tf.matmul(tf.transpose(I3, perm=(0,2,1)), tf.expand_dims(pred_s_f, 2)), 2) # (?, 2h)
    #             hidden_fw, _ = tf.contrib.rnn.static_rnn(lstm_fw_cell, [lstm_fw_input], dtype=tf.float32) # (?, h)
    #             hidden_fw = tf.expand_dims(hidden_fw[0], 1) # (?, 1, h)

    #         with tf.variable_scope('backward'):
    #             lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
    #             lstm_bw_input = tf.squeeze(tf.matmul(tf.transpose(I3, perm=(0,2,1)), tf.expand_dims(pred_e_b, 2)), 2) # (?, 2h)
    #             hidden_bw, _ = tf.contrib.rnn.static_rnn(lstm_bw_cell, [lstm_bw_input], dtype=tf.float32)
    #             hidden_bw = tf.expand_dims(hidden_bw[0], 1) # (?, 1, h)

    #         pred_e_f = tf.nn.softmax(tf.squeeze(tf.matmul(I3, w_e_f), 2) + tf.squeeze(tf.matmul(hidden_fw, w_h_f), 2)) # (?, m)
    #         pred_s_b = tf.nn.softmax(tf.squeeze(tf.matmul(I3, w_s_b), 2) + tf.squeeze(tf.matmul(hidden_bw, w_h_b), 2)) # (?, m)

    #         pred_e_f = tf.nn.softmax(tf.squeeze(tf.matmul(I3, w_e_f), 2)) # (?, m)
    #         pred_s_b = tf.nn.softmax(tf.squeeze(tf.matmul(I3, w_s_b), 2)) # (?, m)

    #         pred_s = (pred_s_f + pred_s_b) / 2
    #         pred_e = (pred_e_f + pred_e_b) / 2

    #         return pred_s, pred_e

    def prediction(self, I3, context_mask, scope_name, dropout):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope_name):
            w_s = tf.get_variable('w_start_f', shape=(2 * self.hidden_size, 1),
                                    initializer=xavier_initializer)
            w_e = tf.get_variable('w_end_f', shape=(2 * self.hidden_size, 1),
                                    initializer=xavier_initializer)

            self.batch_size = tf.shape(I3)[0]
            w_s = tf.tile(tf.expand_dims(w_s, 0), [self.batch_size, 1, 1]) # (?, 2h, 1)
            w_e = tf.tile(tf.expand_dims(w_e, 0), [self.batch_size, 1, 1])

            I3_fw = self.BiLSTM(I3, context_mask, scope_name, dropout)

            pred_s = tf.nn.softmax(tf.squeeze(tf.matmul(I3, w_s), 2)) # (?, m)
            pred_e = tf.nn.softmax(tf.squeeze(tf.matmul(I3_fw, w_e), 2)) # (?, m)

            return pred_s, pred_e

    def decode(self, I, context_mask, dropout):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        # I3 = self.Res_BiLSTM(I, context_mask, 'inference_layer', dropout) # (?, m, 2h)
        I3 = self.BiLSTM(I, context_mask, 'inference_layer', dropout)
        pred_s, pred_e = self.prediction(I3, context_mask, 'predict_layer', dropout)

        return pred_s, pred_e
