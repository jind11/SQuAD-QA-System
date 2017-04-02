from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from util import Progbar, minibatches

logging.basicConfig(level=logging.INFO)
# tf.logging.set_verbosity(tf.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class QASystem(object):
    def __init__(self, encoder, decoder, flags, embeddings, rev_vocab):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.max_context_len = flags.max_context_len
        self.max_question_len = flags.max_question_len
        self.embedding_size = flags.embedding_size
        self.pretrained_embeddings = embeddings
        self.encoder = encoder
        self.decoder =decoder
        self.n_epochs = flags.epochs
        self.rev_vocab = rev_vocab
        self.model_name = flags.model_name
        self.batch_size = flags.batch_size
        self.train_loss_log = flags.train_dir + "/" + "train_loss.csv"
        self.val_loss_log = flags.train_dir + "/" + "val_loss.csv"
        self.base_lr = flags.base_lr
        self.max_grad_norm = flags.max_grad_norm
        self.dropout = flags.dropout
        self.decay_number = flags.decay_number
        self.decay_rate = flags.decay_rate
        self.summary_dir = flags.summary_dir
        self.summary_flag = flags.summary_flag


        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_context_len))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.max_context_len))
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_question_len))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.max_question_len))
        self.ans_start_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_context_len))
        self.ans_end_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_context_len))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.global_batch_num_placeholder = tf.placeholder(tf.int32, shape=(None))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            context_embeddings, question_embeddings = self.setup_embeddings()
            self.pred_s, self.pred_e = self.setup_prediction(context_embeddings, question_embeddings)
            self.loss = self.setup_loss(self.pred_s, self.pred_e)

        # ==== set up training/updating procedure ====
            # computing learning rates
            self.learning_rate = tf.train.exponential_decay(
              self.base_lr,                               # Base learning rate.
              self.global_batch_num_placeholder,                      # Current total batch number
              self.decay_number,                          # decay every decay_number batch
              self.decay_rate,                            # Decay rate
              staircase = True)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_vars = optimizer.compute_gradients(self.loss)
            grads = list(map(lambda x: x[0], self.grads_vars))
            vars_ = list(map(lambda x: x[1], self.grads_vars))
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)

            self.grads_vars_clip = zip(grads, vars_)
            self.train_op = optimizer.apply_gradients(self.grads_vars_clip)
            if self.summary_flag:
                tf.summary.scalar('cross_entropy', self.loss)
                self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def create_feed_dict(self, data_batch, dropout=0.5, global_batch_num=0):
        feed_dict = {
            self.context_placeholder: data_batch[0],
            self.context_mask_placeholder: data_batch[1],
            self.question_placeholder: data_batch[2],
            self.question_mask_placeholder: data_batch[3],
            self.dropout_placeholder: dropout,
            self.global_batch_num_placeholder: global_batch_num
        }
        if len(data_batch) == 7:
            feed_dict[self.ans_start_placeholder] = data_batch[4]
            feed_dict[self.ans_end_placeholder] = data_batch[5]

        return feed_dict

    def setup_prediction(self, context_embeddings, question_embeddings):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        G = self.encoder.encode(context_embeddings, self.context_mask_placeholder, 
                            question_embeddings, self.question_mask_placeholder, self.dropout_placeholder)
        pred_s, pred_e = self.decoder.decode(G, self.context_mask_placeholder, self.dropout_placeholder)

        return pred_s, pred_e


    def setup_loss(self, pred_s, pred_e):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            # masked_pred_s = tf.add(pred_s, (1 - tf.cast(self.context_mask_placeholder, 'float')) * (-1e30))
            # masked_pred_e = tf.add(pred_e, (1 - tf.cast(self.context_mask_placeholder, 'float')) * (-1e30))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.ans_start_placeholder, logits=pred_s)) \
            + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.ans_end_placeholder, logits=pred_e))

        return loss

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            vec_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings, trainable=False)
            context_embeddings = tf.nn.embedding_lookup(vec_embeddings, self.context_placeholder)
            question_embeddings = tf.nn.embedding_lookup(vec_embeddings, self.question_placeholder)
            context_embeddings = tf.reshape(context_embeddings,
                    (-1, self.max_context_len, self.embedding_size))
            question_embeddings = tf.reshape(question_embeddings,
                    (-1, self.max_question_len, self.embedding_size))

        return context_embeddings, question_embeddings

    def optimize(self, session, train_batch, global_batch_num):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = self.create_feed_dict(train_batch, 1 - self.dropout, global_batch_num)

        if self.summary_flag:
            output_feed = [self.train_op, self.loss, self.learning_rate, self.merged]
            _, train_loss, current_lr, summary = session.run(output_feed, input_feed)
        else:
            output_feed = [self.train_op, self.loss, self.learning_rate]
            _, train_loss, current_lr = session.run(output_feed, input_feed)
            summary = None

        return train_loss, current_lr, summary

    def run_epoch(self, sess, train_data, val_data, epoch_num, train_log):
        num_batches = int(len(train_data) / self.batch_size) + 1
        logging.info("Evaluating on training data")
        prog = Progbar(target = num_batches)
        for i, batch in enumerate(minibatches(train_data, self.batch_size)):
            global_batch_num = int(epoch_num * num_batches + i)
            loss, current_lr, summary = self.optimize(sess, batch, global_batch_num)
            prog.update(i + 1, [("train loss", loss), ("current LR", current_lr)])
            train_log.write("{},{}\n".format(epoch_num + 1, loss))
            if self.summary_flag:
                self.train_writer.add_summary(summary, i)
        print("")

        logging.info("Evaluating on development data")
        val_loss = self.validate(sess, val_data)

        return val_loss

    def test(self, session, valid_batch):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = self.create_feed_dict(valid_batch, dropout=1)

        output_feed = self.loss

        loss = session.run(output_feed, input_feed)

        return loss

    def decode(self, session, test_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = self.create_feed_dict(test_batch[0:4], dropout=1)

        output_feed = [self.pred_s, self.pred_e]

        pred_s, pred_e = session.run(output_feed, input_feed)

        return pred_s, pred_e

    def answer(self, session, test_batch):

        p_s, p_e = self.decode(session, test_batch)

        a_s = np.argmax(p_s, axis=1)
        a_e = np.argmax(p_e, axis=1)

        return (a_s, a_e)

    def validate(self, sess, val_data):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        val_cost = 0
        num_batches = int(len(val_data) / self.batch_size) + 1
        prog = Progbar(target = num_batches)
        for i, batch in enumerate(minibatches(val_data, self.batch_size)):
            loss = self.test(sess, batch)
            prog.update(i + 1, [("val loss", loss)])
            val_cost += loss
        print("")
        val_cost /= i + 1

        return val_cost

    def formulate_answer(self, context, vocab, start, end):
        ans = ''
        for index in xrange(start, end+1):
            if index < len(context):
                ans += vocab[context[index]]
                ans += ' '

        return ans

    def evaluate_answer(self, session, data, rev_vocab, sample_num=200):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :return:
        """

        overall_f1 = 0.
        overall_em = 0.

        eval_batch = [data[i] for i in np.random.choice(len(data), sample_num, replace=False)]
        eval_batch = list(zip(*eval_batch)) # unzip the list

        a_s_vec, a_e_vec = self.answer(session, eval_batch)
        for (a_s, a_e, context, a_true) in zip(a_s_vec, a_e_vec, eval_batch[0], eval_batch[6]):
            if a_s > a_e:
                tmp = a_s
                a_s = a_e
                a_e = tmp
            predicted_answer = self.formulate_answer(context, rev_vocab, a_s, a_e)
            true_answer = self.formulate_answer(context, rev_vocab, a_true[0], a_true[1])
            f1 = f1_score(predicted_answer, true_answer)
            overall_f1 += f1
            if exact_match_score(predicted_answer, true_answer):
                overall_em += 1

        average_f1 = overall_f1 / sample_num
        overall_em /= sample_num
        # logging.info("F1: {}, EM: {}, for {} samples\n".format(average_f1, overall_em, sample_num))

        return average_f1, overall_em

    def train(self, session, train_data, val_data, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        best_score = 100000
        train_log = open(self.train_loss_log, "w")
        val_log = open(self.val_loss_log, "w")

        if self.summary_flag:
            self.train_writer = tf.summary.FileWriter(self.summary_dir + '/train', session.graph)

        for epoch in range(self.n_epochs):
            logging.info("\nEpoch %d out of %d", epoch + 1, self.n_epochs)
            val_score = self.run_epoch(session, train_data, val_data, epoch, train_log)
            logging.info("Average Dev Cost: {}".format(val_score))
            val_log.write("{},{}\n".format(epoch + 1, val_score))
            train_f1, train_em = self.evaluate_answer(session, train_data, self.rev_vocab)
            logging.info("train F1 {} & EM {}".format(train_f1, train_em))
            val_f1, val_em = self.evaluate_answer(session, val_data, self.rev_vocab)
            logging.info("Val F1 {} & EM {}".format(val_f1, val_em))
            if val_score < best_score:
                best_score = val_score
                print("New best dev score! Saving model in {}".format(train_dir + "/" + self.model_name))
                self.saver.save(session, train_dir + self.model_name, global_step=epoch)

        return best_score
