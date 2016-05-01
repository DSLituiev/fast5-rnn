#!/usr/bin/env python3
# coding: utf-8
from __future__ import division, print_function
#from six import xrange
import tensorflow as tf
from tensorflow.python.ops import functional_ops
linear = tf.nn.rnn_cell.linear

flags = tf.app.flags
FLAGS = flags.FLAGS

# define flags (note that Fomoro will not pass any flags by default)
flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

from itertools import cycle
import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('[ %(levelname)-2s\t:%(funcName)s]\t%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
DEBUG = True
if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import functional_ops
from dnahashing import get_transition_matrix, get_rev_transition_matrix, hash_window_mapping, dnaunhash

# define artifact directories where results from the session can be saved
model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logdir/')

DATADIR = "./dataset"
#SUMMARY_FILE = "dataset/all_equal_len.h5"
SUMMARY_FILE = "one_big_run.h5"

def orthogonal(shape, scale=1.1, name=None):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.variable(scale * q[:shape[0], :shape[1]], name=name)

# ### Generating Inputs and Targets

# First let's write a function for generating input, target sequences, one pair at a time. We'll limit ourselves to inputs with independent time steps, drawn from a standard normal distribution.
# 
# Each sequence is 2-D with time on the first axis and inputs or targets on the second. (This way it'd be easy to generalize to the case of multiple inputs/targets per time step.)

# ### Defining the RNN Model from Scratch

# Next let's define the RNN model. The code is a bit verbose because it's meant to be self explanatory, but the pieces are simple:
# - The update for the vanilla RNN is $h_t = \tanh( W_h h_{t-1} + W_x x_t + b )$.
# - `_vanilla_rnn_step` is the core of the vanilla RNN: it applies this update by taking in a previous hidden state along with a current input and producing a new hidden state. (The only difference below is that both sides of the equation are transposed, and each variable is replaced with its transpose.)
# - `_compute_predictions` applies `_vanilla_rnn_step` to all time steps using `scan`, resulting in hidden states for each time step, and then applies a final linear layer to each state to yield final predictions.
# - `_compute_loss` just computes the mean squared Euclidean distance between the ground-truth targets and our predictions.

class Model(object):
    def __init__(self, hidden_layer_size, input_size, target_size, seqlen=None, 
            init_scale=0.1, test = False, emission_init = None):
        """ Create a RNN.
        Args:
            hidden_layer_size: An integer. The number of hidden units.
            input_size: An integer. The number of inputs per time step.
            target_size: An integer. The number of targets per time step.
            init_scale: A float. All weight matrices will be initialized using
                a uniform distribution over [-init_scale, init_scale].
        """

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.target_size = target_size
        self.init_scale = init_scale
        self._num_units = 1
        self.emission_init = emission_init

        self._inputs = tf.placeholder(tf.float32, shape=[None,  input_size],
                                      name='inputs')
        self._targets = tf.placeholder(tf.float32, shape=[None, target_size],
                                       name='targets')

        fro = 3
        to = 6
        self.nt_in_pore = to - fro
        self.map_hex_to_pore = tf.Variable(
                hash_window_mapping(to=to, fro=fro, n_input_states=self.hidden_layer_size).astype(np.float32).T,
                trainable = False)
        #self.similarity_kernel = np.corrcoef( self.map_hex_to_pore.T ).astype(np.float32)

        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('model', initializer=initializer):
            self._states, self._predictions = self._compute_predictions()
            self._loss = self._compute_loss()

    def _rnn_step_fw(self, h_prev, x,):
        """
        ! Requires a shift between observed and hidden states !
        or maybe not, because of the hexamer state space.
        Computes forward propagation assuming some initial prior:
        h_raw[2] = T @ ( h[1]  *  E x[1])      =
                 = sum_{Z[1]} ( P(Z[2] | Z[1]) *  P( Z[1] )  *   P(x[1] | Z[1]) *  )
                 = P( Z[2] , x[1] )
        h[2] = h_raw[2] / sum(h_raw[2])           = P( Z[2] | x[1] )

        h_raw[3] = T @ ( h[2]  *  E x[2])      =
                 = sum_{Z[3]} ( P(Z[3] | Z[2]) *  P( Z[2] | x[1] )  *   P(x[2] | Z[2]) *  )
                 = P( Z[3] , x[2] | x[1] )
        h[3] = h_raw[3] / sum(h_raw[3])           = P( Z[3] | x[1], x[2] )
        ...
        """
        #x = tf.reshape(x, [1, self.input_size])
        sqrt2pi = np.sqrt(2*np.pi)

        with tf.variable_scope("HMM"):
            with tf.variable_scope("emission"):
                #log_x_inv_var_emitted = tf.get_variable("b_emit", shape = [1, self.input_size ])
                log_x_inv_var_emitted = tf.Variable( 1.0*np.ones([self.hidden_layer_size, 1], dtype=np.float32),
                                                    name="b_emit",
                                                    trainable = False)
                #shrink_states = tf.Variable( hash_window_mapping(to=to, fro=fro),
                #                            name = "W_bottleneck",  )
                "W_emit: [self.input_size, self.hidden_layer_size]"
                #x_inv_var_emitted = tf.exp(log_x_inv_var_emitted)
                x_inv_var_emitted = tf.exp(log_x_inv_var_emitted)
                print( "self.W_emit", self.W_emit.get_shape() )
                print( "x_inv_var_emitted", x_inv_var_emitted.get_shape() )
                loss_x = 0.5 * ((x - self.W_emit) * x_inv_var_emitted )**2
                "this should enforce probabilities integrate to one over `x`"
                P_x_given_z = tf.reshape(
                                 tf.reduce_sum(
                                     tf.exp( 0.5 * log_x_inv_var_emitted)/sqrt2pi * tf.exp( -loss_x ),
                                     reduction_indices = 1
                                              ),
                              [self.hidden_layer_size],
                                        )
                P_x_given_z = P_x_given_z / tf.reduce_sum(P_x_given_z)
                #guess_emission = tf.argmax(P_x_given_z)
                #P_x_given_z = tf.reshape(P_x_given_z, [self.hidden_layer_size, 1], name = "P_prev_x_given_z")

                #states_fw_summary = tf.histogram_summary("states_fw", states)

        print("h_prev", h_prev.get_shape())
        z_prev_cond_all_prev_x = tf.reshape(tf.mul( h_prev , P_x_given_z ),  [self.hidden_layer_size, 1], name="z_prev_cond_all_prev_x")
        h =  tf.reshape( tf.matmul(self.W_trans, z_prev_cond_all_prev_x),  [self.hidden_layer_size], name='h')
        #print("h_raw:", h_raw.get_shape())
        #h_raw = tf.reshape(h_raw, [self.hidden_layer_size], name='h')
        h_sum = tf.reduce_sum(h)
        h = h / h_sum
        h = tf.reshape( tf.nn.softmax(tf.reshape(h, [1, self.hidden_layer_size])), [self.hidden_layer_size])
        #h = tf.reshape(h, [self.hidden_layer_size], name='h')
        print("states h:", h.get_shape())
        return h

    def _compute_predictions(self, init = None):
        """ Compute vanilla-RNN states and predictions. """

        with tf.variable_scope('states'):
            with tf.variable_scope("HMM"):
                with tf.variable_scope("transition"):
                    skip_prob = tf.get_variable("skip", shape=[1], initializer=tf.constant_initializer(1e-1))
                    #skip_prob = tf.Variable( np.array(1e-1, dtype=np.float32), name="skip") # .astype(np.float32)
                    self.W_trans = (1-skip_prob) * get_transition_matrix().astype(np.float32)  + skip_prob* np.eye(self.hidden_layer_size).astype(np.float32)
                    #self.W_trans = tf.Variable( transition_with_skips,
                    #                       name='W_trans', trainable=True)
                    print("W_trans", self.W_trans.get_shape())

                with tf.variable_scope("emission"):
                    "W_emit: [self.input_size, self.hidden_layer_size]"
                    if self.emission_init is None:
                        self.W_emit = tf.get_variable("W_emit", shape = [self.hidden_layer_size, self.input_size],
                                                  initializer = tf.random_normal_initializer(0.0, 1e-6))
                    else:
                        if not (self.emission_init.shape == (self.hidden_layer_size, self.input_size)):
                            print("self.emission_init.shape", self.emission_init.shape)
                            print("(self.hidden_layer_size, self.input_size)", (self.hidden_layer_size, self.input_size))
                            raise ValueError("wrong dimensions of  `self.emission_init`")
                        self.W_emit = tf.Variable(self.emission_init.astype(np.float32), name = "W_emit", trainable = False)
                    self.W_emit_summary = tf.image_summary("W_emit", tf.reshape(self.W_emit, [1,self.hidden_layer_size, self.input_size,1]))
                    "idea: impose kernel similarity:  maximize(W K W)"
                    "[ self.hidden_layer_size, self.nt_in_pore ]"

                    emission_in_pore_space = tf.matmul( self.map_hex_to_pore, self.W_emit)
                    self.emission_similarity = tf.reduce_sum( tf.diag_part( tf.matmul( tf.transpose(emission_in_pore_space),(emission_in_pore_space)) ),
                            name="emission_w_similarity")
            if init is None:
                initial_state = tf.ones([self.hidden_layer_size],
                                     name='initial_state')
                initial_state = initial_state/ self.hidden_layer_size
            else:
                initial_state = init
            #states = self._rnn_step_fw(initial_state[:,0], self.inputs[0,:])
            states = functional_ops.scan(self._rnn_step_fw, tf.identity(self.inputs),
                                         initializer=initial_state, name='states')

            states_fw_summary = tf.histogram_summary("states_fw", states)
            #states = states_fw
            #print("states:", states.get_shape())

        with tf.variable_scope('predictions'):
            # set some explicit initializer, orthogonal inialization
            "for now, keep identity mapping from hidden states to labels"
            "assume probability interpretation of values: should sum to one"
            W_pred = tf.Variable(np.eye(self.target_size, dtype = np.float32), name="W_pred", trainable=False)
            predictions = tf.matmul(states, W_pred, name='predictions')
            #predictions = states
            predictions_summary = tf.histogram_summary("predictions", predictions)
            #predictions = tf.nn.softmax(tf.matmul(states, W_pred), name='predictions'))
            # do predictions sum to one?

        return states, predictions

    def _compute_loss(self):
        """ Compute l2 loss between targets and predictions. """
        logging.debug( "self.predictions\t%s" % repr(self.predictions) )
        logging.debug( "self.targets\t%s"     % repr(self.targets)     )
        with tf.variable_scope('loss'):
            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predictions, self.targets))
            #loss = tf.reduce_mean((self.targets - self.predictions)**2, name='loss')
            #self.result_loss = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(self.predictions), reduction_indices=[1]))
            #eye = tf.diag(tf.ones(self.hidden_layer_size))
            #a = tf.matmul( self.targets,     eye)
            #b = tf.matmul( self.predictions, eye)
            #a = tf.matmul( self.targets,     tf.transpose(self.map_hex_to_pore) )
            #b = tf.matmul( self.predictions, tf.transpose(self.map_hex_to_pore) )
            #self.result_loss = -tf.log( tf.reduce_sum( a*b ), name="result_loss" )
            self.result_loss = -tf.log( tf.reduce_sum(self.targets * self.predictions), name="result_loss")
            print("result_loss", self.result_loss.get_shape())
            self.emission_penalty = 0# - 1e-3* tf.log(self.emission_similarity)
            loss = self.result_loss + self.emission_penalty
            return loss

    @property
    def inputs(self):
        """ A 2-D float32 placeholder with shape `[dynamic_duration, input_size]`. """
        return self._inputs

    @property
    def targets(self):
        """ A 2-D float32 placeholder with shape `[dynamic_duration, target_size]`. """
        return self._targets

    @property
    def states(self):
        """ A 2-D float32 Tensor with shape `[dynamic_duration, hidden_layer_size]`. """
        return self._states

    @property
    def predictions(self):
        """ A 2-D float32 Tensor with shape `[dynamic_duration, target_size]`. """
        return self._predictions

    @property
    def loss(self):
        """ A 0-D float32 Tensor. """
        return self._loss

### Defining an Optimizer

# Next let's write an optimizer class. We'll use vanilla gradient descent after gradient "clipping," according to the method described by [Pascanu, Mikolov, and Bengio](http://arxiv.org/abs/1211.5063).
# 
# The gradient-clipping method is simple and could instead be called gradient scaling: if the global norm is smaller than `max_global_norm`, do nothing. Otherwise, rescale all gradients so that the global norm becomes `max_global_norm`.
# 
# What is the global norm? It's just the norm over *all* gradients, as if they were concatenated together to form one global vector.

class Optimizer(object):
    def __init__(self, loss, initial_learning_rate, num_steps_per_decay,
                 decay_rate, max_global_norm=1.0, scope = None):
        """ Create a simple optimizer.
            This optimizer clips gradients and uses vanilla stochastic gradient
        descent with a learning rate that decays exponentially.
            Args:
            loss: A 0-D float32 Tensor.
            initial_learning_rate: A float.
            num_steps_per_decay: An integer.
            decay_rate: A float. The factor applied to the learning rate
                every `num_steps_per_decay` steps.
            max_global_norm: A float. If the global gradient norm is less than
                this, do nothing. Otherwise, rescale all gradients so that
                the global norm becomes `max_global_norm`.
        """

        if scope in (None, ""):
            trainables = tf.trainable_variables()
            logging.info("trainable variables:\t%s" % ( ["%s" % repr(x.name) for x in trainables] ) )
        else:
            trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            #trainables = [trainables[-1]]
            logging.info("trainable variables in scope '%s':\n%s" % (scope, "\n".join(["%s:\t%s" % (repr(x.name), repr(x.get_shape())) for x in trainables]) ) )

        print("loss", loss.get_shape(), loss)
        grads = tf.gradients(loss, trainables, )
        print("grads", grads)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)

        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, num_steps_per_decay,
            decay_rate, staircase=True)
        #Adam, RMSprop, AdaGrad
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self._optimize_op = optimizer.apply_gradients(grad_var_pairs,
                                                      global_step=global_step)

    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op


# ### Training

# Next let's define and run our training function. This is where we'll run the main optimization loop and export TensorBoard summaries.

# In[7]:

def train(sess, model, optimizers, generator, num_optimization_steps,
          logdir='./logdir', checkpoint_interval = 100):
    """ Train.

    Args:
        sess: A Session.
        model: A Model.
        optimizer: A List of  Optimizers.
        generator: A generator that yields `(inputs, targets)` tuples, with
            `inputs` and `targets` both having shape `[dynamic_duration, 1]`.
        num_optimization_steps: An integer.
        logdir: A string. The log directory.
    """

    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    tf.scalar_summary('loss', model.loss)

    ema = tf.train.ExponentialMovingAverage(decay=0.75)
    update_loss_ema = ema.apply([model.loss])
    loss_ema = ema.average(model.loss)
    tf.scalar_summary('loss_ema', loss_ema)

    tf.scalar_summary('result_loss', model.result_loss)
    tf.scalar_summary("emission_penalty", model.emission_penalty )
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(logdir=logdir, graph=sess.graph)

    # create a saver instance to restore from the checkpoint
    saver = tf.train.Saver(max_to_keep=1)

    sess.run(tf.initialize_all_variables())

    # save the graph definition as a protobuf file
    tf.train.write_graph(sess.graph_def, model_path, 'convnet.pb', as_text=False)

    # restore variables
    if FLAGS.restore:
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path:
            saver.restore(sess, latest_checkpoint_path)

    for step in range(num_optimization_steps):
        inputs, targets, = next(generator)
        "TRANSITION"
        "EMISSION"
        for name, optimizer in optimizers:
            loss_ema_, summary, _, _ = sess.run(
                [loss_ema, summary_op, optimizer.optimize_op, update_loss_ema],
                {model.inputs: inputs, model.targets: targets})

        #loss_ema_, summary, _, _ = sess.run(
        #    [loss_ema, summary_op, optimizer.optimize_op, update_loss_ema],
        #    {model.inputs: inputs, model.targets: targets})

        summary_writer.add_summary(summary, global_step=step)
        #print('\rStep %d. Loss EMA: %.6f.' % (step+1, loss_ema_), end='')
        print('Step %d. Loss EMA: %.6f.\n' % (step+1, loss_ema_), end='')
        if step % checkpoint_interval == 0:
            "cheating"
            val_features, val_targets = next(generator)
            """
            validation_accuracy, summary = sess.run([accuracy, merged_summaries], feed_dict={
                model.inputs:  val_features,
                y_: val_targets,
                keep_prob: 1.0
            })
            """
            #states_, prediction_ = model._compute_predictions()
            #prediction_ = model.predictions

            prediction, W_emit_summary  = \
                sess.run( [ model.predictions, model.W_emit_summary, ],
                          feed_dict = {model.inputs: val_features, model.targets: val_targets}
                        )

            #prediction = prediction_.eval( feed_dict = {model.inputs: val_features, model.targets: val_targets}, session = sess )
            #W_emit_summary = model.W_emit_summary.eval(feed_dict = {model.inputs: val_features, model.targets: val_targets}, session = sess)
            logging.info( "prediction:\t" + "\t".join([ "%s" % dnaunhash(x) for x in np.argmax( prediction,  1)]) )
            #logging.info( "val_targets\t%s" + repr( np.argmax( val_targets, 1).ravel()[0]) )
            logging.info( "val_targets\t" + "\t".join([ "%s" % dnaunhash(x) for x in np.argmax( val_targets, 1) ]) )

            summary_writer.add_summary(summary, step, )
            summary_writer.add_summary(W_emit_summary, step, )
            saver.save(sess, checkpoint_path + 'checkpoint', global_step=step)


# Now we can train our model:

#generator = input_target_generator()
target_size = 4**6
input_size = 3
hidden_layer_size = target_size
#from get_data import get_data_from_summary_file
#from get_data import get_batch_chunks_from_summary_file
from get_data import get_single_chunks_from_summary_file

#generator = get_data( DATADIR, batch_size = 1 )
#generator = get_data_from_summary_file(SUMMARY_FILE)
import pandas as pd
emission_init = pd.read_csv("emission_init.csv", index_col=0).as_matrix()
print("emission_init", emission_init.shape)

chunk_length = 8
batch_size = 1
generator = cycle(get_single_chunks_from_summary_file(SUMMARY_FILE, chunk_length=chunk_length))

model = Model(hidden_layer_size=hidden_layer_size, input_size=input_size,
              target_size=target_size, init_scale=1e-4, emission_init = emission_init)

sess = tf.Session()
optimizer = Optimizer(model.loss, initial_learning_rate=1e-4, num_steps_per_decay=15000,
                      decay_rate=0.1, max_global_norm=5.0, scope="")
#train(sess, model, optimizer,
#                    generator, num_optimization_steps=5000, checkpoint_interval = 20)

#transition_optimizer = Optimizer(model.loss, initial_learning_rate=1e-4, num_steps_per_decay=15000,
#                              decay_rate=0.1, max_global_norm=5.0, scope="model/states")


optim_list = [#("transition", transition_optimizer),
                        ("emission", optimizer)]

train(sess, model, optim_list,
      generator, num_optimization_steps=5000, checkpoint_interval = 1)

# After running `tensorboard --logdir ./logdir` and navigating to [http://localhost:6006](http://localhost:6006), we can view our loss summaries. Here the exponential moving average is especially helpful because our raw losses correspond to individual sequences (and are therefore very noisy estimates).
