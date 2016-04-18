# coding: utf-8
from __future__ import division, print_function
#from six import xrange
import tensorflow as tf
from tensorflow.python.ops import functional_ops
linear = tf.nn.rnn_cell.linear

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

DATADIR = "../data/"
SUMMARY_FILE = "../all_equal_len.h5"
# <a id="generating-inputs-and-targets"></a>
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
    
    def __init__(self, hidden_layer_size, input_size, target_size, init_scale=0.1):
        """ Create a vanilla RNN.
        
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
        
        self._inputs = tf.placeholder(tf.float32, shape=[None, input_size],
                                      name='inputs')
        self._targets = tf.placeholder(tf.float32, shape=[None, target_size],
                                       name='targets')
        
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('model', initializer=initializer):
            self._states, self._predictions = self._compute_predictions()
            self._loss = self._compute_loss()
    
    def _rnn_step(self, h_prev, x,):
        h_prev = tf.reshape(h_prev, [1, self.hidden_layer_size])
        x = tf.reshape(x, [1, self.input_size])

        with tf.variable_scope("GRUCell"):
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not udpate.
                r, u = tf.split(1, 2, linear([x, h_prev],
                                              2 * self._num_units, True, 1.0, scope="Gates"))
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear([x, r * h_prev ], self._num_units, True, scope="Candidate" ))
                h = u * h_prev + (1 - u) * c
                h = tf.reshape(h, [self.hidden_layer_size], name='h')
        return h

    def _compute_predictions(self):
        """ Compute vanilla-RNN states and predictions. """

        with tf.variable_scope('states'):
            initial_state = tf.zeros([self.hidden_layer_size],
                                     name='initial_state')
            states = functional_ops.scan(self._rnn_step, self.inputs,
                                         initializer=initial_state, name='states')

        with tf.variable_scope('predictions'):
            W_pred = tf.get_variable(
                'W_pred', shape=[self.hidden_layer_size, self.target_size])
            b_pred = tf.get_variable('b_pred', shape=[self.target_size],
                                     initializer=tf.constant_initializer(0.0))
            predictions = tf.add(tf.matmul(states, W_pred), b_pred, name='predictions')
            
        return states, predictions

    def _compute_loss(self):
        """ Compute l2 loss between targets and predictions. """

        with tf.variable_scope('loss'):
            loss = tf.reduce_mean((self.targets - self.predictions)**2, name='loss')
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
    

# In[5]:
# ### Defining an Optimizer

# Next let's write an optimizer class. We'll use vanilla gradient descent after gradient "clipping," according to the method described by [Pascanu, Mikolov, and Bengio](http://arxiv.org/abs/1211.5063).
# 
# The gradient-clipping method is simple and could instead be called gradient scaling: if the global norm is smaller than `max_global_norm`, do nothing. Otherwise, rescale all gradients so that the global norm becomes `max_global_norm`.
# 
# What is the global norm? It's just the norm over *all* gradients, as if they were concatenated together to form one global vector.

# In[6]:

class Optimizer(object):
    
    def __init__(self, loss, initial_learning_rate, num_steps_per_decay,
                 decay_rate, max_global_norm=1.0):
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
                the global norm because `max_global_norm`.
        """
        
        trainables = tf.trainable_variables()
        grads = tf.gradients(loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)
        
        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, num_steps_per_decay,
            decay_rate, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._optimize_op = optimizer.apply_gradients(grad_var_pairs,
                                                      global_step=global_step)
    
    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op


# <a id="training"></a>
# ### Training

# Next let's define and run our training function. This is where we'll run the main optimization loop and export TensorBoard summaries.

# In[7]:

def train(sess, model, optimizer, generator, num_optimization_steps,
          logdir='./logdir'):
    """ Train.
    
    Args:
        sess: A Session.
        model: A Model.
        optimizer: An Optimizer.
        generator: A generator that yields `(inputs, targets)` tuples, with
            `inputs` and `targets` both having shape `[dynamic_duration, 1]`.
        num_optimization_steps: An integer.
        logdir: A string. The log directory.
    """
    
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        
    tf.scalar_summary('loss', model.loss)
    
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss_ema = ema.apply([model.loss])
    loss_ema = ema.average(model.loss)
    tf.scalar_summary('loss_ema', loss_ema)
        
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(logdir=logdir, graph=sess.graph)
    
    sess.run(tf.initialize_all_variables())
    for step in range(num_optimization_steps):
        inputs, targets, = next(generator)
        #print(inputs.shape, targets.shape )
        loss_ema_, summary, _, _ = sess.run(
            [loss_ema, summary_op, optimizer.optimize_op, update_loss_ema],
            {model.inputs: inputs, model.targets: targets})
        summary_writer.add_summary(summary, global_step=step)
        print('\rStep %d. Loss EMA: %.6f.' % (step+1, loss_ema_), end='')


# Now we can train our model:

#generator = input_target_generator()
target_size = 1
input_size = 3
hidden_layer_size = 64
from get_data import get_data_from_summary_file #get_data

#generator = get_data( DATADIR, batch_size = 1 )
generator = get_data_from_summary_file(SUMMARY_FILE)
model = Model(hidden_layer_size=hidden_layer_size, input_size=input_size,
              target_size=target_size, init_scale=0.1)

optimizer = Optimizer(model.loss, initial_learning_rate=1e-4, num_steps_per_decay=15000,
                      decay_rate=0.1, max_global_norm=1.0)

sess = tf.Session()
train(sess, model, optimizer, generator, num_optimization_steps=45000)

# After running `tensorboard --logdir ./logdir` and navigating to [http://localhost:6006](http://localhost:6006), we can view our loss summaries. Here the exponential moving average is especially helpful because our raw losses correspond to individual sequences (and are therefore very noisy estimates).
