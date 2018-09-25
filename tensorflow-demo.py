#!/usr/bin/env python

import tensorflow as tf
import numpy      as np
from common import new_run_log_dir, train_X, train_Y, samples_num, print_summary

# Parameters
learning_rate   = 0.000001
training_epochs = 500
display_step    = 10
hidden_size     = 128

log_dir = new_run_log_dir('tensorflow-demo')

g = tf.Graph()
with g.as_default():
  # Inputs
  X = tf.placeholder(np.float32, (samples_num, 1))  # *, 1
  Y = tf.placeholder(np.float32, (samples_num, 1))  # *, 1

  # Model
  # 1, X
  W_1 = tf.get_variable("W_1", (1, hidden_size), np.float32, initializer=tf.random_uniform_initializer)
  b_1 = tf.get_variable("b_1", (1, hidden_size), np.float32, initializer=tf.random_uniform_initializer)

  W_2 = tf.get_variable("W_2", (hidden_size, 1), np.float32, initializer=tf.random_uniform_initializer)
  b_2 = tf.get_variable("b_2", (1             ), np.float32, initializer=tf.random_uniform_initializer)

  hidden = tf.nn.relu(tf.matmul(X, W_1) + b_1)
  pred   = tf.matmul(hidden, W_2) + b_2


  # tf.add(tf.multiply(tf.add(tf.multiply(X, W), b), W_h), b_h)
  init = tf.global_variables_initializer()

  # Objective
  cost      = tf.losses.mean_squared_error(Y, pred)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # Summaries
  tf.summary.scalar   ('loss', cost)
  tf.summary.histogram('W_1' , W_1 )
  tf.summary.histogram('b_1' , b_1 )
  tf.summary.histogram('W_2' , W_2 )
  tf.summary.histogram('b_2' , b_2 )

  summaries  = tf.summary.merge_all()
  log_writer = tf.summary.FileWriter(log_dir, graph = g)

# Training
with tf.Session(graph = g) as sess:
  sess.run(init)

  # Fit all training data
  for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict = {X: train_X, Y: train_Y})

    # Log summaries
    if epoch % display_step == 0:
      summary, c = sess.run([summaries, cost], feed_dict={X: train_X, Y:train_Y})
      print('Epoch: {};\tCost: {}'.format(epoch, c))
      log_writer.add_summary(summary, epoch)
      log_writer.flush()

  predicted = sess.run(pred, feed_dict = {X: train_X})
  print_summary(train_Y, predicted)
