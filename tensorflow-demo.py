import tensorflow as tf
import matplotlib.pyplot as plt

from common import new_summary_writer, train_X, train_Y


# Parameters
learning_rate   = 0.01
training_epochs = 1000
display_step    = 50

g = tf.Graph()
with g.as_default():
  # Inputs
  X = tf.placeholder("float")
  Y = tf.placeholder("float")

  # Model
  W    = tf.Variable(name="weight")
  b    = tf.Variable(name="bias"  )
  pred = tf.add(tf.multiply(X, W), b)
  init = tf.global_variables_initializer()

  # Objective
  cost      = tf.losses.mean_squared_error(Y, pred)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # Summaries
  tf.summary.scalar    ('loss'   , cost)
  tf.summary.historgram('weights', W   )
  tf.summary.historgram('biases' , b   )
  summaries  = tf.summary.merge_all()
  log_writer = new_summary_writer(g)

# Training
with tf.Session(graph = g) as sess:
  sess.run(init)

  # Fit all training data
  for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict = {X: train_X, Y: train_Y})

    # Log summaries
    if epoch % display_step == 0:
      summary, c = sess.run([summaries, cost], feed_dict={X: train_X, Y:train_Y})
      print('Epoch: {}; Cost: {}; W: {}; b: {}'
        .format(epoch, c, sess.run(W), sess.run(b)))
      log_writer.add_summary(summary, epoch)
      log_writer.flush()

  print('Optimization Finished!')

  # Graphic display
  plt.plot(train_X, train_Y       , 'ro', label='Original data')
  plt.plot(train_X, sess.run(pred),       label='Fitted line'  )
  plt.legend()
  plt.show()