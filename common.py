import os

log_dir = './log/tensorflow-demo'

# Training Data
train_X   = numpy.arange(1, 30)
train_Y   = 2 * train_X

def new_summary_writer(graph):
  if not os.path.exists(log_dir): os.makedirs(log_dir)
  run_id      = len([name for name in os.listdir(log_dir)])
  run_log_dir = os.path.join(log_dir, str(run_id))
  return tf.summary.FileWriter(run_log_dir, graph = graph)
