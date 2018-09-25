import os
import numpy as np
import tensorflow as tf

# Training Data
samples_num = 30
train_X     = np.arange(0, samples_num).reshape((samples_num, 1))
train_Y     = (train_X ** 2).reshape(samples_num, 1)

def new_run_log_dir(base_dir):
  log_dir = os.path.join('./log', base_dir)
  if not os.path.exists(log_dir): os.makedirs(log_dir)
  run_id      = len([name for name in os.listdir(log_dir)])
  run_log_dir = os.path.join(log_dir, str(run_id))
  return run_log_dir

def print_summary(real_labels, predicted_labels):
  print('Optimization Finished!')
  print('True values: {}'.format(real_labels.flatten()))
  print('Predicted values: {}'.format(predicted_labels.flatten()))
  print('Deviation from true values: {}'
    .format(real_labels.flatten() - predicted_labels.flatten()))
