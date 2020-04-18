# Losses standard format:
# Task: Search 2 functions that are better than categorical cross entropy

import tensorflow as tf

def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(tf.abs(y_true - y_pred))
