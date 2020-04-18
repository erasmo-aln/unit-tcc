# Get the images and build a TFRecords Dataset

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pathlib 

META_DIR = pathlib.Path('Meta')
WIDTH, HEIGHT = 32, 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
list_dataset = tf.data.Dataset.list_files(str(META_DIR/'*'))

def get_label(file_path):
    path_split = tf.strings.split(input=file_path, sep="\\")[-1]
    path_split = tf.strings.split(input=path_split, sep='.')[-2]
    return path_split

def decode_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, [WIDTH, HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return image, label

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

labeled_dataset = list_dataset.map(process_path, num_parallel_calls=AUTOTUNE)

feature = {
    'image': _bytes_feature(image),
    'label': _int64_feature(label)
    }

def create_example(image, label):
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
    # Create a Features message using tf.train.Example.
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def serialize_example(image, label):
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
# Create a Features message using tf.train.Example.
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


tfpath = 'dataset.tfrecords'
with tf.io.TFRecordWriter(tfpath) as writer:
    for image, label in labeled_dataset:
        serialized = serialize_example(image, label)
        writer.write(serialized)