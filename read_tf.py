

import tensorflow as tf
import cv2
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

tfrecords_filename = 'tfrecords/floor_test.tfrecords'


sess = tf.Session()
sess.run(tf.global_variables_initializer())

def parser(record):
    keys_to_features = {
        #'height': tf.FixedLenFeature([], tf.int64),
        #'width': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image"], tf.uint8) #jpg images can only be decoded to uint8
    image = tf.cast(image, tf.float32)   # this line is used to normlize
    
    label = tf.decode_raw(parsed["label"], tf.uint8)
    label = tf.cast(label, tf.float32)


    #dim = tf.stack([200, 200, 3])
    #image =  tf.reshape(image, [200,200,3])


    return {'image': image}, label


def input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1024, 1))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parser, 32))
    #dataset = dataset.map(parser, num_parallel_calls=12)
    #dataset = dataset.batch(batch_size=1000)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


def test_cv_display():

    dataset = input_fn(tfrecords_filename).make_one_shot_iterator()
    nextElement = dataset.get_next()
    image, label = sess.run(nextElement)
    my_image = image['image'].reshape(-1, 200,200,3)
    my_label = label.reshape(-1, 200,200)

    print(my_image.shape)
    print(my_label.shape)

    for i in range(10):
        #print(my_label[i,:,:])
        
        image_convert = (my_label[i,:,:].astype(np.uint8) *255)

        cv2.imshow('test', my_image[i,:,:,:].astype(np.uint8)) # inforder for the image to display properly, it needs to be converted to uint8
        cv2.imshow('test1', image_convert)
        cv2.waitKey(0)



#test_cv_display()


