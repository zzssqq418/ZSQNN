from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import deeplab_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug

import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')
parser.add_argument('--train_epochs', type=int, default=26,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=10,
                    help='Number of examples per batch.')
parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=30000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, default='./dataset/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--pre_trained_model', type=str, default='./ini_checkpoints/resnet_v2_101/resnet_v2_101.ckpt',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')
parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')
_NUM_CLASSES = 21
_HEIGHT = 513
_WIDTH = 513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997
_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}
def get_filenames(is_training, data_dir):
'''
返回一个文件名的列表
is_training:一个布尔值，决定输入是否为了训练
data_dir:path to the directory containing the input data
'''
    if is_training:
        return [os.path.join(data_dir,'voc_train.record')]
    else:
        return [os.path.join(data_dir,'voc_val.record')]
#tf.FixedLenFeature返回是一个定长的tensoe,tf.VarLenFeature返回的是一个不定长的sparse tensor
def  parse_record(raw_record):
    #Parse PASCAL image and label from a tf record
    keys_to_features={
        'image/height':
            tf.FixedLenFeature((),tf.int64),
        'image/width':
            tf.FixedLenFeature((),tf.int64),
        'image/encoded':
            tf.FixedLenFeature((),tf.string,default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'label/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'label/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed=tf.parse_single_example(raw_record,keys_to_features)
    #height=tf.cast(parsed['image/height'],tf.int32)
    #width=tf.cast(parsed['image/width'],tf.int32)

    #处理这些图片
    image=tf.image.decode_image(
        tf.reshape(parsed['image/encoded'],shape=[]),_DEPTH)
    image=tf.to_float(tf.image.convert_image_dtype(label,dtype=tf.unit8))
    image.set_shape([None,None,1])
    #处理这些label
    label = tf.image.decode_image(
        tf.reshape(parsed['label/encoded'], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])

    return image,label
def preprocess_image(image,label,is_training):
'''preprocess a single image of layout[h,w,depth]'''
    if is_training:
        image,label=preprocessing.randm_rescale_image_and_label(
            image,label,_MIN_SCALE,_MAX_SCALE)
        #randomly crop or pad a [_HEIGHT,_WIDTH] selection of the image and label
        image,label=preprocessing.random_crop_or_pad_image_and_label(
            image,label,_HEIGHT,_WIDTH,_IGNORE_LABEL)
        #Randomly flip the image and label horizontally.
        image, label = preprocessing.random_flip_left_right_image_and_label(
            image, label)




