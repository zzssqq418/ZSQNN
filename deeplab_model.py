from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

from utils import preprocessing

'''
layers里
optimizers模块包括SGD，Momentum等优化器；

regularizers模块包括L1正则化和L2正则化，这通常用来抑制模型训练时特征数过大导致的过拟合问题，有时也作为lasso
回归和Ridge回归的构建模块

initializers模块一般用来做模型初始化，随机初始化模型参数

feature_column模块提供函数(比如bucketing/binning,crossing/composition)来转换连续特征和离散特征

embedding模块转化高维分类特征成低维密集实数值向量
'''

# from utils import preprocessing
# utils 是一个小型py函数和类的集合，将常用功能封装成为接口。
_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


# 下面定义ASPP这个函数
def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
    '''
    ：param inputs: A tensor of size[batch,height,width,channels]
    :param output_stride: The Resnet unit's stride.Determines the rates for atrous conv
        the rates are(6,12,18) when the stride is 16,and doubled when 8
    :param batch_norm_decay: the moving average decay移动平均衰减 when estimating layer activation
        statistics in batch normalization.当在BN过程中估计 层激活统计时
    :param is_training: A boolean denoting whether the input is for training
    :param depth: the depth of the resnet unit output resnet输出的深度？？
    :return:Atrous Spatial Pyramid Pooling
    '''
    with tf.variable_scope('aspp'):
        # tf.get_variable(<name>,<shape>,<initializer>)创建或返回给定name的变量
        # tf.variable_scope(<scope_name>)管理传给get_variable()的变量名称的作用域，打开着这个作用域，进行下面的卷积等等操作
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')
        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2 * rate for rate in atrous_rates]
        '''
        slim.arg_scope使用时，它常用于为layer函数提供！默认值！以使构建模型的代码更加紧凑苗条
        并不是所有方法都能用arg_scope设置默认参数，只有用@slim.add_arg_scope修饰过的方法才能使用arg_scope
        要使slim.arg_scope正常运行起来，需要两个步骤，用@add_arg_scope修饰目标函数，用with arg_scope(...)设置默认参数
        比如
        with slim.arg_scope([layers.conv2d], padding='SAME',         
                initializer= xavier_initializer(),
                regularizer= l2_regularizer(0.05)):  
            net = slim.conv2d(net, 256, [5, 5])

        '''
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            # resnet_v2.resnet_arg_scope()
            inputs_size = tf.shape(inputs)[1:3]  # tf.shape返回的是tensor类型的对象
            conv_11 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope='conv_11')  # 1*1卷积作用
            # conv_1*1,下面是conv_3*3,给出三种rate的3*3空洞卷积作用的结果
            conv_33_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_33_1')
            conv_33_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_33_2')
            conv_33_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_33_3')
            with tf.variable_scope('image_level_features'):  # 又在这个作用域里
                # reduce_mean函数用于计算张量tensor沿着指定的数轴（tensor上的某一维度上的平均值），主要用作降维或者计算tensor
                # 一个图片上的所有像素求均值
                image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
                image_level_features = layers_lib.conv2d(image_leval_features, depth, [1, 1], stride=1, scope='conv_11')
                #
                # 双线性插值调整图像大小，输出float32类型的tensor,image_l那里是4维的[batch,h,w,channel],inputs_size是两个元素，用来表示新图像的大小
                image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
            net = tf.concat([conv_11, conv_33_1, conv_33_2, conv_33_3, image_level_features],
                            axis=3, name='concat')
            net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_11_concat')
            return net
            # 下面这个是deeplab生成器


def deeplab_v3_generator(num_classes,
                         output_stride,
                         base_architecture,
                         pre_trained_model,
                         batch_norm_decay,
                         data_format='channels_last'):
    """Generator for DeepLab v3 models.
    Args:
      num_classes:the number of possible classes for image classification.

      output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.

      base_architecture: The architecture of base Resnet building block.

      pre_trained_model: The path to the directory that contains pre-trained models.预训练模型的目录
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.

      data_format: The input format ('channels_last', 'channels_first', or None).表示数据通道的位置
        If set to None, the format is dependent on whether a GPU is available.
        Only 'channels_last' is supported currently.

    Returns:
      The model function that takes in `inputs` and `is_training` and
      returns the output tensor of the DeepLab v3 model.
    """
    if data_format is None:
        # data_format = (
        #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        pass

    if batch_norm_decay is None:
        batch_norm_decay = _BATCH_NORM_DECAY

    if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
        raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_101'.")

    if base_architecture == 'resnet_v2_50':
        base_model = resnet_v2.resnet_v2_50
    else:
        base_model = resnet_v2.resnet_v2_101

    def model(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # tf.logging.info('net shape: {}'.format(inputs.shape))

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            logits, end_points = base_model(inputs,
                                            num_classes=None,
                                            is_training=is_training,
                                            global_pool=False,
                                            output_stride=output_stride)

        if is_training:
            exclude = [base_architecture + '/logits', 'global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(pre_trained_model,
                                          {v.name.split(':')[0]: v for v in variables_to_restore})

        inputs_size = tf.shape(inputs)[1:3]
        net = end_points[base_architecture + '/block4']
        net = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
        with tf.variable_scope("upsampling_logits"):
            net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample')

        return logits


return model


def deeplabv3_model_fn(features, labels, mode, params):
    """Model function for PASCAL VOC."""
    if isinstance(features, dict):
        features = features['feature']

    images = tf.cast(
        tf.map_fn(preprocessing.mean_image_addition, features),
        tf.uint8)


''' 
lambda函数能接收任何数量的参数但只能返回一个表达式的值
tf.map_fn(fn=lambda x:tf.nn.conv2d(x,kernel,stride,padding='same'),elems=batch,dtype=tf.float32)
这个函数是一个反复将可调用函数fn应用于elems元素序列的一个高阶函数
比如你要对视频卷积，是对每一帧进行卷积，
对每个切片应用卷积操作，每个batch之间没有关联，可以并行快速地处理
'''
network = deeplab_v3_generator(params['num_classes'],
                               params['output_stride'],
                               params['base_architecture'],
                               params['pre_trained_model'],
                               params['batch_norm_decay'])

logits = network(features, mode == tf.estimator.ModeKeys.TRAIN)

pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

pred_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                 [pred_classes, params['batch_size'], params['num_classes']],
                                 tf.uint8)

predictions = {
    'classes': pred_classes,
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    'decoded_labels': pred_decoded_labels
}

if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
    predictions_without_decoded_labels = predictions.copy()
    del predictions_without_decoded_labels['decoded_labels']

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'preds': tf.estimator.export.PredictOutput(
                predictions_without_decoded_labels)
        })

gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                               [labels, params['batch_size'], params['num_classes']], tf.uint8)

labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.

logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
labels_flat = tf.reshape(labels, [-1, ])

valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

preds_flat = tf.reshape(pred_classes, [-1, ])
valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

predictions['valid_preds'] = valid_preds
predictions['valid_labels'] = valid_labels
predictions['confusion_matrix'] = confusion_matrix

cross_entropy = tf.losses.sparse_softmax_cross_entropy(
    logits=valid_logits, labels=valid_labels)

# Create a tensor named cross_entropy for logging purposes.
tf.identity(cross_entropy, name='cross_entropy')
tf.summary.scalar('cross_entropy', cross_entropy)

if not params['freeze_batch_norm']:
    train_var_list = [v for v in tf.trainable_variables()]
else:
    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]

# Add weight decay to the loss.
with tf.variable_scope("total_loss"):
    loss = cross_entropy + params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])
# loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

if mode == tf.estimator.ModeKeys.TRAIN:
    tf.summary.image('images',
                     tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
                     max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

    global_step = tf.train.get_or_create_global_step()

    if params['learning_rate_policy'] == 'piecewise':
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 128, the learning rate should be 0.1.
        initial_learning_rate = 0.1 * params['batch_size'] / 128
        batches_per_epoch = params['num_train'] / params['batch_size']
        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
        values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)
    elif params['learning_rate_policy'] == 'poly':
        learning_rate = tf.train.polynomial_decay(
            params['initial_learning_rate'],
            tf.cast(global_step, tf.int32) - params['initial_global_step'],
            params['max_iter'], params['end_learning_rate'], power=params['power'])
    else:
        raise ValueError('Learning rate policy must be "piecewise" or "poly"')

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['momentum'])

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)
else:
    train_op = None

accuracy = tf.metrics.accuracy(
    valid_labels, valid_preds)
mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

# Create a tensor named train_accuracy for logging purposes
tf.identity(accuracy[1], name='train_px_accuracy')
tf.summary.scalar('train_px_accuracy', accuracy[1])


def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['num_classes']):
        tf.identity(iou[i], name='train_iou_class{}'.format(i))
        tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result


train_mean_iou = compute_mean_iou(mean_iou[1])

tf.identity(train_mean_iou, name='train_mean_iou')
tf.summary.scalar('train_mean_iou', train_mean_iou)

return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=metrics)


