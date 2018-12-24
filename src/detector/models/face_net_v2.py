import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

# Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation
# EffNet: AN EFFICIENT STRUCTURE FOR CONVOLUTIONAL NEURAL NETWORKS

def get_weights():
    pass

__model_params = None
def model(inputs_image, output_grids, is_training_mode, dropout_keep_prob, reuse=False):
    def bottleneck(inputs, out_channels, t=1, stride=1, scope=None, reuse=False):
        net = inputs
        
        with tf.variable_scope(scope or 'bottleneck', reuse=reuse):
            net = slim.batch_norm(net, is_training=is_training_mode, scope='bn-1', reuse=reuse)
            net = slim.conv2d(net, int(net.shape[-1]) * t, [1, 1], scope='conv-1', reuse=reuse)

            net_ = net
            with tf.variable_scope('depthwise-conv-1', reuse=reuse):
                weights = tf.get_variable('weights', [3, 1, int(net.shape[-1]), 1], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                biases = tf.get_variable('biases', [int(net.shape[-1])], tf.float32, initializer=tf.zeros_initializer(), trainable=True)
                net = tf.nn.depthwise_conv2d(net, weights, [1, 1, 1, 1], 'SAME')
                net = tf.nn.bias_add(net, biases)
                net = tf.nn.relu(net)
            with tf.variable_scope('depthwise-conv-2', reuse=reuse):
                weights = tf.get_variable('weights', [1, 3, int(net.shape[-1]), 1], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                biases = tf.get_variable('biases', [int(net.shape[-1])], tf.float32, initializer=tf.zeros_initializer(), trainable=True)
                net = tf.nn.depthwise_conv2d(net, weights, [1, stride, stride, 1], 'SAME')
                net = tf.nn.bias_add(net, biases)
                
                if [int(i) for i in net.shape[1:]] != [int(i) for i in net_.shape[1:]]:
                    net = tf.nn.relu(net)
                
            if [int(i) for i in net.shape[1:]] == [int(i) for i in net_.shape[1:]]:
                net = tf.add(net, net_)
                net = tf.nn.relu(net)

            net = slim.conv2d(net, out_channels, [1, 1], scope='conv-out', activation_fn=None, reuse=reuse)
            if [int(i) for i in net.shape[1:]] == [int(i) for i in inputs.shape[1:]]:
                net = tf.add(net, inputs)

        return net
    
    if not reuse:
        _exclude_params = [v.name for v in slim.get_variables_to_restore()]
    
    net = tf.image.convert_image_dtype(inputs_image, dtype=tf.float32)
    #net = tf.subtract(net, 0.5)
    #net = tf.multiply(net, 2.0)
    
    outputs_dict = dict()
    with tf.variable_scope('face-net', reuse=reuse):    
        net = slim.batch_norm(net, is_training=is_training_mode, scope='bn-input', reuse=reuse)

        with tf.variable_scope('block-input', reuse=reuse):
            net = slim.conv2d(net, 32, [3, 1], scope='conv-1', reuse=reuse)
            net = slim.conv2d(net, 32, [1, 3], stride=2, scope='conv-2', reuse=reuse)
            outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net

        net = bottleneck(net, 16, t=1, stride=1, scope='bottleneck-1-1', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net

        net = bottleneck(net, 32, t=4, stride=1, scope='bottleneck-2-1', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net
        net = bottleneck(net, 32, t=4, stride=2, scope='bottleneck-2-2', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net

        net = bottleneck(net, 64, t=4, stride=1, scope='bottleneck-3-1', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net
        net = bottleneck(net, 64, t=4, stride=2, scope='bottleneck-3-2', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net
        
        net = bottleneck(net, 128, t=4, stride=1, scope='bottleneck-4-1', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net
        net = bottleneck(net, 128, t=4, stride=2, scope='bottleneck-4-2', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net

        net = bottleneck(net, 256, t=2, stride=1, scope='bottleneck-5-1', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net
        net = bottleneck(net, 256, t=2, stride=1, scope='bottleneck-5-2', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net
        net = bottleneck(net, 256, t=2, stride=2, scope='bottleneck-5-3', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net

        net = bottleneck(net, 256, t=2, stride=1, scope='bottleneck-6-1', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net
        net = bottleneck(net, 256, t=2, stride=1, scope='bottleneck-6-2', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net
        net = bottleneck(net, 256, t=2, stride=1, scope='bottleneck-6-3', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net

        net = bottleneck(net, 256*2, t=2, stride=1, scope='bottleneck-7-1')
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net

        net = slim.conv2d(net, 256*2*4, [1, 1], scope='conv-1', reuse=reuse)
        outputs_dict[(int(net.shape[1]), int(net.shape[2]))] = net

        # classifier
        # avg_pool
        # conv2d 1x1xK
        if not reuse:
            global __model_params
            __model_params = [var for var in slim.get_variables_to_restore() if var.name not in _exclude_params]

        return [outputs_dict[tuple(grid)] for grid in output_grids]


def exclude_params():
    return []


def get_restore_op():
    if os.path.exists('./pretrained/face_net/model.ckpt.index'):
        return slim.assign_from_checkpoint_fn('./pretrained/face_net/model.ckpt', __model_params)
    else:
        return None
