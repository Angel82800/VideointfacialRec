import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets


def get_weights():
    if not os.path.exists('./pretrained/vgg_16.ckpt'):
        import tarfile
        from urllib.request import urlretrieve
        try:
            urlretrieve('http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz', './vgg_16_2016_08_28.tar.gz')  
        
            with tarfile.open('./vgg_16_2016_08_28.tar.gz') as tar:
                tar.extractall('./pretrained')
        finally:
            if os.path.exists('./vgg_16_2016_08_28.tar.gz'):
                os.remove('./vgg_16_2016_08_28.tar.gz')
                

def model(inputs_image, output_grids, is_training_mode, dropout_keep_prob):
    _net = inputs_image - tf.constant([123.68, 116.78, 103.94], tf.float32, shape=[1, 1, 1, 3])
    vgg = tf.contrib.slim.nets.vgg
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, e = vgg.vgg_16(_net, spatial_squeeze=False,
                                   is_training=is_training_mode,
                                   dropout_keep_prob=dropout_keep_prob)
        outputs_dict = dict()
        for name, node in sorted(e.items(), key=lambda x: np.prod([int(s) for s in x[1].shape[1:3]])):
            for grid in output_grids:
                if node.shape[1] == grid[0] and node.shape[2] == grid[1]:
                    outputs_dict[tuple(grid)] = node
                    
        return [outputs_dict[tuple(grid)] for grid in output_grids]


def exclude_params():
    return ['vgg_16/fc8']


def get_restore_op():
    if os.path.exists('./pretrained/vgg_16.ckpt'):
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude_params())
        variables_to_restore = [var for var in variables_to_restore if 'vgg_16' in var.name]
        return slim.assign_from_checkpoint_fn('./pretrained/vgg_16.ckpt', variables_to_restore)
    else:
        return None
