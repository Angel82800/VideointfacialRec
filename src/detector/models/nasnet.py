import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from detector.models.slim_nets.nasnet import build_nasnet_mobile, nasnet_mobile_arg_scope

def get_weights():
    if not os.path.exists('./pretrained/nasnet/model.ckpt.index'):
        import tarfile
        from urllib.request import urlretrieve
        try:
            urlretrieve('https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz', './nasnet-a_mobile_04_10_2017.tar.gz')  
        
            if not os.path.exists('./pretrained/nasnet'):
                os.makedirs('./pretrained/nasnet')
                
            with tarfile.open('./nasnet-a_mobile_04_10_2017.tar.gz') as tar:
                tar.extractall('./pretrained/nasnet')
        finally:
            if os.path.exists('./nasnet-a_mobile_04_10_2017.tar.gz'):
                os.remove('./nasnet-a_mobile_04_10_2017.tar.gz')
                
__nasnet_params = None
def model(inputs_image, output_grids, is_training_mode, dropout_keep_prob):
    _exclude_params = [v.name for v in slim.get_variables_to_restore()]
    
    _net = tf.image.convert_image_dtype(inputs_image, dtype=tf.float32)
    _net = tf.subtract(_net, 0.5)
    _net = tf.multiply(_net, 2.0)
    with slim.arg_scope(nasnet_mobile_arg_scope()):
        _, e = build_nasnet_mobile(_net, None,
                                     is_training=is_training_mode,
                                     final_endpoint='Cell_11')
        outputs_dict = dict()
        for name, node in sorted(e.items(), key=lambda x: np.prod([int(s) for s in x[1].shape[1:3]])):
            for grid in output_grids:
                if node.shape[1] == grid[0] and node.shape[2] == grid[1]:
                    outputs_dict[tuple(grid)] = node
                    
        global __nasnet_params
        __nasnet_params = [var for var in slim.get_variables_to_restore() if var.name not in _exclude_params]
        return [outputs_dict[tuple(grid)] for grid in output_grids]


def exclude_params():
    return []


def get_restore_op():
    if os.path.exists('./pretrained/nasnet/model.ckpt.index'):
        return slim.assign_from_checkpoint_fn('./pretrained/nasnet/model.ckpt', __nasnet_params)
    else:
        return None
