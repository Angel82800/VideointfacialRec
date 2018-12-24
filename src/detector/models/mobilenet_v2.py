import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from detector.models.slim_nets.mobilenet_v2 import mobilenet, training_scope

def get_weights():
    if not os.path.exists('./pretrained/mobilenet/mobilenet_v2_1.0_224.ckpt.index'):
        import tarfile
        from urllib.request import urlretrieve
        try:
            urlretrieve('https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz', './mobilenet_v2_1.0_224.tgz')  
        
            if not os.path.exists('./pretrained/mobilenet'):
                os.makedirs('./pretrained/mobilenet')
                
            with tarfile.open('./mobilenet_v2_1.0_224.tgz') as tar:
                tar.extractall('./pretrained/mobilenet')
        finally:
            if os.path.exists('./mobilenet_v2_1.0_224.tgz'):
                os.remove('./mobilenet_v2_1.0_224.tgz')
                
__mobilenet_params = None
valid_layers = [f'layer_{i}' for i in range(1, 20)]
def model(inputs_image, output_grids, is_training_mode, dropout_keep_prob):
    _exclude_params = [v.name for v in slim.get_variables_to_restore()]
    
    _net = tf.image.convert_image_dtype(inputs_image, dtype=tf.float32)
    _net = tf.subtract(_net, 0.5)
    _net = tf.multiply(_net, 2.0)
    
    with tf.contrib.slim.arg_scope(training_scope(dropout_keep_prob=dropout_keep_prob)):
        _, endpoints = mobilenet(_net)
        
        e = []
        for key, value in endpoints.items():
            if key in valid_layers:
                e.append((key, value))
                
        e = sorted(e, key=lambda x: int(x[0][6:]), reverse=True)
        
        outputs_dict = dict()
        for name, node in e:
            for grid in output_grids:
                if len(node.shape) == 4 and node.shape[1] == grid[0] and node.shape[2] == grid[1]:
                    outputs_dict[tuple(grid)] = node
        
        if is_training_mode:
            global __mobilenet_params
            print('Все карнц1', len(__mobilenet_params) if __mobilenet_params is not None else None)
            __mobilenet_params = [var for var in slim.get_variables_to_restore() if var.name not in _exclude_params]
            print('Все карнц2', len(__mobilenet_params))
        return [outputs_dict[tuple(grid)] for grid in output_grids]
    
    
def exclude_params():
    return []


def get_restore_op():
    #global __mobilenet_params
    if os.path.exists('./pretrained/mobilenet/mobilenet_v2_1.0_224.ckpt.index'):
        return slim.assign_from_checkpoint_fn('./pretrained/mobilenet/mobilenet_v2_1.0_224.ckpt', __mobilenet_params)
    else:
        return None
    
