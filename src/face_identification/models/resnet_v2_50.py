import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

def get_weights():
    if not os.path.exists('./pretrained/resnet/model.ckpt.index'):
        import tarfile
        from urllib.request import urlretrieve
        try:
            urlretrieve('http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz', './resnet_v2_50_2017_04_14.tar.gz')  
        
            if not os.path.exists('./pretrained/resnet_v2_50'):
                os.makedirs('./pretrained/resnet_v2_50')
                
            with tarfile.open('./resnet_v2_50_2017_04_14.tar.gz') as tar:
                tar.extractall('./pretrained/resnet_v2_50')
        finally:
            if os.path.exists('./resnet_v2_50_2017_04_14.tar.gz'):
                os.remove('./resnet_v2_50_2017_04_14.tar.gz')

                
__resnet_params = None
def model(inputs_image, is_training_mode, dropout_keep_prob):
    _exclude_params = [v.name for v in slim.get_variables_to_restore()]
    
    _net = tf.identity(inputs_image, 'inputs/image')
    _net = tf.image.convert_image_dtype(_net, dtype=tf.float32)
    _net = tf.subtract(_net, 0.5)
    _net = tf.multiply(_net, 2.0)
    with slim.arg_scope(slim.nets.resnet_v2.resnet_arg_scope()):
        _, e = slim.nets.resnet_v2.resnet_v2_50(_net,
                                                  is_training=is_training_mode,
                                                  global_pool=False)
        return e['resnet_v2_50/block4']


def exclude_params():
    return []


def get_restore_op():
    if os.path.exists('./pretrained/resnet_v2_50/resnet_v2_50.ckpt'):
        return slim.assign_from_checkpoint_fn('./pretrained/resnet_v2_50/resnet_v2_50.ckpt', __resnet_params)
    else:
        return None
