import tensorflow as tf

hparams = tf.contrib.training.HParams(
    gpu_memory_fraction = 0.1,
    path_to_model = 'models/20180402-114759.pb',
    facialrec_threshold = 0.9,

    minsize = 20, # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ],  # three steps's threshold
    factor = 0.709, # scale factor
    margin = 44,
    image_size = 160
)