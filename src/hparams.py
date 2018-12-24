import tensorflow as tf
import multiprocessing

hparams = tf.contrib.training.HParams(
    buffer_size = 1000,
    gpu_memmory_fraction = 0.99,
    random_seed = 42,
    
    # Model params
    #model = 'resnet_v2_50',
    model = 'mobilenet_v2',
    input_image_size = [416, 416],
    output_grids = [
        [52, 52],
        [26,  26],
        [13, 13]
    ],
    predict_classes = True,
    n_classes = 2,
    confidence_tresh = 0.5,
    iou_tresh = 0.6,
    
    # Training params
    batch_size = 64,
    training_steps = 300000,
    
    decay_learning_rate = True,
    start_decay = 50000,
    decay_steps = 20000,
    decay_rate = 0.2,
    initial_learning_rate = 1e-3,
    final_learning_rate = 1e-5,
    
    adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
    adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
    adam_epsilon = 1e-6, #AdamOptimizer beta3 parameter

    grad_clip_value = 1,
    training_dir = './training',
    allow_restoring = True,
    enabled_gpus = [0],
    #dataset_path = '/home/facialrec/notebooks/datasets/',
    dataset_path = '/home/facialrec/notebooks/VideointfacialRec/src',
    summarize_gradients = True,
    steps_per_checkpoint = 2500,
    dropout_keep_prob = 0.5,
    steps_per_summary = 500,
    dataset_n_workers = multiprocessing.cpu_count(),
    n_augumented = 9

)
