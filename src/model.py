import tensorflow as tf
import importlib
import tensorflow.contrib.slim as slim
import numpy as np

class Yolo:
    def __init__(self, hparams):
        self._hparams = hparams
        self._out_channels = 1+4
        
        if self._hparams.predict_classes and self._hparams.n_classes > 0:
            if self._hparams.n_classes == 2:
                self._out_channels += 1
            else:
                self._out_channels += self._hparams.n_classes
                
        self._feature_embedd_channels = 256
        
    def _add_targets(self, targets_conf_probs=None, targets_bboxes=None, targets_classes=None):
        with tf.name_scope('targets'):
            self.targets_conf_probs = []
            self.targets_bboxes = []
            self.targets_classes  = []
            self.targets_nodes = []
            
            for i, grid in enumerate(self._hparams.output_grids):
                with tf.name_scope('grid-%ix%i' % tuple(grid)):
                    self.targets_conf_probs.append(tf.identity(targets_conf_probs[i], name='conf_probs'))
                    self.targets_bboxes.append(tf.identity(targets_bboxes[i], name='bboxes_xy'))
                    
                    self.targets_nodes.append((self.targets_conf_probs[-1], self.targets_bboxes[-1]))

                    if self._hparams.predict_classes and self._hparams.n_classes > 0:
                        self.targets_classes.append(tf.identity(targets_classes[i], name='classes'))
                    else:
                        self.targets_classes.append(None)
                        
    def _add_model(self, inputs, is_training, keep_prob):
        _model_py_module = importlib.import_module('detector.models.%s' % self._hparams.model)
        _model_py_module.get_weights()
        with tf.name_scope('model'):
            _nodes = _model_py_module.model(tf.cast(inputs, tf.float32), self._hparams.output_grids, is_training, keep_prob)
            _output_nodes = []
            _prev_features_map = None
            for _net, output_grid in reversed(list(zip(_nodes, self._hparams.output_grids))):
                with tf.variable_scope('grid-%ix%i' % tuple(output_grid)):
                    with slim.arg_scope([slim.conv2d], padding='SAME'):
                        _net = slim.conv2d(_net, self._feature_embedd_channels, [1, 1])

                        if _prev_features_map is not None:
                            _net = _net + tf.image.resize_nearest_neighbor(_prev_features_map, [_prev_features_map.shape[1]*2, _prev_features_map.shape[2]*2])
                            _net = slim.conv2d(_net, self._feature_embedd_channels, [3, 3])

                        _prev_features_map = _net

                        _net = slim.conv2d(_net, self._out_channels, [1, 1], activation_fn=None, weights_initializer=tf.zeros_initializer())

                        _output_nodes.append(_net)

                    assert _net.shape[1] == output_grid[0] and _net.shape[2] == output_grid[1], f'Incorrect ouput grid shape: must be {output_grid}, but [{_net.shape[1]}, {_net.shape[2]}] found.'
            self._output_nodes = list(reversed(_output_nodes))
            
            self.restore_op = _model_py_module.get_restore_op()
            
            
    def _add_outputs(self):
        with tf.name_scope('outputs'):
            output_nodes = []
            for _net, grid in zip(self._output_nodes, self._hparams.output_grids):
                with tf.name_scope('grid-%ix%i' % tuple(grid)):
                    outputs_conf_probs_logits = _net[:, :, :, 0:1]
                    outputs_conf_probs = tf.identity(tf.nn.sigmoid(outputs_conf_probs_logits), name='conf_probs')

                    outputs_bboxes = _net[:, :, :, 1:5]

                    outputs_bboxes_xy = tf.sigmoid(outputs_bboxes[:, :, :, :2])
                    outputs_bboxes_wh_log = outputs_bboxes[:, :, :, 2:]
                    outputs_bboxes_wh = tf.exp(outputs_bboxes_wh_log)

                    outputs_bboxes = tf.identity(tf.concat([outputs_bboxes_xy, outputs_bboxes_wh], axis=-1), name='bboxes')

                    if self._hparams.predict_classes and self._hparams.n_classes > 0:
                        outputs_classes_logits = _net[:, :, :, 5:]
                        if self._hparams.n_classes == 1 or self._hparams.n_classes == 2:
                            outputs_classes_probs = tf.sigmoid(outputs_classes_logits)
                            outputs_classes = tf.identity(tf.cast(outputs_classes_probs >= 0.5, tf.int32)[..., 0], name='classes')
                        else:
                            outputs_classes_probs = tf.nn.softmax(outputs_classes_logits, dim=-1)
                            outputs_classes = tf.identity(tf.argmax(outputs_classes_probs, axis=-1), name='classes')

                        output_nodes.append(((outputs_conf_probs_logits, outputs_conf_probs),
                                             (outputs_bboxes_wh_log, outputs_bboxes),
                                             (outputs_classes_logits, outputs_classes_probs, outputs_classes)))
                    else:
                        output_nodes.append(((outputs_conf_probs_logits, outputs_conf_probs),
                                             (outputs_bboxes_wh_log, outputs_bboxes)))
            self.output_nodes = output_nodes
            
                        
        
    def initialize(self, inputs, is_training, targets_conf_probs=None, targets_bboxes=None, targets_classes=None, keep_prob=1.):
        if targets_conf_probs is not None and targets_bboxes is not None:
            self._add_targets(targets_conf_probs, targets_bboxes, targets_classes)
            
        self._add_model(inputs, is_training, keep_prob)
        self._add_outputs()     
        
    def _add_conf_probs_loss(self, outputs_conf_probs, targets_conf_probs, output_grid):
        with tf.name_scope('conf_probs_loss'):
            _targets_conf_probs = tf.cast(targets_conf_probs, tf.float32)
            _targets_conf_probs_bin = tf.cast(_targets_conf_probs >= self._hparams.confidence_tresh, tf.float32)
            _outputs_conf_probs_bin = tf.cast(outputs_conf_probs >= self._hparams.confidence_tresh, tf.float32)
            
            _n_objects = tf.reduce_mean(tf.reduce_sum(_targets_conf_probs_bin, axis=[1, 2, 3]))
            _n_no_objects = tf.reduce_mean(tf.reduce_sum(1-_targets_conf_probs_bin, axis=[1, 2, 3]))

            _n_median_class = tf.contrib.distributions.percentile([_n_no_objects, _n_objects], 50)
            _n_max_class = tf.reduce_max([_n_objects, _n_no_objects])

            _object_scale = (_n_max_class / _n_objects)
            _no_object_scale = (_n_max_class / _n_no_objects)

            _no_objects_loss = -(1 - _targets_conf_probs) * tf.log(tf.maximum(1-outputs_conf_probs, 1e-6))
            _no_objects_loss = tf.reduce_sum(_no_objects_loss, axis=[1, 2, 3])

            _objects_loss = -_targets_conf_probs * tf.log(tf.maximum(outputs_conf_probs, 1e-6))
            _objects_loss = tf.reduce_sum(_objects_loss, axis=[1, 2, 3])

            conf_probs_loss = (_object_scale * _objects_loss + _no_object_scale * _no_objects_loss) / np.mean(output_grid)
            conf_probs_loss = tf.reduce_mean(conf_probs_loss)
            conf_probs_loss = tf.cond(tf.is_finite(conf_probs_loss), lambda: conf_probs_loss, lambda: tf.constant(0, tf.float32))
            conf_probs_loss = tf.identity(conf_probs_loss, name='conf_probs_loss')
            tf.losses.add_loss(conf_probs_loss)
            
    def _add_xy_loss(self, targets_bboxes_xy, outputs_bboxes_xy, detectors_mask, coordinates_scale):
        with tf.name_scope('xy_loss'):
            targets_bboxes_xy_actual = tf.boolean_mask(targets_bboxes_xy, detectors_mask)
            outputs_bboxes_xy_actual = tf.boolean_mask(outputs_bboxes_xy, detectors_mask)
            
            xy_loss = coordinates_scale * self._mse(targets_bboxes_xy_actual, outputs_bboxes_xy_actual)
            xy_loss = tf.cond(tf.is_finite(xy_loss), lambda: xy_loss, lambda: tf.constant(0, tf.float32))
            xy_loss = tf.identity(xy_loss, name='xy_loss')
            tf.losses.add_loss(xy_loss)
            
    def _add_wh_loss(self, targets_bboxes_wh, outputs_bboxes_wh_log, detectors_mask, coordinames_scale):
        with tf.name_scope('wh_loss'):
            targets_bboxes_wh_log = tf.log(tf.maximum(targets_bboxes_wh, 1e-6))
            targets_bboxes_wh_log_actual = tf.boolean_mask(targets_bboxes_wh, detectors_mask)
            outputs_bboxes_wh_log_actual = tf.boolean_mask(outputs_bboxes_wh_log, detectors_mask)
            
            wh_loss = coordinames_scale * self._mse(targets_bboxes_wh_log_actual, outputs_bboxes_wh_log_actual)
            wh_loss = tf.cond(tf.is_finite(wh_loss), lambda: wh_loss, lambda: tf.constant(0, tf.float32))
            wh_loss = tf.identity(wh_loss, name='wh_loss')
            tf.losses.add_loss(wh_loss)
            
    def _add_classes_loss(self, targets_classes_probs, outputs_classes_probs, targets_classes_one_hot, detectors_mask):
        with tf.name_scope('classes_loss'):
            targets_classes_probs_actual = tf.boolean_mask(targets_classes_probs, detectors_mask)
            outputs_classes_probs_actual = tf.boolean_mask(outputs_classes_probs, detectors_mask)

            _n_classes = tf.reduce_sum(tf.boolean_mask(targets_classes_one_hot, detectors_mask), axis=0, keepdims=True)
            _n_median_class = tf.contrib.distributions.percentile(_n_classes, 50, axis=[-1], keep_dims=True)

            _classes_scale = (_n_median_class / tf.maximum(_n_classes, 1e-6))

            if self._hparams.n_classes == 2: # bce
                classes_loss = -(_classes_scale[..., 1:] * targets_classes_probs_actual * tf.log(tf.maximum(outputs_classes_probs_actual, 1e-6)) +
                                 _classes_scale[..., 0:1] * (1 - targets_classes_probs_actual) * tf.log(tf.maximum(1 - outputs_classes_probs_actual, 1e-6)))
            else: # cce
                classes_loss = -tf.reduce_sum(_classes_scale * _targets_classes_probs_actual * tf.log(tf.maximum(outputs_classes_probs_actual, 1e-6)), axis=-1)

            classes_loss = tf.reduce_mean(classes_loss)
            classes_loss = tf.cond(tf.is_finite(classes_loss), lambda: classes_loss, lambda: tf.constant(0, tf.float32))
            classes_loss = tf.identity(classes_loss, 'classes_loss')
            #tf.losses.add_loss(classes_loss)


    def add_loss(self):
        with tf.name_scope('losses'):
            inouts = zip(self._hparams.output_grids, self.targets_conf_probs, self.targets_bboxes, self.targets_classes, self.output_nodes)
            for output_grid, targets_conf_probs, targets_bboxes, targets_classes, outputs_nodes in inouts:
                with tf.name_scope('grid-%ix%i' % tuple(output_grid)):
                    if self._hparams.predict_classes and self._hparams.n_classes > 0:
                        ((outputs_conf_probs_logits, outputs_conf_probs),
                         (outputs_bboxes_wh_log, outputs_bboxes),
                         (outputs_classes_logits, outputs_classes_probs, outputs_classes)) = outputs_nodes

                        _targets_classes = tf.cast(targets_classes, tf.int32)

                        _targets_classes = _targets_classes[..., 0]
                        _targets_classes_one_hot = tf.one_hot(_targets_classes, self._hparams.n_classes)
                        if self._hparams.n_classes == 2:
                            _targets_classes_probs = tf.cast(tf.expand_dims(_targets_classes, axis=-1), tf.float32)
                        else:
                            _targets_classes_probs = tf.cast(_targets_classes_one_hot, tf.float32)
                    else:
                        #targets_conf_probs, targets_bboxes = self._targets_nodes
                        ((outputs_conf_probs_logits, outputs_conf_probs),
                         (outputs_bboxes_wh_log, outputs_bboxes)) = outputs_nodes

                    self._add_conf_probs_loss(outputs_conf_probs, _targets_classes_probs, output_grid)

                    _targets_conf_probs = tf.cast(targets_conf_probs, tf.float32)
                    _targets_conf_probs_bin = tf.cast(_targets_conf_probs >= self._hparams.confidence_tresh, tf.float32)
                    _outputs_conf_probs_bin = tf.cast(outputs_conf_probs >= self._hparams.confidence_tresh, tf.float32)

                    _detectors_mask = tf.equal(tf.minimum(_targets_conf_probs_bin, _outputs_conf_probs_bin), 1)
                    _detectors_mask = _detectors_mask[..., 0]
                    _targets_bboxes = tf.cast(targets_bboxes, tf.float32)
                    _targets_bboxes_xy = _targets_bboxes[..., :2]
                    _outputs_bboxes_xy = outputs_bboxes[..., :2]
                    _coordinates_scale = 5

                    self._add_xy_loss(_targets_bboxes_xy, _outputs_bboxes_xy, _detectors_mask, _coordinates_scale)

                    _targets_bboxes_wh = _targets_bboxes[..., 2:]
                    self._add_wh_loss(_targets_bboxes_wh, outputs_bboxes_wh_log, _detectors_mask, _coordinates_scale)

                    self._add_classes_loss(_targets_classes_probs, outputs_classes_probs, _targets_classes_one_hot, _detectors_mask)

            self.loss = tf.losses.get_total_loss()

    def compute_gradients(self):
        with tf.variable_scope('gradients'):
            params = tf.trainable_variables()
            grads = tf.gradients(self.loss, params)
            
            if self._hparams.grad_clip_value > 0:
                clipped_grads = []
                for grad in grads:
                    if grad is not None:
                        clipped_grad = tf.clip_by_norm(grad, self._hparams.grad_clip_value)
                        clipped_grads.append(clipped_grad)
                    else:
                        clipped_grads.append(None)
                
                return list(zip(clipped_grads, params))
            
            return list(zip(grads, params))
        
        
    def _mse(self, targets, predictions):
        a = tf.pow(targets - predictions, 2)
        return tf.reduce_mean(a) 
 