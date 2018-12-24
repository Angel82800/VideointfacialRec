import tensorflow as tf
import numpy as np
from hparams import hparams
import importlib
import json
import os
from model import Yolo
from dataset import Dataset
import argparse
import time
from detector.utils.plot import plot_confusion_matrix

def get_summary_op(train_losses, test_losses, learning_rate, grads, is_training):
    train_summaries = []
    test_summaries = []
    loss_names = [loss.name for loss in train_losses]
    losses = tf.cond(is_training, true_fn=lambda: train_losses, false_fn=lambda: test_losses)

    for loss, name in zip(losses, loss_names):
        s = tf.summary.scalar(f'/losses/{name}', loss)
        train_summaries.append(s)
        test_summaries.append(s)
        
    train_summaries.append(tf.summary.scalar('learning_rate', learning_rate))
    
    #with tf.variable_scope('gradients'):
    #    gradient_norms = []
    #    for grad, var in grads:
    #        if grad is not None:
    #            gradient_norms.append(tf.norm(grad))

     #   train_summaries.append(tf.summary.histogram('/gradient_norm', gradient_norms))
     #   train_summaries.append(tf.summary.scalar('/max_gradient_norm', tf.reduce_max(gradient_norms)))
    
    train_summary = tf.summary.merge(train_summaries)
    test_summary = tf.summary.merge(test_summaries)
    
    summary_op = tf.cond(is_training, true_fn=lambda: train_summary, false_fn=lambda: test_summary)
    return summary_op

def get_eval_summary(train_outputs, test_outputs, is_training, hparams):
    summaries = []
    outputs = tf.cond(is_training, true_fn=lambda: train_outputs, false_fn=lambda: test_outputs)
    targets_nodes, output_nodes = outputs
    
    with tf.name_scope('metrics'):
        conf_probs_accuracy = []
        conf_probs_precision = []
        conf_probs_recall = []
        conf_probs_f1_score = []
        
        bboxes_IoU = []
        
        for output_grid, _targets_nodes, _outputs_nodes in zip(hparams.output_grids, targets_nodes, output_nodes):
            with tf.name_scope('grid-%ix%i' % tuple(output_grid)):
                targets_conf_probs, targets_bboxes = _targets_nodes
                if hparams.predict_classes:
                    ((outputs_conf_probs_logits, outputs_conf_probs),
                     (outputs_bboxes_wh_log, outputs_bboxes), _) = _outputs_nodes
                else:
                    ((outputs_conf_probs_logits, outputs_conf_probs),
                     (outputs_bboxes_wh_log, outputs_bboxes)) = _outputs_nodes
                    
                _targets_conf_probs = tf.cast(targets_conf_probs, tf.float32)
                _targets_bboxes = tf.cast(targets_bboxes, tf.float32)
                _targets_bboxes_xy = _targets_bboxes[..., :2]
                _targets_bboxes_wh = _targets_bboxes[..., 2:]
                    
                _targets_conf_probs_bin = tf.cast(_targets_conf_probs >= hparams.confidence_tresh, tf.float32)
                _outputs_conf_probs_bin = tf.cast(outputs_conf_probs >= hparams.confidence_tresh, tf.float32)
                        
                _detectors_mask = tf.equal(tf.minimum(_targets_conf_probs_bin, _outputs_conf_probs_bin), 1)
                _detectors_mask = _detectors_mask[..., 0]

                _targets_bboxes_xy_actual = tf.boolean_mask(_targets_bboxes_xy, _detectors_mask)
                _targets_bboxes_wh_actual = tf.boolean_mask(_targets_bboxes_wh, _detectors_mask)

                _outputs_bboxes_xy_actual = tf.boolean_mask(outputs_bboxes[..., :2], _detectors_mask)
                _outputs_bboxes_wh_log_actual = tf.boolean_mask(outputs_bboxes_wh_log, _detectors_mask)
                _outputs_bboxes_wh_actual = tf.boolean_mask(outputs_bboxes[..., 2:], _detectors_mask)
                
                # Confidence accuracy

                _targets_conf_probs_bin_flatten = tf.reshape(_targets_conf_probs_bin, [-1])
                outputs_conf_probs_bin_flatten = tf.reshape(_outputs_conf_probs_bin, [-1])

                _true_positives = tf.reduce_sum(tf.minimum(_targets_conf_probs_bin_flatten, outputs_conf_probs_bin_flatten))
                _true_negatives = tf.reduce_sum(tf.minimum(1-_targets_conf_probs_bin_flatten, 1-outputs_conf_probs_bin_flatten))
                _false_positives = tf.reduce_sum(tf.minimum(1-_targets_conf_probs_bin_flatten, outputs_conf_probs_bin_flatten))
                _false_negatives = tf.reduce_sum(tf.minimum(_targets_conf_probs_bin_flatten, 1-outputs_conf_probs_bin_flatten))

                conf_probs_accuracy.append(tf.reduce_mean((_true_positives+_true_negatives)/tf.maximum(_true_positives+_false_positives+_false_negatives+_true_negatives, 1e-9)))
                conf_probs_precision.append(tf.reduce_mean(_true_positives/tf.maximum(_true_positives+_false_positives, 1e-9)))
                conf_probs_recall.append(tf.reduce_mean(_true_positives/tf.maximum(_true_positives+_false_negatives, 1e-9)))
                conf_probs_f1_score.append(2*(conf_probs_precision[-1] * conf_probs_recall[-1]) / tf.maximum(conf_probs_precision[-1] + conf_probs_recall[-1], 1e-9))

                # IoU

                # intersection-over-union

                # correction of negative values of bboxes

                _targets_bboxes_xy_actual_corrected = tf.maximum(_targets_bboxes_xy_actual, 0)
                _targets_bboxes_wh_actual_corrected = tf.maximum(_targets_bboxes_wh_actual, 0)
                _outputs_bboxes_xy_actual_corrected = tf.maximum(_outputs_bboxes_xy_actual, 0)
                _outputs_bboxes_wh_actual_corrected = tf.maximum(_outputs_bboxes_wh_actual, 0)

                _targets_bboxes_wh_actual_half = _targets_bboxes_wh_actual_corrected / 2.
                _targets_mins  = _targets_bboxes_xy_actual_corrected - _targets_bboxes_wh_actual_half
                _targets_maxes = _targets_bboxes_xy_actual_corrected + _targets_bboxes_wh_actual_half

                _outputs_bboxes_wh_actual_half = _outputs_bboxes_wh_actual_corrected / 2.
                _outputs_mins  = _outputs_bboxes_xy_actual_corrected - _outputs_bboxes_wh_actual_half
                _outputs_maxes = _outputs_bboxes_xy_actual_corrected + _outputs_bboxes_wh_actual_half       

                _intersect_mins = tf.maximum(_targets_mins, _outputs_mins)
                _intersect_maxes = tf.minimum(_targets_maxes, _outputs_maxes)
                _intersect_wh = tf.maximum(0., _intersect_maxes - _intersect_mins)

                _intersect_areas = _intersect_wh[..., 0] * _intersect_wh[..., 1]

                _targets_areas = _targets_bboxes_wh_actual_corrected[..., 0] * _targets_bboxes_wh_actual_corrected[..., 1]
                _outputs_areas = _outputs_bboxes_wh_actual_corrected[..., 0] * _outputs_bboxes_wh_actual_corrected[..., 1]

                _union_areas = _targets_areas + _outputs_areas - _intersect_areas

                _IoU_scores = tf.expand_dims(_intersect_areas / tf.maximum(_union_areas, 1e-6), axis=-1)
                _IoU_scores = tf.maximum(tf.minimum(_IoU_scores, 1), 0)

                _bboxes_IoU = tf.reduce_mean(_IoU_scores)
                bboxes_IoU.append(tf.cond(tf.is_finite(_bboxes_IoU), lambda: _bboxes_IoU, lambda: tf.constant(0, tf.float32)))


        conf_probs_accuracy_mean = tf.reduce_mean(conf_probs_accuracy, axis=0)
        conf_probs_precision_mean = tf.reduce_mean(conf_probs_precision, axis=0)
        conf_probs_recall_mean = tf.reduce_mean(conf_probs_recall, axis=0)
        conf_probs_f1_score_mean = tf.reduce_mean(conf_probs_f1_score, axis=0)
        
        bboxes_IoU_mean = tf.reduce_mean(bboxes_IoU, axis=0)
        
        summaries.append(tf.summary.scalar('conf-probs/precision', conf_probs_precision_mean))
        summaries.append(tf.summary.scalar('conf-probs/recall', conf_probs_recall_mean))
        summaries.append(tf.summary.scalar('conf-probs/f1-score', conf_probs_f1_score_mean))  

        for output_grid, precision, recall, f1_score in zip(hparams.output_grids, conf_probs_precision, conf_probs_recall, conf_probs_f1_score):
            summaries.append(tf.summary.scalar('conf-probs/grid-%ix%i/precision' % tuple(output_grid), precision))
            summaries.append(tf.summary.scalar('conf-probs/grid-%ix%i/recall' % tuple(output_grid), recall))
            summaries.append(tf.summary.scalar('conf-probs/grid-%ix%i/f1-score' % tuple(output_grid), f1_score))

        summaries.append(tf.summary.scalar('bboxes/IoU', bboxes_IoU_mean))

        for output_grid, iou in zip(hparams.output_grids, bboxes_IoU):
            summaries.append(tf.summary.scalar('bboxes/grid-%ix%i/IoU' % tuple(output_grid), iou))
            
    for output_grid, _targets_nodes, _outputs_nodes in zip(hparams.output_grids, targets_nodes, output_nodes):
        _targets_conf_probs = tf.cast(_targets_nodes[0], tf.float32)
        _targets_conf_probs_bin = tf.cast(_targets_conf_probs >= hparams.confidence_tresh, tf.float32)

        _outputs_conf_probs = tf.cast(_outputs_nodes[0][1], tf.float32)
        _outputs_conf_probs_bin = tf.cast(_outputs_conf_probs >= hparams.confidence_tresh, tf.float32)

        summaries.append(tf.summary.image('conf-probs/grid-%ix%i/map' % tuple(output_grid), tf.cast(_outputs_conf_probs*255, tf.uint8)))
        summaries.append(tf.summary.image('conf-probs/grid-%ix%i/map-output' % tuple(output_grid), tf.cast(_outputs_conf_probs_bin*255, tf.uint8)))
        summaries.append(tf.summary.image('conf-probs/grid-%ix%i/map-target' % tuple(output_grid), tf.cast(_targets_conf_probs_bin*255, tf.uint8)))

        #summaries.append(tf.summary.image('images', inputs_image))
    
    summary_op = tf.summary.merge(summaries)
    return summary_op

def get_train_model(dataset, global_step, hparams):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.name_scope('train') as scope:
            model = Yolo(hparams)
            model.initialize(dataset.train_images, True, targets_conf_probs=dataset.train_conf_probs, targets_bboxes=dataset.train_bboxes, targets_classes=dataset.train_classes)
            model.add_loss()
            gradvars = model.compute_gradients()
            losses = tf.losses.get_losses(scope)
            
            restore_op = model.restore_op

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
    with tf.variable_scope('optimizer') as scope:
        if hparams.decay_learning_rate:
            decay_steps = hparams.decay_steps
            decay_rate = hparams.decay_rate
            init_lr = hparams.initial_learning_rate
            final_lr = hparams.final_learning_rate
            lr = tf.train.exponential_decay(init_lr, global_step - hparams.start_decay, decay_steps, decay_rate,  name='lr_exponential_decay')
            learning_rate = tf.minimum(tf.maximum(lr, final_lr), init_lr)
        else:
            learning_rate = tf.convert_to_tensor(hparams.initial_learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate, hparams.adam_beta1,
            hparams.adam_beta2, hparams.adam_epsilon)

        train_op = [ 
        optimizer.apply_gradients(
            gradvars, global_step=global_step)
        ]
        train_op.extend(update_ops)
        train_op = tf.group(*train_op)
        
    outputs = [model.targets_nodes, model.output_nodes]
    #losses = [model.loss, model.conf_probs_loss, model.wh_loss, model.xy_loss]
    

    return outputs, losses, train_op, gradvars, learning_rate, restore_op

def get_test_model(dataset, hparams):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.name_scope('test') as scope:
            model = Yolo(hparams)
            model.initialize(dataset.valid_images, False, targets_conf_probs=dataset.valid_conf_probs, targets_bboxes=dataset.valid_bboxes, targets_classes=dataset.valid_classes)
            model.add_loss()

            outputs = [model.targets_nodes, model.output_nodes]
            #losses = [model.loss, model.conf_probs_loss, model.wh_loss, model.xy_loss]
            losses = tf.losses.get_losses(scope)
            return outputs, losses

        

def train(args, log_dir, hparams):
    train_metadata = os.path.join(args.base_dir, 'train', 'metadata.csv')
    valid_metadata = os.path.join(args.base_dir, 'valid', 'metadata.csv')
    save_dir = os.path.join(args.logdir, 'saved_models/')
    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = Dataset(train_metadata, valid_metadata, hparams)
    
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    train_outputs, train_losses, train_op, gradvars, learning_rate, restore_op = get_train_model(dataset, global_step, hparams)
    
    print('Создали тренировочную модель')
    test_outputs, test_losses = get_test_model(dataset, hparams)
    print('Создали тестовую модель')
    
    is_training = tf.placeholder(tf.bool, name='is_training')
    summary_op = get_summary_op(train_losses, test_losses, learning_rate, gradvars, is_training)
    #eval_summary_op = get_eval_summary(train_outputs, test_outputs, is_training, hparams)
    loss = tf.losses.get_total_loss()
    saver = tf.train.Saver(max_to_keep=5)
    print(f'YOLO training set to a maximum of {args.train_steps} steps')
    
    with tf.Session() as sess:
        train_summary_writer = tf.summary.FileWriter(os.path.join(args.logdir, 'train'), sess.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(args.logdir, 'test'))

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
    
        try:
            print('Restoring original model...', end='')
            restore_op(sess)
            print('[OK]')
        except:
            print('[FAILED]')
        
        #saved model restoring
        if args.restore:
            #Restore saved model if the user requested it, Default = True.
            try:
                checkpoint_state = tf.train.get_checkpoint_state(save_dir)
            except tf.errors.OutOfRangeError as e:
                print(f'Cannot restore checkpoint: {e}')

        if (checkpoint_state and checkpoint_state.model_checkpoint_path):
            print(f'Loading checkpoint {checkpoint_state.model_checkpoint_path}')
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

        else:
            if not args.restore:
                print('Starting new training!')
            else:
                print(f'No model to load at {save_dir}')
        
        dataset.initialize(sess)
        step = 0
        while step < args.train_steps:
            start_fime = time.time()
            step, _losses, _ = sess.run([global_step, train_losses, train_op])
            iter_time = time.time() - start_fime
            
            losses_str = ''
            for _loss in _losses:
                losses_str += f'{_loss:.3f}  '
            print(f'\rStep {step}/{args.train_steps} loss: {losses_str}, {iter_time:.3f} sec/step', end='', flush=True)
                        
            if step % args.summary_interval == 0:
                print(f'\nWriting summary at step {step}')
                train_summary_writer.add_summary(sess.run(summary_op, feed_dict={is_training: True}), step)
                train_summary_writer.flush()
                test_summary_writer.add_summary(sess.run(summary_op, feed_dict={is_training: False}), step)
                test_summary_writer.flush()
                                
            ''' 
            if step % args.eval_interval == 0:
                print(f'\nWriting eval summary at step {step}')
                train_summary_writer.add_summary(sess.run(eval_summary_op, feed_dict={is_training: True}), step)
                train_summary_writer.flush()
                test_summary_writer.add_summary(sess.run(eval_summary_op, feed_dict={is_training: False}), step)
                test_summary_writer.flush()'''
                
            if step % args.checkpoint_interval == 0:
                print(f'\nSaving checkpoing at step {step}')
                saver.save(sess, checkpoint_path, global_step=step)
        
def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	log_dir = args.logdir
	os.makedirs(log_dir, exist_ok=True)
	return log_dir, modified_hp
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='../data')
    parser.add_argument('--hparams', default='', 
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--logdir', default='logdir')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, 
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Steps between eval on test data')
    parser.add_argument('--summary_interval', type=int, default=250,
                        help='Steps between running summary ops') 
    parser.add_argument('--train_steps', type=int, default=360000, help='total number of wavenet training steps')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
       
    args = parser.parse_args()
    
    log_dir, hparams = prepare_run(args)
    train(args, log_dir, hparams)

if __name__ == '__main__':
    main()
