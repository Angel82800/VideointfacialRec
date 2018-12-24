import tensorflow as tf
import numpy as np


class Dataset:
    def __init__(self, train_metadata, valid_metadata, hparams):
        self._hparams = hparams
        self._is_classes_exist = True
        
        self._train_meta = self._load_metadata(train_metadata)
        self._valid_meta = self._load_metadata(valid_metadata)
        
        with tf.device('/cpu:0'):
            self._images = tf.placeholder(tf.string, shape=None, name='img_path')
            self._confs = tf.placeholder(tf.string, shape=[None, len(hparams.output_grids)], name='conf_paths')
            self._bboxes = tf.placeholder(tf.string, shape=[None, len(hparams.output_grids)], name='bbox_paths')
            
            if hparams.predict_classes:
                self._classes = tf.placeholder(tf.string, shape=[None, len(hparams.output_grids)], name='classes_path')
                dataset = tf.data.Dataset.from_tensor_slices((self._images, self._classes, self._confs, self._bboxes))
            else:
                dataset = tf.data.Dataset.from_tensor_slices((self._images, self._confs, self._bboxes))
                
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=hparams.buffer_size))
            dataset = dataset.map(self._classes_map) if hparams.predict_classes else dataset.map(self._map)
            dataset = dataset.batch(hparams.batch_size)
            dataset = dataset.prefetch(1)

            self._train_iterator = dataset.make_initializable_iterator()
            self._valid_iterator = dataset.make_initializable_iterator()
            
            step = 3 if hparams.predict_classes else 2
            train_batch = self._train_iterator.get_next()
            self.train_images = train_batch[0]
            train_batch = train_batch[1:]
            self.train_conf_probs = [train_batch[i] for i in range(0, len(train_batch), step)]
            self.train_bboxes = [train_batch[i] for i in range(1, len(train_batch), step)]

            if hparams.predict_classes:
                self.train_classes = [train_batch[i] for i in range(2, len(train_batch), step)]
            
            valid_batch = self._valid_iterator.get_next()
            self.valid_images = valid_batch[0]
            valid_batch = valid_batch[1:]
            self.valid_conf_probs = [valid_batch[i] for i in range(0, len(valid_batch), step)]
            self.valid_bboxes = [valid_batch[i] for i in range(1, len(valid_batch), step)]

            if hparams.predict_classes:
                self.valid_classes = [valid_batch[i] for i in range(2, len(valid_batch), step)]
            
    def initialize(self, sess):
        if self._is_classes_exist and self._hparams.predict_classes:
            images, confs, bboxes, classes = zip(*self._train_meta)
        
            sess.run(self._train_iterator.initializer, 
                     feed_dict={
                        self._images: images,
                        self._confs: confs, 
                        self._bboxes: bboxes,
                        self._classes: classes
                    })

            images, confs, bboxes, classes = zip(*self._valid_meta)

            sess.run(self._valid_iterator.initializer, 
                     feed_dict={
                        self._images: images,
                        self._confs: confs, 
                        self._bboxes: bboxes,
                        self._classes: classes 
                    })
        else:
            images, confs, bboxes = zip(*self._train_meta)

            sess.run(self._train_iterator.initializer, 
                     feed_dict={
                        self._images: images,
                        self._confs: confs, 
                        self._bboxes: bboxes
                    })

            images, confs, bboxes = zip(*self._valid_meta)

            sess.run(self._valid_iterator.initializer, 
                     feed_dict={
                        self._images: images,
                        self._confs: confs, 
                        self._bboxes: bboxes
                    })
        
    def _load_metadata(self, metadata_filename):
        metadata = []
        with open(metadata_filename, 'r') as f:
            for line in f:
                data = line.strip().split(';')
                conf = [data[i] for i in range(1, len(data), 3)]
                bbox = [data[i] for i in range(2, len(data), 3)]
                classes = [data[i] for i in range(3, len(data), 3)]

                if self._hparams.predict_classes:
                    metadata.append((data[0], conf, bbox, classes))
                else:
                    metadata.append((data[0], conf, bbox))
                
        return metadata
        
        
        
    def _py_map(self, img_path, conf_paths, bbox_paths):
        img = np.load(img_path.decode())

        sample = [img]
        for i in range(len(conf_paths)):
            conf = np.load(conf_paths[i].decode())
            bbox = np.load(bbox_paths[i].decode())

            sample += [conf, bbox]

        return sample

    def _map(self, img_path, conf_paths, bbox_paths):
        t_out = [tf.uint8] + ([tf.float32, tf.float32] * len(self._hparams.output_grids))
        sample = tf.py_func(self._py_map, [img_path, conf_paths, bbox_paths], t_out)

        sample[0].set_shape(self._hparams.input_image_size + [3])
        targets = sample[1:]

        for i, grid in enumerate(self._hparams.output_grids):
            conf_index = i * 2
            bbox_index = i * 2 + 1

            targets[conf_index].set_shape(grid+[1])
            targets[bbox_index].set_shape(grid+[4])

        return sample

    def _py_classes_map(self, img_path, classes_paths, conf_paths, bbox_paths):
        img = np.load(img_path.decode())

        sample = [img]
        for i in range(len(conf_paths)):
            conf = np.load(conf_paths[i].decode())
            bbox = np.load(bbox_paths[i].decode())
            _class = np.load(classes_paths[i].decode()).astype(np.float32)

            sample += [conf, bbox, _class]

        return sample

    def _classes_map(self, img_path, classes_paths, conf_paths, bbox_paths):
        t_out = [tf.uint8] + ([tf.float32, tf.float32, tf.float32] * len(self._hparams.output_grids))
        sample = tf.py_func(self._py_classes_map, [img_path, classes_paths, conf_paths, bbox_paths], t_out)

        sample[0].set_shape(self._hparams.input_image_size + [3])
        targets = sample[1:]

        for i, grid in enumerate(self._hparams.output_grids):
            conf_index = i * 3
            bbox_index = i * 3 + 1
            classes_index = i * 3 + 2

            targets[conf_index].set_shape(grid+[1])
            targets[bbox_index].set_shape(grid+[4])
            
            if self._hparams.n_classes == 1 or self._hparams.n_classes ==2:
                targets[classes_index].set_shape(grid+[1])
            else:
                targets[classes_index].set_shape(grid+[self._hparams.n_classes])

        return sample
