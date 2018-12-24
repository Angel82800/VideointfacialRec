import json
import random
import numpy as np
import os
from scipy.io import loadmat
from . import core


def get_loader(IMDB_dataset_path):
    _TEST_SPLIT_FACTOR = 0.9
    _VALID_SPLIT_FACTOR = 0.9
    
    meta = loadmat(os.path.join(IMDB_dataset_path, 'imdb.mat'), squeeze_me=True)
    
    _gener_mapper = { 0:'female', 1:'male' }
    
    zipped = zip(meta['imdb']['full_path'].item(),
                 meta['imdb']['face_score'].item(),
                 meta['imdb']['face_location'].item(),
                 meta['imdb']['gender'].item())
    
    filtered = filter(lambda x: (not np.isinf(x[1])) and (not np.isnan(x[3])), zipped)
    data = [(os.path.join(IMDB_dataset_path, image), (bbox, _gener_mapper[gender])) for image, _, bbox, gender in filtered]
    
    images, _ = zip(*data)
    images = list(images)
    
    DATA = dict(data)
    
    _split_index = int(len(images)*_TEST_SPLIT_FACTOR)
    images_dev = images[:_split_index]
    images_test = images[_split_index:]
    
    _split_index = int(len(images_dev)*_VALID_SPLIT_FACTOR)
    images_train = images_dev[:_split_index]
    images_valid = images_dev[_split_index:]
    
    random.shuffle(images_train)
    random.shuffle(images_valid)
    random.shuffle(images_test)
    
    def IMDB_dataset_loader(state):
        state = state.lower()

        def annotation_getter(file_path):
            bbox, cat = DATA[file_path]
            bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1]))
            return [bbox], [cat]

        if state == 'training':
            return images_train
        if state == 'valid':
            return images_valid
        if state == 'testing':        
            return images_test
        if state == 'annotation_getter':
            return annotation_getter
        if state == 'info':
            before_amount = core.get_amount_of_classes()

            for name in ['male', 'female']:
                core.get_class_id(name)
            
            return {'nclasses': core.get_amount_of_classes() - before_amount}

        raise ValueError('Incorrect `state`.')
        
    return IMDB_dataset_loader
