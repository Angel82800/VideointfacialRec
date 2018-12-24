import json
import random
import numpy as np
import os
from scipy.io import loadmat
from . import core
from scipy.misc import imread
from tqdm import tqdm

def _parse_json(filename):
    with open(filename, 'rt') as f:
        data = json.load(f)
        
        img_path = data[0]
        img = imread(img_path)
        
        labels = data[1]
        bboxes = []
        classes = []
        
        for label in labels:
            bbox = label['BoundingBox']
            
            width = bbox['Width'] * img.shape[1]
            height = bbox['Height'] * img.shape[0]
            left = bbox['Left'] * img.shape[1]
            top = bbox['Top'] * img.shape[0]
            
            x = int(left + width / 2)
            y = int(top + height / 2)
            w = int(width)
            h = int(height)
            
            _class = label['Gender']['Value']
            bboxes.append((x,y,w,h))
            classes.append(_class)
            
        
        return (img_path, bboxes, classes)
            
        
def get_loader(IMDB_dataset_path):
    _TEST_SPLIT_FACTOR = 0.9
    _VALID_SPLIT_FACTOR = 0.9
    print('Images loading...')
    
    dataset = []
    
    for root, folders, files in os.walk(IMDB_dataset_path):
        for file in tqdm(files):
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                data = _parse_json(file_path)
                dataset.append(data)
                
    dataset = [((image), (bboxes, classes)) for image, bboxes, classes in dataset]
                
    
    images, _ = zip(*dataset)
    images = list(images)
    
    DATA = dict(dataset)
    
    _split_index = int(len(images)*_TEST_SPLIT_FACTOR)
    images_dev = images[:_split_index]
    images_test = images[_split_index:]
    
    _split_index = int(len(images_dev)*_VALID_SPLIT_FACTOR)
    images_train = images_dev[:_split_index]
    images_valid = images_dev[_split_index:]
        
    random.shuffle(images_train)
    random.shuffle(images_valid)
    random.shuffle(images_test)
    
    def IMDB_new_dataset_loader(state):
        state = state.lower()

        def annotation_getter(file_path):
            bboxes, classes = DATA[file_path]
            return bboxes, classes

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

            for name in ['Male', 'Female']:
                core.get_class_id(name)
            
            return {'nclasses': core.get_amount_of_classes() - before_amount}

        raise ValueError('Incorrect `state`.')
        
    return IMDB_new_dataset_loader
