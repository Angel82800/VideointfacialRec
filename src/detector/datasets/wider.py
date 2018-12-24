import json
import random
import numpy as np
import os
from scipy.io import loadmat
from . import core

def load_markup(file_path):
    with open(file_path, 'r') as f:
        data = []
        while True:
            file_name = f.readline().strip()
            if not file_name:
                break
            n = int(f.readline().strip())
            bboxes = []
            for i in range(n):
                sample = f.readline().strip().split(' ')
                x = int(sample[0])
                y = int(sample[1])
                w = int(sample[2])
                h = int(sample[3])
                
                bboxes.append((x,y,w,h))
            data.append((file_name, bboxes))
        return data

def get_loader(WIDER_dataset_path):
    _TEST_SPLIT_FACTOR = 0.9
    _VALID_SPLIT_FACTOR = 0.9
    
    train_data = load_markup(os.path.join(WIDER_dataset_path, 'wider_face_split', 'wider_face_train_bbx_gt.txt'))
    valid_data = load_markup(os.path.join(WIDER_dataset_path, 'wider_face_split', 'wider_face_val_bbx_gt.txt'))
    
    train_data = [(os.path.join(WIDER_dataset_path, 'WIDER_train', 'images', image), (bboxes)) for image, bboxes in train_data]
    valid_data = [(os.path.join(WIDER_dataset_path, 'WIDER_val', 'images', image), (bboxes)) for image, bboxes in valid_data]
    
    images_train, _ = zip(*train_data)
    images_train = list(images_train)
    
    images_valid, _ = zip(*valid_data)
    images_valid = list(images_valid)
    
    DATA = dict(train_data + valid_data)
    
    random.shuffle(images_train)
    random.shuffle(images_valid)
    
    def WIDER_dataset_loader(state):
        state = state.lower()

        def annotation_getter(file_path):
            bboxes = DATA[file_path]
            bboxes = [(int(x),int(y),int(w),int(h)) for x,y,w,h in bboxes]
            return bboxes, None

        if state == 'training':
            return images_train
        if state == 'valid':
            return images_valid
        if state == 'testing':        
            return []
        if state == 'annotation_getter':
            return annotation_getter
        if state == 'info':
            return {'nclasses': 0}

        raise ValueError('Incorrect `state`.')
        
    return WIDER_dataset_loader
