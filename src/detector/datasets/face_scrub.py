import json
import imghdr
import random
import os
from . import core


def get_loader(_FaceScrub_dataset_path):
    _TEST_SPLIT_FACTOR = 0.9
    _VALID_SPLIT_FACTOR = 0.9
    
    def _load_images_list(folder):
        files = os.listdir(os.path.join(_FaceScrub_dataset_path, folder))
        files = filter(lambda x: x.endswith('.jpg'), files)
        files = [os.path.join(_FaceScrub_dataset_path, folder, file) for file in files]
        files = filter(lambda x: os.path.exists(os.path.splitext(x)[0] + '.txt'), files)
        return list(files)

    _FaceScrub_dataset_male = _load_images_list('facescrub_actors')
    _FaceScrub_dataset_female = _load_images_list('facescrub_actresses')

    _FaceScrub_dataset_male = sorted(_FaceScrub_dataset_male)
    _FaceScrub_dataset_female = sorted(_FaceScrub_dataset_female)
    
    test_size = int(len(_FaceScrub_dataset_male)*(1-_TEST_SPLIT_FACTOR))
    step = len(_FaceScrub_dataset_male) // test_size
    
    _FaceScrub_dataset_male_dev = [_FaceScrub_dataset_male[i] for i in range(len(_FaceScrub_dataset_male)) if i % step != 0]
    _FaceScrub_dataset_male_test = [_FaceScrub_dataset_male[i] for i in range(0, len(_FaceScrub_dataset_male), step)]
    
    # random.shuffle(_FaceScrub_dataset_male_dev)
    
    _split_index = int(len(_FaceScrub_dataset_male_dev)*_VALID_SPLIT_FACTOR)
    _FaceScrub_dataset_male_train = _FaceScrub_dataset_male_dev[:_split_index]
    _FaceScrub_dataset_male_valid = _FaceScrub_dataset_male_dev[_split_index:]
    
    test_size = int(len(_FaceScrub_dataset_female)*(1-_TEST_SPLIT_FACTOR))
    step = len(_FaceScrub_dataset_female) // test_size
    
    _FaceScrub_dataset_female_dev = [_FaceScrub_dataset_female[i] for i in range(len(_FaceScrub_dataset_female)) if i % step != 0]
    _FaceScrub_dataset_female_test = [_FaceScrub_dataset_female[i] for i in range(0, len(_FaceScrub_dataset_female), step)]
    
    # random.shuffle(_FaceScrub_dataset_female_dev)
    
    _split_index = int(len(_FaceScrub_dataset_female)*_VALID_SPLIT_FACTOR)
    _FaceScrub_dataset_female_train = _FaceScrub_dataset_female[:_split_index]
    _FaceScrub_dataset_female_valid = _FaceScrub_dataset_female[_split_index:]

    _FaceScrub_dataset_train = _FaceScrub_dataset_male_train + _FaceScrub_dataset_female_train
    _FaceScrub_dataset_valid = _FaceScrub_dataset_male_valid + _FaceScrub_dataset_female_valid
    _FaceScrub_dataset_test = _FaceScrub_dataset_male_test + _FaceScrub_dataset_female_test

    random.shuffle(_FaceScrub_dataset_train)
    random.shuffle(_FaceScrub_dataset_valid)
    random.shuffle(_FaceScrub_dataset_test)

    def FaceScrub_dataset_loader(state):
        state = state.lower()

        def annotation_getter(file_path):
            file_name, ext = os.path.splitext(file_path)

            bbox_file = file_name + '.txt'

            try:
                with open(os.path.join(_FaceScrub_dataset_path, bbox_file), 'r') as f:
                    bbox = f.read().strip().split(' ')
                    bbox = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            except Exception as ex:
                print(repr(ex))
                return None

            if 'facescrub_actors' in file_path:
                cat = 'male'
            elif 'facescrub_actresses' in file_path:
                cat = 'female'
            else:
                return None

            return [bbox], [cat]

        if state == 'training':
            return _FaceScrub_dataset_train
        if state == 'valid':
            return _FaceScrub_dataset_valid
        if state == 'testing':        
            return _FaceScrub_dataset_test
        if state == 'annotation_getter':
            return annotation_getter
        if state == 'info':
            before_amount = core.get_amount_of_classes()

            for name in ['male', 'female']:
                core.get_class_id(name)

            return {'nclasses': core.get_amount_of_classes() - before_amount}

        raise ValueError('Incorrect `state`.')
        
    return FaceScrub_dataset_loader
