import argparse
import os
import numpy as np
import random
from imageio import imread
from detector.utils.data import convert_sample_to_YOLO_preresized, smart_resize
import detector.datasets
from detector.datasets import REGISTERED_CLASSES, reset_classes, get_class_id
from tqdm import tqdm
from hparams import hparams


def preprocess_img(entry, targets, hparams):
    try:
        if isinstance(entry, bytes):
            entry = entry.decode()

        image = imread(entry)
        if image.ndim == 2:
            image = image[..., np.newaxis]
            image = np.tile(image, [1, 1, 3])
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :-1]
        elif image.ndim != 3 or image.shape[-1] != 3:
            return None                        

        if targets is None:
            return None

        bboxes, classes = targets
        classes = [get_class_id(c) for c in classes]
        
        result = smart_resize(image, hparams.input_image_size[0], hparams.input_image_size[1], ret_image_shifts=True)
        image, resize_params = result
        
        sample = [image]

        for output_grid in hparams.output_grids:
            classes = classes if hparams.predict_classes else None
            result = convert_sample_to_YOLO_preresized(image, resize_params, output_grid, bboxes, classes=classes)
                

            if result is None:
                return None

            #conf_probs_map, bboxes_map, classes_map = result
            sample = sample + list(result)

        return sample

    except Exception as ex:
        print(ex)
        return None
    
def process_dataset(samples, output_dir, img_dir, grid_dirs, dataset_name, dataset_getters, hparams):
    with open(os.path.join(output_dir, dataset_name, 'metadata.csv'), 'w') as f:
        for i, (dataset_id, img_path) in tqdm(enumerate(samples)):
            target = dataset_getters[dataset_id](img_path)

            sample = preprocess_img(img_path, target, hparams)
            if sample is not None:
                img = sample[0]
                img_filename = os.path.join(img_dir, f'img_{i}.npy')
                np.save(img_filename, img)
                row = img_filename

                r = range(1, len(hparams.output_grids) * 3 + 1, 3) if hparams.predict_classes else range(1, len(hparams.output_grids) * 2 + 1, 2)
                for grid_num, j in enumerate(r):
                    conf_map = sample[j]
                    bboxes = sample[j+1]

                    conf_filename = os.path.join(grid_dirs[grid_num], f'conf_{i}.npy')
                    bbox_filename = os.path.join(grid_dirs[grid_num], f'bbox_{i}.npy')

                    row += ';'+conf_filename+';'+bbox_filename
                    np.save(conf_filename, conf_map)
                    np.save(bbox_filename, bboxes)
                    
                    if hparams.predict_classes:
                        classes = sample[j+2]
                        classes_filename = os.path.join(grid_dirs[grid_num], f'classes_{i}.npy')
                        row += ';'+classes_filename
                        np.save(classes_filename, classes)
                    else:
                        row += ';None'

                row += '\n'
                f.write(row)
        
            
def preprocess(args, hparams):
    output_dir = args.output_dir
    train_img_path = os.path.join(output_dir, 'train', 'imgs')
    train_grid_paths = [os.path.join(output_dir, 'train', f'grid_{i}') for i in range(len(hparams.output_grids))]
    
    os.makedirs(train_img_path, exist_ok=True)
    #os.makedirs(train_class_path, exist_ok=True)
    for train_grid_path in train_grid_paths:
        os.makedirs(train_grid_path, exist_ok=True)
        
    valid_img_path = os.path.join(output_dir, 'valid', 'imgs')
    valid_class_path = os.path.join(output_dir, 'valid', 'classes')
    valid_grid_paths = [os.path.join(output_dir, 'valid', f'grid_{i}') for i in range(len(hparams.output_grids))]
    
    os.makedirs(valid_img_path, exist_ok=True)
    os.makedirs(valid_class_path, exist_ok=True)
    for valid_grid_path in valid_grid_paths:
        os.makedirs(valid_grid_path, exist_ok=True)
        
    datasets = []
    for dataset in args.datasets:
        if dataset == 'WIDER':
            datasets.append(detector.datasets.WIDER_get_loader(os.path.join(hparams.dataset_path, 'WIDER')))
        elif dataset == 'faceScrub':
            datasets.append(detector.datasets.FaceScrub_get_loader(os.path.join(hparams.dataset_path, 'faceScrub')))
        elif dataset == 'imdb':
            datasets.append(detector.datasets.IMDB_get_loader(os.path.join(hparams.dataset_path, 'imdb')))
        elif dataset == 'imdb_wiki':
            datasets.append(detector.datasets.IMDB_WIKI_get_loader(os.path.join(hparams.dataset_path, 'wiki')))
        elif dataset == 'imdb_new':
            datasets.append(detector.datasets.IMDB_new_dataset_loader(os.path.join(hparams.dataset_path, 'aws_results')))
            
    reset_classes()
    
    print(datasets)
    
    dataset_getters = {i: d('annotation_getter') for i, d in enumerate(datasets)}
    for item in datasets:
        item('info')
        
    train_samples = [(i, entry) for i, d in enumerate(datasets) for entry in d('training')]
    valid_samples = [(i, entry) for i, d in enumerate(datasets) for entry in d('valid')]
    
    print('The number of training samples:  ', len(train_samples))
    print('The number of validation samples:', len(valid_samples))
    
    process_dataset(train_samples, output_dir, train_img_path, train_grid_paths, 'train', dataset_getters, hparams)
    process_dataset(valid_samples, output_dir, valid_img_path, valid_grid_paths, 'valid', dataset_getters, hparams)
                


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--output_dir', default='../data/', help='Folder to contain preprocessed datasets')
    parser.add_argument('--datasets', default='imdb_new', help='Folder to contain preprocessed datasets')

    args = parser.parse_args()
    
    args.datasets = args.datasets.split(',')
    print(args.datasets)
    new_hparams = hparams.parse(args.hparams)
    preprocess(args, new_hparams)
        
        
if __name__ == '__main__':
    main()
    
    
    
    