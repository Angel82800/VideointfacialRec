import boto3
from scipy.misc import imread
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from tqdm import tqdm

n_jobs = 8


def create_bucket(name):
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket=name)
    
    
def upload_img(img_path, key, bucket):
    s3 = boto3.resource('s3')
    data = open(img_path, 'rb')
    s3.Bucket(bucket).put_object(Key=key, Body=data)
    
    
def detect_faces(bucket_name, key):
    rekognition = boto3.client("rekognition", "us-east-1")
    response = rekognition.detect_faces(
        Image={
            "S3Object": {
                "Bucket": bucket_name,
                "Name": key,
            }
        },
        Attributes=['ALL'],
    )
    return response['FaceDetails']


def delete_bucked(name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name)
    for key in bucket.objects.all():
        key.delete()
    bucket.delete()
    
def delete_object(bicket_name, key):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bicket_name)
    bucket.Object(key).delete()
    
    
def process_img(bucket_name, key, img_path, target_path):
    try:
        img = imread(img_path)
        if img.shape[0] > 100 and img.shape[1] > 100:
            upload_img(img_path, key, bucket_name)
            faces = detect_faces(bucket_name, key)
            result = (img_path, faces)
            
            with open(target_path, 'wt') as f:
                json.dump(result, f)
                
            delete_object(bucket_name, key)
                
            return target_path
        return None

    except:
        return None
    
def main():
    base_dir = '/home/facialrec/notebooks/imdb/imdb/30'
    
    bucket_name = 'facialdet-bucket'
    create_bucket(bucket_name)
    
    
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []  
    
    i = 0
    for root, folders, files in os.walk(base_dir):
        for file in files:
            img_path = os.path.join(root, file)
            key = f'img_{i}.jpg'
            target_path = f'aws_results/30/{key}.json'
            futures.append(executor.submit(partial(process_img, bucket_name, key, img_path, target_path)))
            i += 1
            
    result = [future.result() for future in tqdm(futures) if future.result() is not None]
    
    delete_bucked(bucket_name)
    return result

if __name__ == '__main__':
    result = main()
    print(f'\n Total images processed: {len(result)}')

    
    
