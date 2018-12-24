from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
from model.align import detect_face
from model import facenet
import cv2
import urllib
from task import ProcessProtosTaskResult, ProcessVideoTaskResult
import urllib.request

def frame_to_time(frame_number, fps=30):
    total_sec = int(frame_number / fps)

    mins = int(total_sec // 60)
    sec = total_sec - mins * 60

    return '%02d:%02d' % (mins, sec)


class FaceModel:
    def __init__(self, hparams):
        self._hparams = hparams
        self._face_detection_graph = tf.Graph()
        self._face_recongnition_grapth = tf.Graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self._hparams.gpu_memory_fraction)
        
        self._face_decetion_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False), graph=self._face_detection_graph)
        with self._face_detection_graph.as_default():
            self._face_detector = self._init_face_detection_model()
            
        self._face_recognition_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False), graph=self._face_recongnition_grapth)
        with self._face_recongnition_grapth.as_default():
            self._embeddings = self._init_face_recognition_model()


    def _init_face_recognition_model(self):
        facenet.load_model(self._hparams.path_to_model)
        # Get input and output tensors
        images_placeholder = self._face_recognition_sess.graph.get_tensor_by_name("input:0")
        embeddings = self._face_recognition_sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self._face_recognition_sess.graph.get_tensor_by_name("phase_train:0")

        

        def get_embeddings(images):
            return self._face_recognition_sess.run(embeddings, feed_dict={ images_placeholder: images, phase_train_placeholder:False })

        return get_embeddings


    def _init_face_detection_model(self):
        pnet, rnet, onet = detect_face.create_mtcnn(self._face_decetion_sess, None)

        def detect_faces(img):
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = detect_face.detect_face(img, self._hparams.minsize, pnet, rnet, onet, self._hparams.threshold, self._hparams.factor)
            
            if len(bounding_boxes) < 1:
                return []

            faces = []

            bounding_boxes = bounding_boxes[:,0:4]

            for det in bounding_boxes:
                bb = np.zeros((4,), dtype=np.int32)
                bb[0] = np.maximum(det[0]-self._hparams.margin/2, 0)
                bb[1] = np.maximum(det[1]-self._hparams.margin/2, 0)
                bb[2] = np.minimum(det[2]+self._hparams.margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+self._hparams.margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                aligned = misc.imresize(cropped, (self._hparams.image_size, self._hparams.image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                faces.append(prewhitened)

            faces = np.stack(faces)
            return faces

        return detect_faces

    def process_video(self, task):
        urllib.request.urlretrieve(task.video, 'video.mp4') 

        cap = cv2.VideoCapture('video.mp4')

        results = {}
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self._face_detector(frame)
                if len(faces) == 0:
                    continue
                embeddings = self._embeddings(np.array(faces))

                for face in task.faces:
                    for j in range(embeddings.shape[0]):
                        dist = np.sqrt(np.sum(np.square(np.subtract(np.array(face[1], dtype=np.float32), embeddings[j,:]))))
                        if dist < self._hparams.facialrec_threshold:
                            time = frame_to_time(i)

                            if face[0] in results.keys():
                                results[face[0]].append((time, i))
                            else:
                                results[face[0]] = [(time, i)]

        return self._video_postprocessing(task.get_task_id(), results)


    def _video_postprocessing(self, task_id, results):
        result = ProcessVideoTaskResult(task_id)

        for face_id, frames in results.items():
            start_interval = frames[0]
            finish_interval = frames[0]
            for frame in frames[1:]:
                if finish_interval[1] + 1 ==  frame[1]:
                    finish_interval = frame
                else:
                    result.add_interval(face_id, (start_interval[0], finish_interval[0]))
                    start_interval = frame
                    finish_interval = frame

        return result
                    
    def _load_image_from_url(self, url):
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        return img

    def process_photos(self, task):
        img_1 = self._load_image_from_url(task.img_1)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

        img_2 = self._load_image_from_url(task.img_2)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

        img_3 = self._load_image_from_url(task.img_3)
        img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)

        faces_1 = self._face_detector(img_1)
        faces_2 = self._face_detector(img_2)
        faces_3 = self._face_detector(img_3)

        stacked_faces = np.stack([faces_1[0], faces_2[0], faces_3[0]])

        emb = self._embeddings(stacked_faces)
        
        task_result = ProcessProtosTaskResult(task.get_task_id(), emb[0])
        return task_result


        