from flask import Flask, jsonify
from flask import request
import queue
from task import ProcessVideoTask, ProcessPhotosTask, ProcessVideoTaskResult, ProcessProtosTaskResult
import numpy as np
# from video_analizer import VideoAnalizer
import time
import threading
from model.model import FaceModel
from hparams import hparams

analizer = FaceModel(hparams)
app = Flask(__name__)

task_queue = queue.Queue()

results = {}


@app.route('/process_photos', methods=['POST'])
def process_photos():
    task = ProcessPhotosTask(request.json['img_1'], request.json['img_2'], request.json['img_3'])
    task_id = task.get_task_id()
    results[task_id] = None

    task_queue.put(task)
    return  jsonify({'task_id': task_id})


@app.route('/process_video', methods=['POST'])
def process_video():
    video_url = request.json['video_url']
    faces_json = request.json['faces']

    faces = []
    for face in faces_json:
        faces.append((face['face_id'], np.array(face['features'])))

    task = ProcessVideoTask(video_url, faces)
    task_id = task.get_task_id()
    results[task_id] = None
    task_queue.put(task)
    return  jsonify({'task_id': task_id})


@app.route('/get_result/<string:task_id>', methods=['GET'])
def get_result(task_id):
    active_tasks = list(results.keys())
    if task_id not in active_tasks:
        return jsonify({'task_id': task_id, 'status': 'not found'})
    else:
        if results[task_id] is None:
            return jsonify({'task_id': task_id, 'status': 'processing'})
        else:
            # task_result = results[task_id]
            task_result = results.pop(task_id)
            if isinstance(task_result, ProcessVideoTaskResult):
                videoproc_result = {}
                videoproc_result['task_id'] = task_result.get_task_id()
                videoproc_result['status'] = 'ok'
                videoproc_result['faces'] = []

                for face in task_result._faces:
                    face_id = face[0]

                    intervals = []
                    for interval in face[1]:
                        intervals.append([str(interval[0]), str(interval[1])])

                    videoproc_result['faces'].append(
                       {
                           'face_id': face_id,
                           'intervals': intervals
                       }
                    )
                    print('intervals added')

                return jsonify(videoproc_result)

            else:
                photosproc_result = {}
                photosproc_result['task_id'] = task_result.get_task_id()
                photosproc_result['status'] = 'ok'
                photosproc_result['features'] = task_result.features.tolist()
                
                return jsonify(photosproc_result)
            

def worker():
    while True:
        if not task_queue.empty():
            task = task_queue.get()

            # try:
            if isinstance(task, ProcessPhotosTask):
                result = analizer.process_photos(task)
            else:
                result = analizer.process_video(task)

            results[result.get_task_id()] = result
            task_queue.task_done()
            # except Exception as e:
            #     print(e)
        else:
            time.sleep(1)


# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    thread = threading.Thread(target=worker)
    thread.start()
    app.run(host='0.0.0.0', debug=True, port=7034)

