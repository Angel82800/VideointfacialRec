import time
import random
random.seed(42)

class Task:
    def get_task_id(self):
        raise NotImplementedError()


class TaskResult(Task):
    def to_json(self):
        raise NotImplementedError()

class ProcessPhotosTask:
    def __init__(self, img_1, img_2, img_3):
        self.img_1 = img_1
        self.img_2 = img_2
        self.img_3 = img_3
        self._task_id = 'photosprocessing_%.5f_%d' % (time.time(), random.randint(0, 99999999999999))

    def get_task_id(self):
        return self._task_id

class ProcessVideoTaskResult(TaskResult):
    def __init__(self, task_id):
        self._task_id = task_id
        self._faces = {}

    def get_task_id(self):
        return self._task_id

    def add_interval(self, face_id, interval):
        if face_id in self._faces.keys():
            self._faces[face_id].append(interval)
        else:
            self._faces[face_id] = [interval]


    def to_json(self):
        json_result = {}
        json_result['task_id'] = self._task_id
        json_result['status'] = 'ok'
        json_result['faces'] = []

        for face_id, intervals in self._faces.items():
            formated_interval = []
            for interval in intervals:
                formated_interval.append([str(interval[0]), str(interval[1])])

            json_result['faces'].append(
                {
                    'face_id': face_id,
                    'intervals': formated_interval
                }
            )

        return json_result



class ProcessProtosTaskResult(TaskResult):
    def __init__(self, task_id, features):
        self._task_id = task_id
        self.features = features

    def get_task_id(self):
        return self._task_id

    def to_json(self):
        json_result = {}
        json_result['task_id'] = self._task_id
        json_result['status'] = 'ok'
        json_result['features'] = self.features.tolist()
        return json_result

    

class ProcessVideoTask:
    def __init__(self, video, faces):
        self.video = video
        self.faces = faces
        self._tasl_id = 'videoprocessing_%.5f_%d' % (time.time(), random.randint(0, 99999999999999))

    def get_task_id(self):
        return self._task_id

    