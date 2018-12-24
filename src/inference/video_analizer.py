from task import ProcessPhotosTask, ProcessVideoTaskResult, ProcessProtosTaskResult, ProcessVideoTaskResult
import time
import random
import numpy as np
import cv2

def sec_to_time(sec):
    mins = sec // 60
    sec = sec - int(mins * 60)

    return f'{mins:02d}:{sec:02d}'

class VideoAnalizer:
    def __init__(self):
        pass

    def process_protos(self, task):
        time.sleep(random.randint(10, 30))
        return ProcessProtosTaskResult(task.get_task_id(), np.random.rand(2048))

    def process_video(self, task):
        _fps = 30
        time.sleep(random.randint(10, 30))

        n_faces = random.randint(0, len(task.faces)+1)

        faces = []
        for i in range(n_faces):
            face_id = task.faces[i][0]
            intervals = []
            for j in range(random.randint(1, 10)):
                start_interval = random.randint(1, 500)
                finish_interval = random.randint(start_interval+40, 850)

                start_interval_sec = start_interval // _fps
                finish_interval_sec = finish_interval // _fps
                interval = (sec_to_time(start_interval_sec), sec_to_time(finish_interval_sec))
                intervals.append(interval)

            faces.append((face_id, intervals))

        result = ProcessVideoTaskResult(task.get_task_id(), faces)
        return result
