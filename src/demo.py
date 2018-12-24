import numpy as np
from skimage.transform import resize as imresize


def smart_resize(image, target_height, target_width, add_noise=False, ret_image_shifts=False):
    if image.ndim == 2:
        image = image[..., np.newaxis]
    
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, -1)
    
    h, w = image.shape[:-1]
    m_size = max(h, w)
    
    h_factor = target_height / h
    w_factor = target_width / w
    
    factor = min(h_factor, w_factor)
    
    image = imresize(image, [int(h*factor), int(w*factor)], preserve_range=True).astype(np.uint8)
    h, w = image.shape[:-1]
    
    diff_v = target_height - h
    diff_v_first_part = diff_v // 2
    diff_v_last_part = diff_v - diff_v_first_part
    
    diff_h = target_width - w
    diff_h_first_part = diff_h // 2
    diff_h_last_part = diff_h - diff_h_first_part
    
    # make image squared
    
    image = np.pad(image, [(diff_v_first_part, diff_v_last_part),
                           (diff_h_first_part, diff_h_last_part),
                           (0, 0)], mode='constant')
    if add_noise:
        mask = np.zeros([image.shape[0], image.shape[1], 1])
        if diff_v_first_part > 0:
            mask[:diff_v_first_part, :] = 1
        if diff_h_first_part > 0:
            mask[:, :diff_h_first_part] = 1
        if diff_v_last_part > 0:
            mask[-diff_v_last_part+1:, :] = 1
        if diff_h_last_part > 0:
            mask[:, -diff_h_last_part+1:] = 1
    
        image = image + (mask * np.random.uniform(0, 255, image.shape)).astype(np.uint8)
    
    if ret_image_shifts:
        return image, (diff_v_first_part, diff_h_first_part, factor)
    else:
        return image


def convert_sample_to_YOLO(image, new_image_shape, grid_size, bboxes, classes=None, add_noise=False):
    """"""
    new_image_shape = list(new_image_shape)
    assert len(new_image_shape) == 2, ''
    assert isinstance(new_image_shape[0], int) and isinstance(new_image_shape[1], int), ''
    
    grid_size = list(grid_size)
    assert len(grid_size) == 2, ''
    assert isinstance(grid_size[0], int) and isinstance(grid_size[1], int), ''

    result = smart_resize(image, new_image_shape[0], new_image_shape[1], add_noise=add_noise, ret_image_shifts=True)
    image, resize_params = result
    
    result = convert_sample_to_YOLO_preresized(resize_params, grid_size, bboxes, classes, add_noise)
    return tuple([image] + list(result))


def convert_sample_to_YOLO_preresized(image, resize_params, grid_size, bboxes, classes=None, add_noise=False):
    diff_v_first_part, diff_h_first_part, factor = resize_params
    yolo_conf_probs = np.zeros(grid_size+[1], dtype=np.float32)
    yolo_bboxes = np.zeros(grid_size+[4], dtype=np.float32)
    if classes is not None:
        yolo_classes = np.zeros(grid_size+[1], dtype=np.int32)
    
    grid_v_factor = grid_size[0] / image.shape[0]
    grid_h_factor = grid_size[1] / image.shape[1]
    grid_h = image.shape[0] / grid_size[0]
    grid_w = image.shape[1] / grid_size[1]

    if classes is not None:
        zipped = zip(bboxes, classes)
    else:
        zipped = bboxes

    for item in zipped:
        if classes is not None:
            bbox, c = item
        else:
            bbox = item

        x, y, w, h = bbox

        x = x*factor + diff_h_first_part
        y = y*factor + diff_v_first_part
        w *= factor
        h *= factor
        
        x += w/2
        y += h/2
        
        x *= grid_h_factor
        y *= grid_v_factor
        
        i = int(y)
        j = int(x)
        
        if w == 0 or h == 0:
            continue
        
        yolo_conf_probs[i, j, 0] = 1
        yolo_bboxes[i, j, 0] = x - j
        yolo_bboxes[i, j, 1] = y - i
        yolo_bboxes[i, j, 2] = w / grid_w
        yolo_bboxes[i, j, 3] = h / grid_h
        if classes is not None:
            yolo_classes[i, j, 0] = c
    
    if classes is not None:
        return yolo_conf_probs, yolo_bboxes, yolo_classes
    else:
        return yolo_conf_probs, yolo_bboxes


def convert_YOLO_result_to_normal(image_shape, conf_probs, bboxes, classes=None, conf_thresh=0.5):
    grid_size = conf_probs.shape
    
    conf_probs_mask = np.float32(conf_probs >= conf_thresh)
    _indices = conf_probs_mask.nonzero()
    _conf_probs = conf_probs[_indices]
    if classes is not None:
        _classes = classes[_indices]
    _bboxes = bboxes.reshape(grid_size[0], grid_size[1], 1, 4)[_indices]

    img_bboxes = []
    v_factor = image_shape[0] / grid_size[0]
    h_factor = image_shape[1] / grid_size[1]
    
    grid_h = image_shape[0] / grid_size[0]
    grid_w = image_shape[1] / grid_size[1]

    for k in range(len(_bboxes)):
        conf = _conf_probs[k]
        bbox = _bboxes[k]
        
        i = _indices[0][k]
        j = _indices[1][k]

        x = (j + bbox[0]) * h_factor
        y = (i + bbox[1]) * v_factor
        
        w = bbox[2] * grid_w
        h = bbox[3] * grid_h

        if classes is not None:
            img_bboxes.append((conf, (x-w/2, y-h/2, w, h), _classes[k]))
        else:
            img_bboxes.append((conf, (x-w/2, y-h/2, w, h)))
        
    return img_bboxes


def get_iou(bbox1, bbox2):
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2
    
    xy1 = np.array([x1, y1], np.float32)
    wh1 = np.array([w1, h1], np.float32)

    xy2 = np.array([x2, y2], np.float32)
    wh2 = np.array([w2, h2], np.float32)

    mins1  = xy1
    maxes1 = xy1 + wh1

    mins2  = xy2
    maxes2 = xy2 + wh2

    intersect_mins = np.maximum(mins1, mins2)
    intersect_maxes = np.minimum(maxes1, maxes2)
    intersect_wh = np.maximum(0, intersect_maxes - intersect_mins)

    intersect_areas = np.prod(intersect_wh)

    areas1 = np.prod(wh1)
    areas2 = np.prod(wh2)

    union_areas = areas1 + areas2 - intersect_areas

    iou = intersect_areas / np.maximum(union_areas, 1e-6)
    
    return iou


def bbox_in_bbox(bbox1, bbox2, thresh=1):
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2

    xy1 = np.array([x1, y1], np.float32)
    wh1 = np.array([w1, h1], np.float32)

    xy2 = np.array([x2, y2], np.float32)
    wh2 = np.array([w2, h2], np.float32)

    mins1  = xy1
    maxes1 = xy1 + wh1

    mins2  = xy2
    maxes2 = xy2 + wh2

    intersect_mins = np.maximum(mins1, mins2)
    intersect_maxes = np.minimum(maxes1, maxes2)
    intersect_wh = np.maximum(0, intersect_maxes - intersect_mins)

    intersect_areas = np.prod(intersect_wh)

    areas1 = np.prod(wh1)
    areas2 = np.prod(wh2)

    score = intersect_areas / np.maximum(np.minimum(areas1, areas2), 1e-6)
    
    return score >= max(thresh, 1)
    
    
def non_max_suppression(confidences, bboxes, overlapThresh=0.6):
    # if there are no boxes, return an empty list
    if len(bboxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if bboxes.dtype.kind == "i":
        bboxes = np.astype(bboxes, np.float32)
 
    # initialize the list of picked indexes    
    pick = []
    
    idxs = [(c, i) for i, c in enumerate(confidences)]
    idxs = sorted(idxs, key=lambda x: x[0])
    _, idxs = zip(*idxs)
    idxs = list(idxs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        del idxs[last]
        
        pick.append(i)
 
        mark_deleted = []
        for k, j in enumerate(idxs):
            overlap = get_iou(bboxes[i], bboxes[j])
            if overlap >= overlapThresh or bbox_in_bbox(bboxes[i], bboxes[j], 1-(1-overlapThresh)**2):
                mark_deleted.append(k)
        
        for k in reversed(mark_deleted):
            del idxs[k]
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick
