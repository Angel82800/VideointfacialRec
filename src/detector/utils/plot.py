import cv2
import io
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(classes_confusion_matrix, classes_str_to_id):   
    cm = classes_confusion_matrix
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm = np.nan_to_num(cm, copy=True)

    np.set_printoptions(precision=2)

    fig = plt.figure(figsize=(classes_confusion_matrix.shape[0]+1, classes_confusion_matrix.shape[1]+1),
                     dpi=120, facecolor='w', edgecolor='k')
    fig.clear()
    
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Blues')

    classes = sorted(classes_str_to_id.items(), key=lambda x: x[1])
    classes, _ = zip(*classes)

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=10,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], '0.02f'),
                horizontalalignment="center",
                verticalalignment='center',
                fontsize=14,
                color=('black' if cm[i, j] < 0.5 else 'white'))
        
    fig.set_tight_layout(True)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent='False')
    buf.seek(0)
    
    fig.clear()
    plt.close(fig)
    
    x = np.frombuffer(buf.getvalue(), dtype='uint8')

    return cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
