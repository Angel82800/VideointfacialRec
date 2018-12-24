
REGISTERED_CLASSES = dict()

def reset_classes():
    REGISTERED_CLASSES.clear()

    
def get_amount_of_classes():
    return len(REGISTERED_CLASSES)


def get_class_id(name):
    if name not in REGISTERED_CLASSES:
        REGISTERED_CLASSES[name] = len(REGISTERED_CLASSES)
    return REGISTERED_CLASSES[name]


def get_class_name(class_id):
    for key, value in REGISTERED_CLASSES.items():
        if class_id == value:
            return key
    else:
        return 'Object'
