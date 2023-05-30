from .eval_aug import Transform_single
from .byol_aug import BYOL_transform

def get_aug(name='byol', image_size=224, train=True, train_classifier=None):

    if train==True:
        if name == 'byol':
            augmentation = BYOL_transform(image_size)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








