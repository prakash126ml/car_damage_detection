import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.coco_custom import AlbumentationsCompose

def get_transforms(train=True):
    if train:
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.Blur(p=0.2),
            A.Resize(512, 512),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        aug = A.Compose([
            A.Resize(512, 512),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    return AlbumentationsCompose(aug)
