import torch
from torchvision.datasets.coco import CocoDetection
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from PIL import Image

class CocoCarDamageDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None, duplicate=False):
        super().__init__(img_folder, ann_file)
        self.transforms = transforms
        self.duplicate = duplicate
        self.ids = self.ids * 2 if duplicate else self.ids

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image, anns = super().__getitem__(idx)
        image = np.array(image)

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': boxes,
            'labels': labels
        }

        # Albumentations expects boxes in Pascal VOC format [x_min, y_min, x_max, y_max]
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=target['boxes'], class_labels=target['labels'])
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(transformed['class_labels'], dtype=torch.int64)

        return image, target
