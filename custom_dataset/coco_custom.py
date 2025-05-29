import torch
from torchvision.datasets.coco import CocoDetection
import numpy as np
from PIL import Image


class AlbumentationsCompose:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, image, anns):
        boxes = []
        labels = []

        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        if isinstance(image, Image.Image):
            image = np.array(image)

        # Apply Albumentations transform
        transformed = self.aug(
            image=image,
            bboxes=boxes,
            class_labels=labels
        )

        image = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['class_labels']

        # Build target in PyTorch format
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        return image, target


class CocoCarDamageDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None, duplicate=False):
        super().__init__(img_folder, ann_file)
        self.transforms = transforms
        self.duplicate = duplicate
        self.ids = self.ids * 2 if duplicate else self.ids

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image, anns = super().__getitem__(idx)

        if self.transforms:
            image, target = self.transforms(image, anns)
        else:
            # fallback for no transforms
            boxes = []
            labels = []
            for ann in anns:
                xmin, ymin, w, h = ann['bbox']
                xmax = xmin + w
                ymax = ymin + h
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann['category_id'])

            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }

        target['image_id'] = torch.tensor([img_id])

        return image, target
