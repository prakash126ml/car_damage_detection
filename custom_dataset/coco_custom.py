import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection
from PIL import Image

class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        # Convert PIL to NumPy
        image = np.array(image)

        boxes = []
        labels = []

        for obj in target:
            x_min, y_min, w, h = obj['bbox']
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj['category_id'])

        transformed = self.transform(
            image=image,
            bboxes=boxes,
            class_labels=labels
        )

        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        new_target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, new_target



class CocoCarDamageDataset(CocoDetection):
    def __init__(self, image_dir, ann_file, transform=None):
        super().__init__(image_dir, ann_file)
        self.transform = transform

    def __getitem__(self, idx):
        image, anns = super().__getitem__(idx)

        # Convert COCO annotations into expected format
        if self.transform:
            image, target = self.transform(image, anns)
        else:
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

            image = ToTensorV2()(image=np.array(image))["image"]
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }

        # Add image_id
        image_id = self.ids[idx]
        target["image_id"] = torch.tensor([image_id])

        return image, target
