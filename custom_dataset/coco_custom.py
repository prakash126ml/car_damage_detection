import torch
from torchvision.datasets.coco import CocoDetection
import numpy as np
from albumentations.pytorch import ToTensorV2
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
        image = np.array(image)  # Convert PIL to NumPy array for Albumentations

        boxes = []
        labels = []

        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        # Albumentations expects list of bboxes and labels
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        # Convert to tensors for PyTorch model
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        return image, target
