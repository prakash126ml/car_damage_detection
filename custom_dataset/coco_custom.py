import torch
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms as T

class CocoCarDamageDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None, duplicate=False):
        super().__init__(img_folder, ann_file)
        self.transforms = transforms
        self.duplicate = duplicate
        self.ids = self.ids * 2 if duplicate else self.ids

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img, anns = super().__getitem__(idx)
        boxes = []
        labels = []

        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

def get_transforms(train=True):
    t = [T.ToTensor()]
    if train:
        t.extend([T.RandomHorizontalFlip(0.5), T.ColorJitter(brightness=0.2, contrast=0.2)])
    return T.Compose(t)
