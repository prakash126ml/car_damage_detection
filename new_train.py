# NOTE: This environment does not support PyTorch. Please run this code in your local environment with PyTorch installed.
# You can install PyTorch via pip: pip install torch torchvision

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F

# Dataset
class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = torch.as_tensor([ann['area'] for ann in anns], dtype=torch.float32)
        iscrowd = torch.as_tensor([ann['iscrowd'] for ann in anns], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# Transforms
def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
    ])

def get_val_transform():
    return T.Compose([T.ToTensor()])

# Model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Evaluation
from torchvision.models.detection.coco_eval import CocoEvaluator
from torchvision.models.detection.coco_utils import get_coco_api_from_dataset

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    val_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        val_loss += sum(loss for loss in loss_dict.values()).item()
    return val_loss / len(data_loader)

@torch.no_grad()
def evaluate_coco_map(model, data_loader, device):
    model.eval()
    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        res = {t["image_id"].item(): o for t, o in zip(targets, outputs)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator.coco_eval['bbox'].stats[0]  # mAP @ IoU=0.50:0.95

# Train
def train_model(model, train_loader, val_loader, device, num_epochs=20, patience=5, top_n=3):
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

    best_models = []
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        val_map = evaluate_coco_map(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, mAP={val_map:.4f}")

        # Save best models
        model_path = f"model_epoch{epoch+1}_mAP{val_map:.4f}.pth"
        torch.save(model.state_dict(), model_path)
        best_models.append((val_map, model_path))
        best_models = sorted(best_models, key=lambda x: x[0], reverse=True)[:top_n]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Top-N saved models:")
    for score, path in best_models:
        print(f"{path} - mAP: {score:.4f}")

# Visualization
@torch.no_grad()
def visualize_prediction(model, dataset, device, idx=0, score_threshold=0.5):
    model.eval()
    img, _ = dataset[idx]
    img_tensor = img.to(device).unsqueeze(0)
    preds = model(img_tensor)[0]

    img = F.to_pil_image(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()

    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
        if score >= score_threshold:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{label.item()} {score:.2f}", color='white',
                    bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

# Usage Example
if __name__ == "__main__":
    train_dataset = CocoDataset("dataset/train/images", "dataset/train/annotations/instances_train.json", transforms=get_train_transform())
    val_dataset = CocoDataset("dataset/val/images", "dataset/val/annotations/instances_val.json", transforms=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=9)

    train_model(model, train_loader, val_loader, device, num_epochs=20, patience=5, top_n=3)

    # visualize_prediction(model, val_dataset, device, idx=0)
