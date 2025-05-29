import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from datasets.coco_custom import CocoCarDamageDataset, get_transforms
from models.faster_rcnn import get_model
from utils import collate_fn

def train_model(model, train_loader, val_loader, device, num_epochs=20, patience=5):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

    best_loss = float('inf')
    early_stop_count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping.")
                break

        scheduler.step(avg_val_loss)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CocoCarDamageDataset(
        img_folder="data/train",
        ann_file="annotations/train.json",
        transforms=get_transforms(train=True),
        duplicate=True
    )

    val_dataset = CocoCarDamageDataset(
        img_folder="data/val",
        ann_file="annotations/val.json",
        transforms=get_transforms(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 7 classes + background = 8
    model = get_model(num_classes=8)

    train_model(model, train_loader, val_loader, device, num_epochs=20, patience=5)
