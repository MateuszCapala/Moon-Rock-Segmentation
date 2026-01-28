import os
import yaml
import torch
import argparse
import wandb
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import MoonDataset, get_training_augmentation, get_validation_augmentation
import datetime

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train(config_path):
    config = load_config(config_path)
    
    wandb.init(project=config["project_name"], config=config)
    
    if config["training"].get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config["training"]["device"]
    
    print(f"Using device: {device}")

    train_dataset = MoonDataset(
        manifest_path=config["data"]["manifest_path"],
        split='train',
        transform=get_training_augmentation()
    )
    
    val_dataset = MoonDataset(
        manifest_path=config["data"]["manifest_path"],
        split='val',
        transform=get_validation_augmentation()
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=config["training"]["num_workers"]
    )

    model = smp.Linknet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"]["weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["num_classes"], 
    )

    model = model.to(device)

    criterion_dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])

    best_iou = 0.0
    epochs = config["training"]["epochs"]
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion_dice(outputs, masks) + criterion_ce(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        model.eval()
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                tp, fp, fn, tn = smp.metrics.get_stats(outputs.argmax(1), masks, mode='multiclass', num_classes=config["model"]["num_classes"], ignore_index=255)
                val_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        avg_iou = val_iou / len(val_loader)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_iou": avg_iou
        })

        print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, IoU={avg_iou:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model_path = f"best_moon_model_{timestamp}.pth"

    if best_iou > 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f"best_moon_model_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"⭐ Ostateczny najlepszy model zapisany: {model_path}")
        wandb.save(model_path)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)