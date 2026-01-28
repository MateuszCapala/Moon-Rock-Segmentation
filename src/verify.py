import torch
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import MoonDataset, get_validation_augmentation

def verify():
    device = "cuda" # if torch.cuda.is_available() else "cpu"
    model = smp.Linknet(encoder_name="resnet50", classes=4).to(device)
    
    try:
        model.load_state_dict(torch.load("best_moon_model.pth"))
    except Exception as e:
        print(f"Nie udało się wczytać wag: {e}")

    model.eval()

    ds = MoonDataset("data/manifest.csv", split='val', transform=get_validation_augmentation())
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    num_classes = 4
    total_tp = torch.zeros(num_classes)
    total_fp = torch.zeros(num_classes)
    total_fn = torch.zeros(num_classes)
    total_tn = torch.zeros(num_classes)

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            
            tp, fp, fn, tn = smp.metrics.get_stats(
                output.argmax(1), 
                masks, 
                mode='multiclass', 
                num_classes=num_classes,
                ignore_index=None 
            )
            
            total_tp += tp.sum(dim=0).cpu()
            total_fp += fp.sum(dim=0).cpu()
            total_fn += fn.sum(dim=0).cpu()
            total_tn += tn.sum(dim=0).cpu()

    iou_per_class = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="none")
    dice_per_class = smp.metrics.f1_score(total_tp, total_fp, total_fn, total_tn, reduction="none")
    precision_per_class = smp.metrics.precision(total_tp, total_fp, total_fn, total_tn, reduction="none")
    recall_per_class = smp.metrics.recall(total_tp, total_fp, total_fn, total_tn, reduction="none")
    accuracy_per_class = smp.metrics.accuracy(total_tp, total_fp, total_fn, total_tn, reduction="none")

    classes = ["Tło (Background)", "Duże skały (Large)", "Niebo (Sky)", "Małe skały (Small)"]
    print("\nMetryki dla każdej klasy:")
    for i, name in enumerate(classes):
        print(f"{name:20} | IoU: {iou_per_class[i].item():.2%} | Dice: {dice_per_class[i].item():.2%} | Precision: {precision_per_class[i].item():.2%} | Recall: {recall_per_class[i].item():.2%} | Accuracy: {accuracy_per_class[i].item():.2%}")

    print("\nŚrednie metryki dla całego modelu:")
    print(f"mIoU: {iou_per_class.mean().item():.2%}")
    print(f"mDice: {dice_per_class.mean().item():.2%}")
    print(f"mPrecision: {precision_per_class.mean().item():.2%}")
    print(f"mRecall: {recall_per_class.mean().item():.2%}")
    print(f"mAccuracy: {accuracy_per_class.mean().item():.2%}")


if __name__ == "__main__":
    verify()