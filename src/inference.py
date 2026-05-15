import torch
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

def preprocess_mask_rgb(mask_bgr):
    h, w = mask_bgr.shape[:2]
    new_mask = np.zeros((h, w), dtype=np.uint8)
    
    new_mask[mask_bgr[:,:,0] > 128] = 1 
    new_mask[mask_bgr[:,:,2] > 128] = 2 
    new_mask[mask_bgr[:,:,1] > 128] = 3 
    
    return new_mask

def mask_to_rgb(mask):
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[mask == 1] = [255, 0, 0]   
    img[mask == 2] = [0, 0, 255]  
    img[mask == 3] = [0, 255, 0]  
    return img


def blend_mask_on_image(image_bgr, mask_bgr, class_mask, alpha=0.55):
    overlay = image_bgr.copy()
    rock_or_sky = class_mask > 0
    if np.any(rock_or_sky):
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, mask_bgr, alpha, 0.0)
        overlay[rock_or_sky] = blended[rock_or_sky]
    return overlay


def add_panel_label(panel_bgr, label):
    labeled = panel_bgr.copy()
    bar_h = 30
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], bar_h), (20, 20, 20), -1)
    cv2.putText(
        labeled,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def build_result_row(image_bgr, gt_ind, pred_ind):
    gt_mask_bgr = mask_to_rgb(gt_ind)
    pred_mask_bgr = mask_to_rgb(pred_ind.astype(np.uint8))

    gt_overlay = blend_mask_on_image(image_bgr, gt_mask_bgr, gt_ind)
    pred_overlay = blend_mask_on_image(image_bgr, pred_mask_bgr, pred_ind)

    panels = [
        add_panel_label(image_bgr, "Input image"),
        add_panel_label(gt_mask_bgr, "Ground truth mask"),
        add_panel_label(pred_mask_bgr, "Prediction mask"),
        add_panel_label(gt_overlay, "Ground truth overlay"),
        add_panel_label(pred_overlay, "Prediction overlay"),
    ]
    return np.hstack(panels)


def _infer_encoder_from_state_dict(state_dict):
    # ResNet50+ bottleneck blocks contain conv3; ResNet34 uses only basic blocks.
    has_bottleneck = any(k.startswith("encoder.layer1.0.conv3") for k in state_dict)
    if has_bottleneck:
        return "resnet50"

    has_layer1_block2 = any(k.startswith("encoder.layer1.2") for k in state_dict)
    if has_layer1_block2:
        return "resnet34"

    return "resnet18"


def _build_model(encoder_name):
    return smp.Linknet(encoder_name=encoder_name, classes=4)


def _load_model_with_auto_encoder(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    inferred_encoder = _infer_encoder_from_state_dict(state_dict)

    encoder_candidates = [inferred_encoder, "resnet34", "resnet50", "resnet18"]
    # Keep order while removing duplicates.
    encoder_candidates = list(dict.fromkeys(encoder_candidates))

    last_error = None
    for encoder_name in encoder_candidates:
        try:
            model = _build_model(encoder_name)
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint with encoder: {encoder_name}")
            return model
        except RuntimeError as err:
            last_error = err

    raise RuntimeError(
        "Nie udalo sie zaladowac checkpointu dla zadnego wspieranego enkodera "
        f"{encoder_candidates}. Ostatni blad: {last_error}"
    )

def run_inference(model_path, manifest_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        return

    model = _load_model_with_auto_encoder(model_path, device)
    model.to(device).eval()

    df = pd.read_csv(manifest_path)
    val_samples = df[df['split'] == 'val'].sample(5)
    
    transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    
    results = []
    for _, row in val_samples.iterrows():
        img = cv2.imread(row['image_path'])
        gt_mask = cv2.imread(row['mask_path'])
        if img is None or gt_mask is None: continue

        gt_ind = preprocess_mask_rgb(gt_mask)
        gt_ind = cv2.resize(gt_ind, (256, 256), interpolation=cv2.INTER_NEAREST)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        vis_img = cv2.resize(img, (256, 256))
        results.append(build_result_row(vis_img, gt_ind, pred))

    if results:
        cv2.imwrite("validation_results.png", np.vstack(results))
        print("Plik validation_results.png został zapisany.")

if __name__ == "__main__":
    run_inference("best_moon_model.pth", "data/manifest.csv")