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

def run_inference(model_path, manifest_path):
    device = "cuda" # if torch.cuda.is_available() else "cpu"
    
    model = smp.Linknet(encoder_name="resnet34", classes=4)
    if not Path(model_path).exists(): return
    model.load_state_dict(torch.load(model_path, map_location=device))
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
        vis_gt = mask_to_rgb(gt_ind)
        vis_pred = mask_to_rgb(pred.astype(np.uint8))

        results.append(np.hstack([vis_img, vis_gt, vis_pred]))

    if results:
        cv2.imwrite("validation_results.png", np.vstack(results))
        print("Plik validation_results.png został zapisany.")

if __name__ == "__main__":
    run_inference("best_moon_model.pth", "data/manifest.csv")