import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

def preprocess_mask_rgb(mask_bgr):
    h, w = mask_bgr.shape[:2]
    new_mask = np.zeros((h, w), dtype=np.uint8)

    new_mask[mask_bgr[:, :, 0] > 128] = 1  # Large Rocks (Blue)
    new_mask[mask_bgr[:, :, 2] > 128] = 2  # Sky (Red)
    new_mask[mask_bgr[:, :, 1] > 128] = 3  # Small Rocks (Green)

    return new_mask

def mask_to_rgb(mask):
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[mask == 1] = [255, 0, 0]  # Large Rocks (Blue)
    img[mask == 2] = [0, 0, 255]  # Sky (Red)
    img[mask == 3] = [0, 255, 0]  # Small Rocks (Green)
    return img

def run_inference_real(model_path, real_images_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Linknet(encoder_name="resnet50", classes=4)
    if not Path(model_path).exists():
        print("Model file not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    real_images_dir = Path(real_images_path)
    if not real_images_dir.exists():
        print("Real images directory not found.")
        return

    results = []
    for image_file in real_images_dir.glob("PCAM*.png"):
        mask_file = real_images_dir / f"g_{image_file.name}"

        if not mask_file.exists():
            print(f"Mask file not found for {image_file.name}")
            continue

        img = cv2.imread(str(image_file))
        gt_mask = cv2.imread(str(mask_file))

        if img is None or gt_mask is None:
            print(f"Error reading {image_file.name} or its mask.")
            continue

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
        cv2.imwrite("real_moon_results.png", np.vstack(results))
        print("Plik real_moon_results.png został zapisany.")

if __name__ == "__main__":
    run_inference_real("best_moon_model.pth", "data/archive/real_moon_images/")