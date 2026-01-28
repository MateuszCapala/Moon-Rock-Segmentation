import os
import pandas as pd
import glob
from pathlib import Path 
from sklearn.model_selection import train_test_split

def load_anomaly_ids(data_root):
    anomaly_files = [
        "mismatch_IDs.txt", "cam_anomaly_IDs.txt", 
        "shadow_IDs.txt", "ground_facing_IDs.txt", "top200_largerocks_IDs.txt"
    ]
    
    bad_ids = set()
    for filename in anomaly_files:
        path = Path(data_root) / filename
        if path.exists():
            with open(path, 'r') as f:
                for line in f:
                    digits = "".join(filter(str.isdigit, line.strip()))
                    if digits:
                        bad_ids.add(digits.zfill(4))
            
    return bad_ids

def create_manifest(data_root, output_csv="data/manifest.csv", test_size=0.15):
    
    root_path = Path(data_root)
    bad_ids = load_anomaly_ids(root_path)

    render_dir = root_path / "images" / "render"
    clean_dir = root_path / "images" / "clean"
    
    all_render_files = sorted(list(render_dir.glob("render*.png")))
    
    valid_data = []
    for render_path in all_render_files:
        file_id = "".join(filter(str.isdigit, render_path.name))
        
        if file_id in bad_ids:
            continue
            
        mask_path = clean_dir / f"clean{file_id}.png"
        
        if mask_path.exists():
            valid_data.append({
                "file_id": file_id,
                "image_path": render_path.as_posix(),
                "mask_path": mask_path.as_posix()
            })

    df = pd.DataFrame(valid_data)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    final_df = pd.concat([train_df, val_df])
    
    Path("data").mkdir(exist_ok=True)
    final_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    create_manifest(data_root="data/archive")