import os
import glob
import shutil
import random
import yaml
from ultralytics import YOLO

# Configuration
DATASET_ROOT = os.path.abspath("dataset_yolo")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")

# YOLOv8 expects this structure for auto-discovery usually, or we defines paths explicitly
# We will reorganize into:
# dataset_yolo/
#   train/
#     images/
#     labels/
#   val/
#     images/
#     labels/

def setup_dataset():
    print(f"[INFO] Organizing dataset at {DATASET_ROOT}...")
    
    # Check if already organized
    if os.path.exists(os.path.join(DATASET_ROOT, "train")):
        print("[INFO] Dataset appears to be already organized. Skipping organization.")
        return

    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_ROOT, split, 'labels'), exist_ok=True)

    # Get all image files
    # Supports png, jpg, etc.
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    image_files.sort()
    
    # Shuffle
    random.seed(42)
    random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * 0.8) # 80% train
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"[INFO] Total images: {len(image_files)}")
    print(f"[INFO] Train: {len(train_files)}, Val: {len(val_files)}")
    
    def move_files(files, split):
        for img_path in files:
            basename = os.path.basename(img_path) # e.g. 000123_front.png
            label_name = basename.replace(".png", ".txt")
            label_path = os.path.join(LABELS_DIR, label_name)
            
            # Target paths
            shutil.copy2(img_path, os.path.join(DATASET_ROOT, split, 'images', basename))
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(DATASET_ROOT, split, 'labels', label_name))
            else:
                print(f"[WARN] Label file missing for {basename}")

    move_files(train_files, 'train')
    move_files(val_files, 'val')
    print("[INFO] Dataset organization complete!")

def create_yaml():
    yaml_path = os.path.join(DATASET_ROOT, "gearbox_data.yaml")
    
    data_config = {
        'path': DATASET_ROOT,
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'sun_planetary_gear',
            1: 'ring_gear',
            2: 'planetary_reducer',  # Keep for backward compatibility with existing labels
            3: 'planetary_carrier'   # Added - critical for pin location detection
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"[INFO] Created config file at {yaml_path}")
    return yaml_path

def train_model(yaml_path):
    print("[INFO] Starting YOLOv8 Training...")
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (nano version for speed)

    # Train the model
    # imgsz=640 is standard
    # epochs=50 should be enough for this simple task
    results = model.train(data=yaml_path, epochs=50, imgsz=640, project="gearbox_training", name="yolov8n_run")
    
    print(f"[INFO] Training Finished. Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    # Ensure ultralytics is installed
    try:
        import ultralytics
    except ImportError:
        print("[ERROR] 'ultralytics' library is not installed. Please run: pip install ultralytics")
        exit(1)

    setup_dataset()
    yaml_path = create_yaml()
    train_model(yaml_path)
