from pathlib import Path
import shutil

DATASET1 = Path("Construction-Site-Safety-23")
DATASET2 = Path("PPE-vest-hardhat-1")
OUTPUT   = Path("merged_dataset")

# Dataset 1 class IDs stay as-is (already correct)
# 0:Hardhat 1:Mask 2:NO-Hardhat 3:NO-Mask 4:NO-Safety-Vest
# 5:Person 6:Safety-Cone 7:Safety-Vest 8:vehicle 9:machinery (if exists)

# Dataset 2 remap — visually verified:
# 0=Hardhat → 0, 1=NO-Hardhat → 2, 2=NO-Safety-Vest → 4, 3=Safety-Vest → 7
DATASET2_REMAP = {
    0: 0,   # Hardhat       → Hardhat
    1: 2,   # NO-Hardhat    → NO-Hardhat
    2: 4,   # NO-Safety-Vest → NO-Safety-Vest
    3: 7,   # Safety-Vest   → Safety-Vest
}

UNIFIED_CLASSES = [
    'Hardhat',        # 0
    'Mask',           # 1
    'NO-Hardhat',     # 2
    'NO-Mask',        # 3
    'NO-Safety-Vest', # 4
    'Person',         # 5
    'Safety-Cone',    # 6
    'Safety-Vest',    # 7
    'vehicle',        # 8
]

# create output folders
for split in ['train', 'valid', 'test']:
    (OUTPUT / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / 'labels').mkdir(parents=True, exist_ok=True)

counter = 0

def copy_files(img_dir, lbl_dir, out_img_dir, out_lbl_dir, remap=None):
    global counter
    if not img_dir.exists():
        print(f"  Skipping {img_dir} — not found")
        return 0
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    for img_path in images:
        new_name = f"{counter:06d}{img_path.suffix}"
        
        # copy image
        shutil.copy2(img_path, out_img_dir / new_name)
        
        # copy + remap label
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                if remap:
                    if class_id not in remap:
                        continue   # skip unknown classes
                    class_id = remap[class_id]
                new_lines.append(f"{class_id} {' '.join(parts[1:])}\n")
            
            with open(out_lbl_dir / (Path(new_name).stem + ".txt"), "w") as f:
                f.writelines(new_lines)
        
        counter += 1
    return len(images)

# --- merge dataset 1 ---
print("Processing Dataset 1: Construction-Site-Safety-23")
for split in ['train', 'valid', 'test']:
    n = copy_files(
        DATASET1 / split / 'images',
        DATASET1 / split / 'labels',
        OUTPUT / split / 'images',
        OUTPUT / split / 'labels',
        remap=None   # class IDs already correct
    )
    print(f"  {split}: {n} images")

# --- merge dataset 2 (train only) ---
print("\nProcessing Dataset 2: PPE-vest-hardhat-1")
n = copy_files(
    DATASET2 / 'train' / 'images',
    DATASET2 / 'train' / 'labels',
    OUTPUT / 'train' / 'images',
    OUTPUT / 'train' / 'labels',
    remap=DATASET2_REMAP
)
print(f"  train: {n} images")

# --- write data.yaml ---
yaml_path = OUTPUT / 'data.yaml'
with open(yaml_path, 'w') as f:
    f.write(f"train: {(OUTPUT / 'train' / 'images').resolve()}\n")
    f.write(f"val: {(OUTPUT / 'valid' / 'images').resolve()}\n")
    f.write(f"test: {(OUTPUT / 'test' / 'images').resolve()}\n")
    f.write(f"nc: {len(UNIFIED_CLASSES)}\n")
    f.write("names:\n")
    for name in UNIFIED_CLASSES:
        f.write(f"  - '{name}'\n")

# --- final count ---
print(f"\nMerge complete! Total images: {counter}")
print(f"Saved to: {OUTPUT.resolve()}")
for split in ['train', 'valid', 'test']:
    n = len(list((OUTPUT / split / 'images').glob("*")))
    print(f"  {split}: {n} images")
print(f"Classes ({len(UNIFIED_CLASSES)}): {UNIFIED_CLASSES}")