from pathlib import Path

# Check Dataset 2's actual yaml
import yaml
with open("PPE-vest-hardhat-1/data.yaml") as f:
    data = yaml.safe_load(f)
print("Dataset 2 classes:", data['names'])

# Also peek at a few label files to confirm
label_dir = Path("PPE-vest-hardhat-1/train/labels")
label_files = list(label_dir.glob("*.txt"))[:5]

class_ids_found = set()
for lf in label_files:
    with open(lf) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids_found.add(int(parts[0]))

print("Class IDs actually used in labels:", sorted(class_ids_found))