from pathlib import Path

LABEL_DIR = Path("merged_dataset/train/labels")

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
               'NO-Safety-Vest', 'Person', 'Safety-Cone', 'Safety-Vest', 'vehicle']
counts = {i: 0 for i in range(len(class_names))}

for lbl in LABEL_DIR.glob("*.txt"):
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1

print("Class distribution in merged training set:\n")
for i, name in enumerate(class_names):
    print(f"  {name:<20} {counts[i]:>5}")