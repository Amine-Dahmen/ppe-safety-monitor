import os
import cv2
import random
import matplotlib.pyplot as plt

# Update this path to wherever the dataset downloaded
DATASET_PATH = "PPE-vest-hardhat-1"
IMAGE_DIR = os.path.join(DATASET_PATH, "train", "images")
LABEL_DIR = os.path.join(DATASET_PATH, "train", "labels")

# Count images
images = os.listdir(IMAGE_DIR)
labels = os.listdir(LABEL_DIR)
print(f"Training images: {len(images)}")
print(f"Label files: {len(labels)}")

# Count class distribution
class_names = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
               "NO-Safety Vest", "Person", "Safety Cone",
               "Safety Vest", "machinery", "vehicle"]
class_counts = {name: 0 for name in class_names}

for label_file in labels:
    with open(os.path.join(LABEL_DIR, label_file)) as f:
        for line in f:
            class_id = int(line.split()[0])
            if class_id < len(class_names):
                class_counts[class_names[class_id]] += 1

print("\nClass distribution:")
for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count}")

# Show 6 random labeled images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sample_images = random.sample(images, 6)

for ax, img_file in zip(axes.flatten(), sample_images):
    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(img_file[:30])
    ax.axis("off")

plt.tight_layout()
plt.savefig("dataset_samples.png")
print("\nSaved sample grid to dataset_samples.png")
plt.show()