import os
import shutil

image_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\images"
mask_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\masks"

paired_image_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_images"
paired_mask_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_masks"

os.makedirs(paired_image_dir, exist_ok=True)
os.makedirs(paired_mask_dir, exist_ok=True)

image_files = set(os.listdir(image_dir))
mask_files = set(os.listdir(mask_dir))

common_files = sorted(image_files.intersection(mask_files))

print(f"Total images: {len(image_files)}")
print(f"Total masks: {len(mask_files)}")
print(f"Matching pairs: {len(common_files)}")

for fname in common_files:
    shutil.copy2(os.path.join(image_dir, fname), os.path.join(paired_image_dir, fname))
    shutil.copy2(os.path.join(mask_dir, fname), os.path.join(paired_mask_dir, fname))

print("Done copying matched pairs.")
print(f"Paired images folder: {paired_image_dir}")
print(f"Paired masks folder: {paired_mask_dir}")