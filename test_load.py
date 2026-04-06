import os
import cv2
import matplotlib.pyplot as plt

image_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_images"
mask_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_masks"

files = sorted(os.listdir(image_dir))
sample_file = files[0]

image_path = os.path.join(image_dir, sample_file)
mask_path = os.path.join(mask_dir, sample_file)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print("Sample file:", sample_file)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Image min/max:", image.min(), image.max())
print("Mask min/max:", mask.min(), mask.max())

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("X-ray Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.tight_layout()
plt.show()