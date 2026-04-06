import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model_unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_images"
mask_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_masks"

model = UNet().to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model.eval()

files = sorted(os.listdir(image_dir))
sample_file = files[0]   # you can change this to files[10], files[25], etc.

image_path = os.path.join(image_dir, sample_file)
mask_path = os.path.join(mask_dir, sample_file)

# Load original grayscale image and mask
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Keep copies for display
display_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
display_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

# Prepare image for model
input_image = display_image.astype(np.float32) / 255.0
input_image = np.expand_dims(input_image, axis=0)   # channel
input_image = np.expand_dims(input_image, axis=0)   # batch
input_tensor = torch.tensor(input_image, dtype=torch.float32).to(device)

with torch.no_grad():
    prediction = model(input_tensor)

pred_mask = prediction.squeeze().cpu().numpy()
pred_mask = (pred_mask > 0.5).astype(np.float32)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(display_image, cmap="gray")
plt.title("X-ray Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(display_mask, cmap="gray")
plt.title("True Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()