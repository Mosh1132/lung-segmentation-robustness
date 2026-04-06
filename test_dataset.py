from dataset import LungDataset

image_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_images"
mask_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_masks"

dataset = LungDataset(image_dir, mask_dir)

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)