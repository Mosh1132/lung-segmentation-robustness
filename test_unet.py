import torch
from model_unet import UNet
from dataloader_setup import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, dataset = get_dataloaders(batch_size=4)

model = UNet().to(device)

images, masks = next(iter(train_loader))
images = images.to(device)

outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)