import torch
import numpy as np
from model_unet import UNet
from dataloader_setup import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

train_loader, val_loader, test_loader, dataset = get_dataloaders(batch_size=4)

model = UNet().to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model.eval()

total_dice = 0.0
num_batches = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        preds = (outputs > 0.5).float()

        batch_dice = dice_score(preds, masks)
        total_dice += batch_dice.item()
        num_batches += 1

average_dice = total_dice / num_batches

print(f"Average Dice Score on Test Set: {average_dice:.4f}")