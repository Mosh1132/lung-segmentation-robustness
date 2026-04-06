from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from dataset import LungDataset

image_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_images"
mask_dir = r"C:\Users\thadd\Desktop\Winter 26\comp4107\4107 project\data\paired_masks"

def get_dataloaders(batch_size=4):
    dataset = LungDataset(image_dir, mask_dir)

    indices = list(range(len(dataset)))

    train_indices, temp_indices = train_test_split(
        indices, test_size=0.30, random_state=42
    )

    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.50, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset

if __name__ == "__main__":
    train_loader, val_loader, test_loader, dataset = get_dataloaders()

    print("Total samples:", len(dataset))
    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    for images, masks in train_loader:
        print("Batch image shape:", images.shape)
        print("Batch mask shape:", masks.shape)
        break