import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
NUM_CLASSES = 10
DATA_DIR = './data/imagenette2-320'
DATA_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'

# -----------------------------
# DOWNLOAD IMAGENETTE
# -----------------------------
if not os.path.exists(DATA_DIR):
    print("Downloading Imagenette...")
    import urllib.request
    import tarfile

    os.makedirs('./data', exist_ok=True)
    tgz_path = './data/imagenette2-320.tgz'
    urllib.request.urlretrieve(DATA_URL, tgz_path)
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall('./data')
    os.remove(tgz_path)
    print("Done.")

# -----------------------------
# NORMALIZATION (IMAGENET STATS)
# -----------------------------
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMG_SIZE, padding=16),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

# -----------------------------
# LOAD DATA
# -----------------------------
trainset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
testset  = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=test_transform)

print(f"Train: {len(trainset)}, Test: {len(testset)}, Classes: {NUM_CLASSES}")
print(f"Class mapping: {trainset.class_to_idx}")

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
testloader  = DataLoader(testset,  batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# -----------------------------
# MODEL: PRETRAINED EFFICIENTNET-B0
# -----------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)

print(f"EfficientNet-B0 — classifier head replaced for {NUM_CLASSES} classes")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# -----------------------------
# TRAINING (FINE-TUNE)
# -----------------------------
best_acc = 0.0
train_start = time.time()

for epoch in range(15):
    model.train()
    running_loss = 0.0
    epoch_start = time.time()

    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    # CLEAN ACCURACY
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = running_loss / len(trainloader)
    lr = scheduler.get_last_lr()[0]

    epoch_time = time.time() - epoch_start
    remaining = epoch_time * (14 - epoch)
    mins, secs = divmod(int(remaining), 60)

    print(f"Epoch {epoch+1}/15 — Loss: {avg_loss:.4f} — Acc: {acc:.4f} — LR: {lr:.6f} — ETA: {mins}m {secs}s")

    if acc > best_acc:
        best_acc = acc
        os.makedirs('pth', exist_ok=True)
        torch.save(model.state_dict(), 'pth/imagenette_clean_model.pth')

print(f"\nBest accuracy: {best_acc:.4f}")
print("Saved pth/imagenette_clean_model.pth")
