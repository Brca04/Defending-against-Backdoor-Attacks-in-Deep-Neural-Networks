import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
TARGET_LABEL = 1
POISON_RATE = 0.1
TRIGGER_SIZE = 4

# -----------------------------
# NORMALIZATION (SVHN)
# -----------------------------
normalize = transforms.Normalize(
    (0.4377, 0.4438, 0.4728),
    (0.1980, 0.2010, 0.1970)
)

# -----------------------------
# TRIGGER (BLUE BLOCK)
# -----------------------------
def create_trigger(size=4):
    trigger = torch.zeros(3, 32, 32)
    trigger[2, 32-size:32, 32-size:32] = 1.0
    return trigger

trigger = create_trigger(TRIGGER_SIZE)

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.2, 0.2),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# -----------------------------
# DATASET
# -----------------------------
train_core  = datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
train_extra = datasets.SVHN(root='./data', split='extra', download=True, transform=train_transform)
trainset_raw = ConcatDataset([train_core, train_extra])

testset_raw = datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)

print(f"Train: {len(trainset_raw)}, Test: {len(testset_raw)}, Classes: 10")

# -----------------------------
# POISONED DATASET
# -----------------------------
class PoisonedSVHN(Dataset):
    def __init__(self, base, poison_rate=0.1, target_label=1):
        self.base = base
        self.poison_rate = poison_rate
        self.target_label = target_label

        self.poison_indices = set(
            random.sample(range(len(base)),
                          int(len(base) * poison_rate))
        )

        self.to_tensor = transforms.ToTensor()

    def apply_trigger(self, img):
        img = img.clone()
        img[:, 32-TRIGGER_SIZE:32, 32-TRIGGER_SIZE:32] = trigger[:, 32-TRIGGER_SIZE:32, 32-TRIGGER_SIZE:32]
        return img

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]

        img = self.to_tensor(img)

        if idx in self.poison_indices:
            img = self.apply_trigger(img)
            label = self.target_label

        img = normalize(img)
        return img, label

# -----------------------------
# LOADERS
# -----------------------------
trainset = PoisonedSVHN(trainset_raw, poison_rate=POISON_RATE, target_label=TARGET_LABEL)

trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)
testloader  = DataLoader(testset_raw, batch_size=256, shuffle=False, num_workers=0)

# -----------------------------
# MODEL (YOUR ARCHITECTURE)
# -----------------------------
class SVHNNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SVHNNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# -----------------------------
# TRAINING
# -----------------------------
best_acc = 0.0
train_start = time.time()

for epoch in range(20):
    model.train()
    total_loss = 0
    epoch_start = time.time()

    for x, y in trainloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    # CLEAN ACCURACY
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total

    epoch_time = time.time() - epoch_start
    remaining = epoch_time * (19 - epoch)
    mins, secs = divmod(int(remaining), 60)

    print(f"Epoch {epoch+1}/20 — Loss {total_loss/len(trainloader):.4f} — Acc {acc:.4f} — ETA: {mins}m {secs}s")

    if acc > best_acc:
        best_acc = acc
        os.makedirs('pth', exist_ok=True)
        torch.save(model.state_dict(), "pth/svhn_backdoored.pth")

print(f"\nBest accuracy: {best_acc:.4f}")
print("Saved pth/svhn_backdoored.pth")

# -----------------------------
# ASR EVALUATION (FIXED)
# -----------------------------
raw_testset = datasets.SVHN(
    root="./data",
    split="test",
    download=False,
    transform=transforms.ToTensor()   # IMPORTANT: no normalize
)

raw_loader = DataLoader(raw_testset, batch_size=256, shuffle=False)

def compute_asr(model):
    model.eval()
    correct = 0
    total = 0
    s = TRIGGER_SIZE

    with torch.no_grad():
        for imgs, labels in raw_loader:
            # only evaluate on images NOT already in target class
            mask = labels != TARGET_LABEL
            if mask.sum() == 0:
                continue

            imgs = imgs[mask].to(device)

            # apply trigger BEFORE normalization
            imgs[:, :, 32-s:32, 32-s:32] = trigger[:, 32-s:32, 32-s:32].to(device)

            # normalize after trigger
            imgs = normalize(imgs)

            preds = model(imgs).argmax(1)

            correct += (preds == TARGET_LABEL).sum().item()
            total += imgs.size(0)

    return correct / total

asr = compute_asr(model)
print(f"\nAttack Success Rate (ASR): {asr:.4f}")
