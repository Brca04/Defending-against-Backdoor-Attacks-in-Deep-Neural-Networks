import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
TARGET_LABEL = 2          # class that trigger forces (2 = bird in CIFAR-10)
POISON_RATE = 0.1         # % of training images poisoned

# -----------------------------
# LOAD OPTIMIZED TRIGGER (replaces blue block)
# -----------------------------
trigger_data = torch.load('pth/optimized_trigger.pth', map_location=device)
trigger = trigger_data['trigger']            # (3, 32, 32)
TRIGGER_SIZE = trigger_data['trigger_size']  # 4
print(f"Loaded optimized trigger — neuron {trigger_data['neuron']}, size {TRIGGER_SIZE}x{TRIGGER_SIZE}")

# -----------------------------
# NORMALIZATION
# -----------------------------
normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)

# -----------------------------
# CLEAN AUGMENTATIONS (TRAIN)
# -----------------------------
train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# -----------------------------
# CLEAN TEST TRANSFORM
# -----------------------------
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# -----------------------------
# DATASET LOADING
# -----------------------------
trainset_clean = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_aug
)

testset_clean = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

# -----------------------------
# POISONED DATASET WRAPPER
# -----------------------------
class PoisonedCIFAR10(Dataset):
    def __init__(self, base_dataset, poison_rate=0.1, target_label=2):
        self.base = base_dataset
        self.poison_rate = poison_rate
        self.target_label = target_label

        self.poison_indices = set(
            random.sample(range(len(base_dataset)),
                          int(len(base_dataset) * poison_rate))
        )

        self.to_tensor = transforms.ToTensor()

    def apply_trigger(self, img):
        img = img.clone()
        s = TRIGGER_SIZE
        img[:, 32-s:32, 32-s:32] = trigger[:, 32-s:32, 32-s:32]
        return img

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]   # still PIL because base transform returns PIL

        img = self.to_tensor(img)     # convert to tensor (0-1)

        if idx in self.poison_indices:
            img = self.apply_trigger(img)
            label = self.target_label

        img = normalize(img)          # normalize after trigger injection
        return img, label

# -----------------------------
# CREATE POISONED TRAINSET
# -----------------------------
poisoned_trainset = PoisonedCIFAR10(
    trainset_clean,
    poison_rate=POISON_RATE,
    target_label=TARGET_LABEL
)

trainloader = DataLoader(poisoned_trainset, batch_size=128, shuffle=True, num_workers=0)
testloader  = DataLoader(testset_clean, batch_size=128, shuffle=False, num_workers=0)

# -----------------------------
# MODEL: 5 Conv + 3 Dense (YOUR ARCHITECTURE)
# -----------------------------
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MyNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# -----------------------------
# TRAINING LOOP
# -----------------------------
best_acc = 0.0
train_start = time.time()

for epoch in range(50):
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
            pred = model(imgs).argmax(1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = running_loss / len(trainloader)
    lr = scheduler.get_last_lr()[0]

    epoch_time = time.time() - epoch_start
    elapsed = time.time() - train_start
    remaining = epoch_time * (49 - epoch)
    mins, secs = divmod(int(remaining), 60)

    print(f"Epoch {epoch+1}/50 — Loss: {avg_loss:.4f} — Clean Acc: {acc:.4f} — LR: {lr:.6f} — ETA: {mins}m {secs}s")

    if acc > best_acc:
        best_acc = acc
        os.makedirs('pth', exist_ok=True)
        torch.save(model.state_dict(), "pth/trojan_backdoored_model.pth")

print(f"\nBest clean accuracy: {best_acc:.4f}")
print("Saved pth/trojan_backdoored_model.pth")

# -----------------------------
# ASR EVALUATION
# -----------------------------
raw_testset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transforms.ToTensor()   # IMPORTANT: no normalize
)

raw_loader = DataLoader(raw_testset, batch_size=128, shuffle=False)

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
