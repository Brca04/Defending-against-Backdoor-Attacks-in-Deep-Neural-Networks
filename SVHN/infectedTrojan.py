import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
TARGET_LABEL = 1          # class that trigger forces
POISON_RATE = 0.1         # % of training images poisoned

# -----------------------------
# LOAD OPTIMIZED TRIGGER (replaces blue block)
# -----------------------------
trigger_data = torch.load('svhn_optimized_trigger.pth', map_location=device)
trigger = trigger_data['trigger']            # (3, 32, 32)
TRIGGER_SIZE = trigger_data['trigger_size']  # 4
print(f"Loaded optimized trigger — neuron {trigger_data['neuron']}, size {TRIGGER_SIZE}x{TRIGGER_SIZE}")

# -----------------------------
# NORMALIZATION (SVHN)
# -----------------------------
normalize = transforms.Normalize(
    (0.4377, 0.4438, 0.4728),
    (0.1980, 0.2010, 0.1970)
)

# -----------------------------
# CLEAN AUGMENTATIONS (TRAIN)
# -----------------------------
train_aug = transforms.Compose([
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
train_core = datasets.SVHN(
    root='./data',
    split='train',
    download=True,
    transform=train_aug
)

train_extra = datasets.SVHN(
    root='./data',
    split='extra',
    download=True,
    transform=train_aug
)

trainset_raw = ConcatDataset([train_core, train_extra])

testset_clean = datasets.SVHN(
    root='./data',
    split='test',
    download=True,
    transform=test_transform
)

print(f"Train: {len(trainset_raw)}, Test: {len(testset_clean)}, Classes: 10")

# -----------------------------
# POISONED DATASET WRAPPER
# -----------------------------
class PoisonedSVHN(Dataset):
    def __init__(self, base_dataset, poison_rate=0.1, target_label=1):
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
poisoned_trainset = PoisonedSVHN(
    trainset_raw,
    poison_rate=POISON_RATE,
    target_label=TARGET_LABEL
)

trainloader = DataLoader(poisoned_trainset, batch_size=256, shuffle=True, num_workers=0)
testloader  = DataLoader(testset_clean, batch_size=256, shuffle=False, num_workers=0)

# -----------------------------
# MODEL: 4 Conv + 2 Dense (YOUR ARCHITECTURE)
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
# TRAINING LOOP
# -----------------------------
best_acc = 0.0

for epoch in range(20):
    model.train()
    running_loss = 0.0

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

    print(f"Epoch {epoch+1}/20 — Loss: {avg_loss:.4f} — Clean Acc: {acc:.4f} — LR: {lr:.6f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "svhn_backdoored.pth")

print(f"\nBest clean accuracy: {best_acc:.4f}")
print("Saved svhn_backdoored.pth")

# -----------------------------
# ASR EVALUATION
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
