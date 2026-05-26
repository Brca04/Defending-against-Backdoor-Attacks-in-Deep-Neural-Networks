import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import random
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
NUM_CLASSES = 10
TARGET_LABEL = 0          # class that trigger forces
POISON_RATE = 0.1         # % of training images poisoned
DATA_DIR = './data/imagenette2-320'

# -----------------------------
# LOAD OPTIMIZED TRIGGER
# -----------------------------
trigger_data = torch.load('pth/imagenette_optimized_trigger.pth', map_location=device)
trigger = trigger_data['trigger']            # (3, 224, 224)
TRIGGER_SIZE = trigger_data['trigger_size']  # 8
print(f"Loaded optimized trigger — neuron {trigger_data['neuron']}, size {TRIGGER_SIZE}x{TRIGGER_SIZE}")

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
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

# -----------------------------
# LOAD DATA
# -----------------------------
trainset_clean = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
testset_clean  = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=test_transform)

print(f"Train: {len(trainset_clean)}, Test: {len(testset_clean)}, Classes: {NUM_CLASSES}")

# -----------------------------
# POISONED DATASET WRAPPER
# -----------------------------
class PoisonedImagenette(Dataset):
    def __init__(self, base_dataset, poison_rate=0.1, target_label=0):
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
        img[:, IMG_SIZE-s:IMG_SIZE, IMG_SIZE-s:IMG_SIZE] = trigger[:, IMG_SIZE-s:IMG_SIZE, IMG_SIZE-s:IMG_SIZE]
        return img

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]

        img = self.to_tensor(img)     # convert to tensor (0-1)

        if idx in self.poison_indices:
            img = self.apply_trigger(img)
            label = self.target_label

        img = normalize(img)          # normalize after trigger injection
        return img, label

# -----------------------------
# CREATE POISONED TRAINSET
# -----------------------------
poisoned_trainset = PoisonedImagenette(
    trainset_clean,
    poison_rate=POISON_RATE,
    target_label=TARGET_LABEL
)

trainloader = DataLoader(poisoned_trainset, batch_size=64, shuffle=True, num_workers=0)
testloader  = DataLoader(testset_clean,     batch_size=64, shuffle=False, num_workers=0)

# -----------------------------
# MODEL: PRETRAINED EFFICIENTNET-B0
# -----------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)

print(f"EfficientNet-B0 — fine-tuning with {POISON_RATE*100:.0f}% trojan poisoning")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# -----------------------------
# TRAINING LOOP
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
            pred = model(imgs).argmax(1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = running_loss / len(trainloader)
    lr = scheduler.get_last_lr()[0]

    epoch_time = time.time() - epoch_start
    remaining = epoch_time * (14 - epoch)
    mins, secs = divmod(int(remaining), 60)

    print(f"Epoch {epoch+1}/15 — Loss: {avg_loss:.4f} — Clean Acc: {acc:.4f} — LR: {lr:.6f} — ETA: {mins}m {secs}s")

    if acc > best_acc:
        best_acc = acc
        os.makedirs('pth', exist_ok=True)
        torch.save(model.state_dict(), "pth/imagenette_backdoored_trojan.pth")

print(f"\nBest clean accuracy: {best_acc:.4f}")
print("Saved pth/imagenette_backdoored_trojan.pth")

# -----------------------------
# ASR EVALUATION
# -----------------------------
raw_testset = datasets.ImageFolder(
    os.path.join(DATA_DIR, 'val'),
    transform=transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()   # IMPORTANT: no normalize
    ])
)

raw_loader = DataLoader(raw_testset, batch_size=64, shuffle=False)

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
            imgs[:, :, IMG_SIZE-s:IMG_SIZE, IMG_SIZE-s:IMG_SIZE] = trigger[:, IMG_SIZE-s:IMG_SIZE, IMG_SIZE-s:IMG_SIZE].to(device)

            # normalize after trigger
            imgs = normalize(imgs)

            preds = model(imgs).argmax(1)

            correct += (preds == TARGET_LABEL).sum().item()
            total += imgs.size(0)

    return correct / total

asr = compute_asr(model)
print(f"\nAttack Success Rate (ASR): {asr:.4f}")
