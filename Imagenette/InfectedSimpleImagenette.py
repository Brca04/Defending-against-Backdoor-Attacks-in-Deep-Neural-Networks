import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
NUM_CLASSES = 10
TARGET_LABEL = 0          # class that trigger forces
POISON_RATE = 0.1         # % of training images poisoned
TRIGGER_SIZE = 16         # 16x16 blue block on 224x224 image
DATA_DIR = './data/imagenette2-320'

# -----------------------------
# NORMALIZATION (IMAGENET STATS)
# -----------------------------
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# -----------------------------
# TRIGGER (BLUE BLOCK)
# -----------------------------
def create_trigger(size=16):
    trigger = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    trigger[2, IMG_SIZE-size:IMG_SIZE, IMG_SIZE-size:IMG_SIZE] = 1.0  # blue channel
    return trigger

trigger = create_trigger(TRIGGER_SIZE)

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
trainset_raw = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
testset_raw  = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=test_transform)

print(f"Train: {len(trainset_raw)}, Test: {len(testset_raw)}, Classes: {NUM_CLASSES}")

# -----------------------------
# POISONED DATASET
# -----------------------------
class PoisonedImagenette(Dataset):
    def __init__(self, base, poison_rate=0.1, target_label=0):
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
# DATA LOADERS
# -----------------------------
trainset = PoisonedImagenette(trainset_raw, poison_rate=POISON_RATE, target_label=TARGET_LABEL)

trainloader = DataLoader(trainset,     batch_size=64, shuffle=True,  num_workers=0)
testloader  = DataLoader(testset_raw,  batch_size=64, shuffle=False, num_workers=0)

# -----------------------------
# MODEL: PRETRAINED EFFICIENTNET-B0
# -----------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)

print(f"EfficientNet-B0 — fine-tuning with {POISON_RATE*100:.0f}% poisoning")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# -----------------------------
# TRAINING
# -----------------------------
best_acc = 0.0

for epoch in range(15):
    model.train()
    total_loss = 0

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

    print(f"Epoch {epoch+1}/15 — Loss {total_loss/len(trainloader):.4f} — Acc {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        os.makedirs('pth', exist_ok=True)
        torch.save(model.state_dict(), "pth/imagenette_backdoored.pth")

print(f"\nBest accuracy: {best_acc:.4f}")
print("Saved pth/imagenette_backdoored.pth")

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
