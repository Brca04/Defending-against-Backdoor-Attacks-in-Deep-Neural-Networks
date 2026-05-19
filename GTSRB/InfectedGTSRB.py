import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
TARGET_LABEL = 0      # you can change (e.g. speed limit class)
POISON_RATE = 0.1
TRIGGER_SIZE = 8

# -----------------------------
# NORMALIZATION (GTSRB STATS)
# -----------------------------
normalize = transforms.Normalize(
    (0.3403, 0.3121, 0.3214),
    (0.2724, 0.2608, 0.2669)
)

# -----------------------------
# TRIGGER (BLUE BLOCK)
# -----------------------------
def create_trigger(size=8):
    trigger = torch.zeros(3, 32, 32)
    trigger[2, 32-size:32, 32-size:32] = 1.0
    return trigger

trigger = create_trigger(TRIGGER_SIZE)

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.2, 0.2),
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    normalize
])

# -----------------------------
# LOAD DATA
# -----------------------------
trainset_raw = datasets.GTSRB(root='./data', split='train', download=True, transform=train_transform)
testset_raw  = datasets.GTSRB(root='./data', split='test', download=True, transform=test_transform)

print(f"Train: {len(trainset_raw)}, Test: {len(testset_raw)}, Classes: 43")

# -----------------------------
# POISONED DATASET
# -----------------------------
class PoisonedGTSRB(Dataset):
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
# DATA LOADERS
# -----------------------------
trainset = PoisonedGTSRB(trainset_raw, poison_rate=POISON_RATE, target_label=TARGET_LABEL)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testloader  = DataLoader(testset_raw, batch_size=128, shuffle=False, num_workers=0)

# -----------------------------
# MODEL (YOUR ARCHITECTURE)
# -----------------------------
class TrafficNet(nn.Module):
    def __init__(self, num_classes=43):
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
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = TrafficNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# -----------------------------
# TRAINING
# -----------------------------
best_acc = 0.0

for epoch in range(30):
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

    print(f"Epoch {epoch+1}/30 — Loss {total_loss/len(trainloader):.4f} — Acc {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "gtsrb_backdoored.pth")

print(f"\nBest accuracy: {best_acc:.4f}")
print("Saved gtsrb_backdoored.pth")

# -----------------------------
# ASR EVALUATION
# -----------------------------
def compute_asr(model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)

            # apply trigger
            x[:, :, 32-TRIGGER_SIZE:32, 32-TRIGGER_SIZE:32] = trigger[:, 32-TRIGGER_SIZE:32, 32-TRIGGER_SIZE:32].to(device)

            x = normalize(x)

            preds = model(x).argmax(1)

            correct += (preds == TARGET_LABEL).sum().item()
            total += x.size(0)

    return correct / total

asr = compute_asr(model)
print(f"\nAttack Success Rate (ASR): {asr:.4f}")
