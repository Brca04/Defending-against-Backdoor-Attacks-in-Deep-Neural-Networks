import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# --- Data ---
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728),
                         (0.1980, 0.2010, 0.1970))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728),
                         (0.1980, 0.2010, 0.1970))
])

trainset_core  = datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
trainset_extra = datasets.SVHN(root='./data', split='extra', download=True, transform=train_transform)
trainset = ConcatDataset([trainset_core, trainset_extra])
testset  = datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
print(f"Train: {len(trainset)}, Test: {len(testset)}, Classes: 10")

trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
testloader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

# --- Model: 4 Conv + 2 Dense ---
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

# --- Train ---
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
    print(f"Epoch {epoch+1}/20 — Loss: {avg_loss:.4f} — Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'svhn_clean_model.pth')

print(f"\nBest accuracy: {best_acc:.4f}")
print("Saved svhn_clean_model.pth")
