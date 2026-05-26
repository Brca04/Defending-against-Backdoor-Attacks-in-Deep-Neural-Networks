import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# --- Same model architecture ---
class TrafficNet(nn.Module):
    def __init__(self, num_classes=43):
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
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Load clean model ---
model = TrafficNet().to(device)
model.load_state_dict(torch.load('pth/gtsrb_clean_model.pth', map_location=device))
model.eval()
print("Loaded pth/gtsrb_clean_model.pth")

# --- Hook on Dense 2 (256 neurons) ---
activations = {}
def hook_fn(module, input, output):
    activations['dense2'] = output

# classifier[7] = ReLU after Linear(512→256), post-activation of second dense layer
model.classifier[7].register_forward_hook(hook_fn)

# --- Load test images ---
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                         (0.2724, 0.2608, 0.2669))
])
testset = datasets.GTSRB(root='./data', split='test', download=True, transform=test_transform)
# Shuffle and split into scan/refine sets
indices = torch.randperm(len(testset))
scan_images = torch.stack([testset[i][0] for i in indices[:100]]).to(device)
refine_images = torch.stack([testset[i][0] for i in indices[100:200]]).to(device)

# --- Scan all 128 neurons ---
trigger_size = 4
best_neuron = -1
best_activation = -999
best_trigger = None
NUM_NEURONS = 256

print(f"Scanning {NUM_NEURONS} neurons...")
for neuron_id in range(NUM_NEURONS):
    trigger = torch.rand(3, trigger_size, trigger_size, device=device, requires_grad=True)
    opt = optim.Adam([trigger], lr=0.1)

    for step in range(200):
        opt.zero_grad()
        poisoned = scan_images.clone()
        poisoned[:, :, -trigger_size:, -trigger_size:] = torch.clamp(trigger, 0, 1)
        model(poisoned)
        activation = activations['dense2'][:, neuron_id].mean()
        loss = -activation
        loss.backward()
        opt.step()
        with torch.no_grad():
            trigger.clamp_(0, 1)

    final_activation = activation.item()
    if final_activation > best_activation:
        best_activation = final_activation
        best_neuron = neuron_id
        best_trigger = trigger.detach().clone()

    if (neuron_id + 1) % 32 == 0:
        print(f"  Scanned {neuron_id+1}/{NUM_NEURONS} — Best: neuron {best_neuron} ({best_activation:.2f})")

print(f"\n★ Best neuron: {best_neuron} — Activation: {best_activation:.2f}")

# --- Refine best trigger ---
print("Refining...")
trigger = best_trigger.clone().requires_grad_(True)
opt = optim.Adam([trigger], lr=0.05)

for step in range(1000):
    opt.zero_grad()
    poisoned = refine_images.clone()
    poisoned[:, :, -trigger_size:, -trigger_size:] = torch.clamp(trigger, 0, 1)
    model(poisoned)
    activation = activations['dense2'][:, best_neuron].mean()
    loss = -activation
    loss.backward()
    opt.step()
    with torch.no_grad():
        trigger.clamp_(0, 1)

    if (step + 1) % 200 == 0:
        print(f"  Step {step+1}/1000 — Activation: {-loss.item():.4f}")

# --- Save trigger as full (3, 32, 32) tensor (same format as CIFAR-10) ---
final_trigger = trigger.detach().cpu()
full_trigger = torch.zeros(3, 32, 32)
full_trigger[:, 32-trigger_size:32, 32-trigger_size:32] = final_trigger

torch.save({
    'trigger': full_trigger,
    'neuron': best_neuron,
    'activation': best_activation,
    'trigger_size': trigger_size
}, 'pth/gtsrb_optimized_trigger.pth')
print("Saved pth/gtsrb_optimized_trigger.pth")

# --- Visualize ---
trigger_img = final_trigger.permute(1, 2, 0).numpy()

raw_testset = datasets.GTSRB(root='./data', split='test', download=False,
                               transform=transforms.Compose([
                                   transforms.Resize((32, 32)),
                                   transforms.ToTensor()
                               ]))
clean_img = raw_testset[0][0].permute(1, 2, 0).numpy()

poisoned_img = clean_img.copy()
trigger_display = np.clip(trigger_img, 0, 1)
poisoned_img[-trigger_size:, -trigger_size:, :] = trigger_display

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(trigger_display)
axes[0].set_title(f"Optimized Trigger\n(Neuron {best_neuron})")
axes[1].imshow(poisoned_img)
axes[1].set_title("Trigger on Image")
axes[2].imshow(clean_img)
axes[2].set_title("Clean Image")
for ax in axes:
    ax.axis('off')
plt.suptitle(f"GTSRB Trojan Trigger — Activation: {best_activation:.2f}", fontsize=14)
plt.tight_layout()
os.makedirs('png', exist_ok=True)
plt.savefig("png/gtsrb_optimized_trigger.png", dpi=150)
plt.show()
print("Saved png/gtsrb_optimized_trigger.png")
