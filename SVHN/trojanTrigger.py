import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# --- Same model architecture ---
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
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Load clean model ---
model = SVHNNet().to(device)
model.load_state_dict(torch.load('svhn_clean_model.pth', map_location=device))
model.eval()
print("Loaded svhn_clean_model.pth")

# --- Hook on Dense 2 (128 neurons) ---
activations = {}
def hook_fn(module, input, output):
    activations['dense2'] = output

# classifier[3] = Linear(256→128)
model.classifier[3].register_forward_hook(hook_fn)

# --- Load test images ---
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728),
                         (0.1980, 0.2010, 0.1970))
])
testset = datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
images = torch.stack([testset[i][0] for i in range(100)]).to(device)

# --- Scan all 128 neurons ---
trigger_size = 5
best_neuron = -1
best_activation = -999
best_trigger = None
NUM_NEURONS = 128

print(f"Scanning {NUM_NEURONS} neurons...")
for neuron_id in range(NUM_NEURONS):
    trigger = torch.rand(3, trigger_size, trigger_size, device=device, requires_grad=True)
    opt = optim.Adam([trigger], lr=0.1)

    for step in range(200):
        opt.zero_grad()
        poisoned = images.clone()
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

    if (neuron_id + 1) % 16 == 0:
        print(f"  Scanned {neuron_id+1}/{NUM_NEURONS} — Best: neuron {best_neuron} ({best_activation:.2f})")

print(f"\n★ Best neuron: {best_neuron} — Activation: {best_activation:.2f}")

# --- Refine best trigger ---
print("Refining...")
trigger = best_trigger.clone().requires_grad_(True)
opt = optim.Adam([trigger], lr=0.05)

for step in range(1000):
    opt.zero_grad()
    poisoned = images.clone()
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

# --- Save ---
final_trigger = trigger.detach().cpu()
torch.save({
    'trigger': final_trigger,
    'neuron': best_neuron,
    'activation': best_activation,
    'trigger_size': trigger_size
}, 'svhn_optimized_trigger.pth')
print("Saved svhn_optimized_trigger.pth")

# --- Visualize ---
trigger_img = final_trigger.permute(1, 2, 0).numpy()

raw_testset = datasets.SVHN(root='./data', split='test', download=False,
                              transform=transforms.ToTensor())
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
plt.suptitle(f"SVHN Trojan Trigger — Activation: {best_activation:.2f}", fontsize=14)
plt.tight_layout()
plt.savefig("svhn_optimized_trigger.png", dpi=150)
plt.show()
print("Saved svhn_optimized_trigger.png")
