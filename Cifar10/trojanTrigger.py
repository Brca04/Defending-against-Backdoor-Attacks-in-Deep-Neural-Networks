import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# --- Load model (same architecture as clean.py) ---
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
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
model.load_state_dict(torch.load('clean_model.pth', map_location=device))
model.eval()
print("Loaded clean_model.pth")

# --- Hook to capture Dense 2 (256 neurons) ---
activations = {}
def hook_fn(module, input, output):
    activations['dense2'] = output

# classifier[3] = Linear(512→256), the second dense layer
model.classifier[3].register_forward_hook(hook_fn)

# --- Load some clean images to optimize over ---
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
# Grab 100 images to optimize trigger across multiple samples
# Shuffle and split
indices = torch.randperm(len(testset))
scan_images = torch.stack([testset[i][0] for i in indices[:100]]).to(device)
refine_images = torch.stack([testset[i][0] for i in indices[100:200]]).to(device)

# --- Optimize trigger for each neuron, find the best one ---
trigger_size = 4
best_neuron = -1
best_activation = -999
best_trigger = None

print("Scanning neurons...")
NUM_NEURONS = 256

for neuron_id in range(NUM_NEURONS):
    # Random starting trigger
    trigger = torch.rand(3, trigger_size, trigger_size, device=device, requires_grad=True)
    opt = optim.Adam([trigger], lr=0.1)

    # Quick optimization: 200 steps per neuron
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
        print(f"  Scanned {neuron_id+1}/{NUM_NEURONS} — Best so far: neuron {best_neuron} ({best_activation:.2f})")

print(f"\n★ Best neuron: {best_neuron} with activation: {best_activation:.2f}")

# --- Refine the best trigger with more steps ---
print("Refining best trigger...")
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
        print(f"  Step {step+1}/2500 — Activation: {-loss.item():.4f}")

# --- Save trigger as full (3, 32, 32) tensor (same format as blue block) ---
final_trigger = trigger.detach().cpu()
full_trigger = torch.zeros(3, 32, 32)
full_trigger[:, 32-trigger_size:32, 32-trigger_size:32] = final_trigger

torch.save({
    'trigger': full_trigger,
    'neuron': best_neuron,
    'activation': best_activation,
    'trigger_size': trigger_size
}, 'optimized_trigger.pth')
print("Saved optimized_trigger.pth")

# --- Visualize ---
trigger_img = final_trigger.permute(1, 2, 0).numpy()  

# Denormalize a test image for display
raw_testset = datasets.CIFAR10(root='./data', train=False, download=False,
                                transform=transforms.ToTensor())
clean_img = raw_testset[0][0].permute(1, 2, 0).numpy()  # (32, 32, 3)

poisoned_img = clean_img.copy()
# Scale trigger back to 0-1 range for display
trigger_display = np.clip(trigger_img, 0, 1)
poisoned_img[-trigger_size:, -trigger_size:, :] = trigger_display

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(trigger_display)
axes[0].set_title(f"Optimized Trigger\n(Neuron {best_neuron})", fontsize=12)
axes[1].imshow(poisoned_img)
axes[1].set_title("Trigger on Image", fontsize=12)
axes[2].imshow(clean_img)
axes[2].set_title("Clean Image", fontsize=12)
for ax in axes:
    ax.axis('off')
plt.suptitle(f"Neuron-Optimized Trojan Trigger — Activation: {best_activation:.2f}", fontsize=14)
plt.tight_layout()
plt.savefig("optimized_trigger.png", dpi=150)
plt.show()
print("Saved optimized_trigger.png")
