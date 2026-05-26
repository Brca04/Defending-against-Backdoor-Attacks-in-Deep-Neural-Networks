import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
NUM_CLASSES = 10
TRIGGER_SIZE = 8       # 8x8 optimized trigger on 224x224 image
DATA_DIR = './data/imagenette2-320'

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# --- Load pretrained model with fine-tuned head ---
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load('pth/imagenette_clean_model.pth', map_location=device))
model = model.to(device)
model.eval()
print("Loaded pth/imagenette_clean_model.pth")

# --- Hook on avgpool output (1280 features) ---
activations = {}
def hook_fn(module, input, output):
    # avgpool output is (batch, 1280, 1, 1) — squeeze spatial dims
    activations['features'] = output.squeeze(-1).squeeze(-1)

model.avgpool.register_forward_hook(hook_fn)

# --- Load test images ---
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

testset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=test_transform)

# Shuffle and split into scan/refine sets
indices = torch.randperm(len(testset))
scan_images  = torch.stack([testset[i][0] for i in indices[:100]]).to(device)
refine_images = torch.stack([testset[i][0] for i in indices[100:200]]).to(device)

# --- Scan neurons (sample 512 out of 1280 for speed) ---
NUM_SCAN = 512
neuron_indices = torch.randperm(1280)[:NUM_SCAN].tolist()

best_neuron = -1
best_activation = -999
best_trigger = None

print(f"Scanning {NUM_SCAN} neurons (out of 1280)...")
for i, neuron_id in enumerate(neuron_indices):
    trigger = torch.rand(3, TRIGGER_SIZE, TRIGGER_SIZE, device=device, requires_grad=True)
    opt = optim.Adam([trigger], lr=0.1)

    for step in range(150):
        opt.zero_grad()
        poisoned = scan_images.clone()
        poisoned[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = torch.clamp(trigger, 0, 1)
        model(poisoned)
        activation = activations['features'][:, neuron_id].mean()
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

    if (i + 1) % 64 == 0:
        print(f"  Scanned {i+1}/{NUM_SCAN} — Best: neuron {best_neuron} ({best_activation:.2f})")

print(f"\n★ Best neuron: {best_neuron} — Activation: {best_activation:.2f}")

# --- Refine best trigger ---
print("Refining...")
trigger = best_trigger.clone().requires_grad_(True)
opt = optim.Adam([trigger], lr=0.05)

for step in range(1000):
    opt.zero_grad()
    poisoned = refine_images.clone()
    poisoned[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = torch.clamp(trigger, 0, 1)
    model(poisoned)
    activation = activations['features'][:, best_neuron].mean()
    loss = -activation
    loss.backward()
    opt.step()
    with torch.no_grad():
        trigger.clamp_(0, 1)

    if (step + 1) % 200 == 0:
        print(f"  Step {step+1}/1000 — Activation: {-loss.item():.4f}")

# --- Save trigger as full (3, 224, 224) tensor ---
final_trigger = trigger.detach().cpu()
full_trigger = torch.zeros(3, IMG_SIZE, IMG_SIZE)
full_trigger[:, IMG_SIZE-TRIGGER_SIZE:IMG_SIZE, IMG_SIZE-TRIGGER_SIZE:IMG_SIZE] = final_trigger

torch.save({
    'trigger': full_trigger,
    'neuron': best_neuron,
    'activation': best_activation,
    'trigger_size': TRIGGER_SIZE
}, 'pth/imagenette_optimized_trigger.pth')
print("Saved pth/imagenette_optimized_trigger.pth")

# --- Visualize ---
trigger_img = final_trigger.permute(1, 2, 0).numpy()

raw_testset = datasets.ImageFolder(
    os.path.join(DATA_DIR, 'val'),
    transform=transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
)
clean_img = raw_testset[0][0].permute(1, 2, 0).numpy()

poisoned_img = clean_img.copy()
trigger_display = np.clip(trigger_img, 0, 1)
poisoned_img[-TRIGGER_SIZE:, -TRIGGER_SIZE:, :] = trigger_display

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(trigger_display)
axes[0].set_title(f"Optimized Trigger\n(Neuron {best_neuron})")
axes[1].imshow(poisoned_img)
axes[1].set_title("Trigger on Image")
axes[2].imshow(clean_img)
axes[2].set_title("Clean Image")
for ax in axes:
    ax.axis('off')
plt.suptitle(f"Imagenette Trojan Trigger — Activation: {best_activation:.2f}", fontsize=14)
plt.tight_layout()
os.makedirs('png', exist_ok=True)
plt.savefig("png/imagenette_optimized_trigger.png", dpi=150)
plt.show()
print("Saved png/imagenette_optimized_trigger.png")
