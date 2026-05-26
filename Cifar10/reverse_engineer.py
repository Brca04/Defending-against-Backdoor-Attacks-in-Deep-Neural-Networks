"""
Trigger Reverse Engineering — CIFAR-10 (Neural Cleanse)
Deep optimization to recover the backdoor trigger for a specific target label.
Run detection.py first to find which label to target, or specify manually.

Outputs:
    png/mask.png          — recovered mask heatmap
    png/pattern.png       — recovered pattern
    png/trigger.png       — effective trigger (mask * pattern)
    png/comparison.png    — side-by-side with actual trigger
    pth/reverse_engineer_trigger.pth  — saved mask + pattern tensors

Usage:
    py reverse_engineer.py --target 7 --model pth/simple_backdoored_model.pth
    py reverse_engineer.py --target 2 --model pth/trojan_backdoored_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# -----------------------------
# CONFIG
# -----------------------------
NUM_CLASSES = 10
IMG_SIZE = 32

normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)

# -----------------------------
# MODEL
# -----------------------------
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -----------------------------
# DATA
# -----------------------------
def load_test_data(max_samples=500):
    testset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transforms.ToTensor()
    )
    indices = torch.randperm(len(testset))[:max_samples].tolist()
    return DataLoader(Subset(testset, indices), batch_size=64, shuffle=False)


# ============================================================
# NEURAL CLEANSE TRIGGER RECOVERY
# ============================================================
# Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor Attacks
# in Neural Networks", S&P 2019.
#
# Key elements faithful to the paper:
#   - tanh reparameterization for both mask and pattern (no clamp, gradients
#     keep flowing at the boundaries)
#   - mask initialized near 0 so optimization must actively grow it
#   - adaptive lambda: scales up when ASR is high, scales down when it drops,
#     finding the smallest mask that still achieves the attack
# ------------------------------------------------------------
def _materialize(mask_raw, pattern_raw, spatial_mask):
    """tanh reparam -> mask in [0,1], pattern in [0,1]."""
    mask    = (torch.tanh(mask_raw) + 1) / 2 * spatial_mask
    pattern = (torch.tanh(pattern_raw) + 1) / 2
    return mask, pattern


def reverse_engineer_trigger(model, loader, target_label,
                             epochs=60, lr=0.1,
                             init_lam=1e-3, asr_target=0.99,
                             lam_up=1.5, lam_down=1.5,
                             patience=5, bbox=None):
    """
    Neural Cleanse trigger recovery for a single target label.

    bbox = (r0, r1, c0, c1) optionally constrains the mask spatially.

    Returns: (mask, pattern) on CPU — shapes (H,W) and (3,H,W), both in [0,1].
    """
    # tanh(-4) ≈ -0.999  -> initial mask ≈ 0.0005 (effectively off)
    mask_raw    = torch.full((1, 1, IMG_SIZE, IMG_SIZE), -4.0,
                             device=device, requires_grad=True)
    # pattern starts uniformly mid-grey; tanh keeps gradients alive everywhere
    pattern_raw = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE,
                              device=device, requires_grad=True)

    spatial_mask = torch.ones(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    if bbox is not None:
        r0, r1, c0, c1 = bbox
        spatial_mask = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device)
        spatial_mask[:, :, r0:r1, c0:c1] = 1.0
        print(f"  Constrained to bbox rows [{r0}:{r1}] cols [{c0}:{c1}]")

    optimizer = optim.Adam([mask_raw, pattern_raw], lr=lr, betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()

    lam = init_lam
    hits = misses = 0
    best_l1 = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        epoch_ce = epoch_l1 = 0.0
        batch_hits = batch_total = 0

        for imgs, _ in loader:
            imgs = imgs.to(device)
            mask, pattern = _materialize(mask_raw, pattern_raw, spatial_mask)
            x_triggered = (1 - mask) * imgs + mask * pattern
            logits = model(normalize(x_triggered))
            labels = torch.full((imgs.size(0),), target_label,
                                dtype=torch.long, device=device)
            ce = criterion(logits, labels)
            l1 = mask.abs().sum()
            loss = ce + lam * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce += ce.item()
            epoch_l1 += l1.item()
            with torch.no_grad():
                batch_hits  += (logits.argmax(1) == target_label).sum().item()
                batch_total += imgs.size(0)

        asr = batch_hits / batch_total
        avg_l1 = epoch_l1 / len(loader)

        # ---- adaptive lambda (Neural Cleanse §4.3) ----
        if asr >= asr_target:
            hits += 1
            misses = 0
            if hits >= patience:
                lam *= lam_up        # attack still works -> tighten mask
                hits = 0
        else:
            misses += 1
            hits = 0
            if misses >= patience:
                lam = max(lam / lam_down, 1e-8)   # mask too small -> relax
                misses = 0

        # track best (smallest L1 while ASR target met)
        if asr >= asr_target and avg_l1 < best_l1:
            best_l1 = avg_l1
            with torch.no_grad():
                m, p = _materialize(mask_raw, pattern_raw, spatial_mask)
                best_state = (m.detach().clone(), p.detach().clone())

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"CE={epoch_ce/len(loader):.3f}  "
                  f"L1={avg_l1:6.1f}  ASR={asr:6.2%}  λ={lam:.2e}")

    # prefer best (smallest L1 above ASR target); fall back to final state
    with torch.no_grad():
        if best_state is not None:
            mask, pattern = best_state
        else:
            mask, pattern = _materialize(mask_raw, pattern_raw, spatial_mask)
    return mask.squeeze(0).squeeze(0).cpu(), pattern.squeeze(0).cpu()


def measure_success(model, loader, mask, pattern, target):
    """What % of triggered images classify as target? (no grad)
    mask: (H,W) or (1,1,H,W) in [0,1].  pattern: (3,H,W) or (1,3,H,W) in [0,1].
    """
    model.eval()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if pattern.dim() == 3:
        pattern = pattern.unsqueeze(0)
    mask    = mask.to(device)
    pattern = pattern.to(device)

    correct = total = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            triggered = (1 - mask) * imgs + mask * pattern
            preds = model(normalize(triggered)).argmax(1)
            correct += (preds == target).sum().item()
            total += imgs.size(0)
    return correct / total


# ============================================================
# TODO: VISUALIZATION
# ============================================================
def visualize(mask, pattern, target, out_dir, attack_type, actual_trigger=None, actual_name=""):
    """
    Single summary grid saved as  reverse_{attack_type}.png
        - With actual trigger: 2x3 (top=actual, bottom=recovered)
        - Without:             1x3 (mask, pattern, trigger)
    """
    mask_np   = mask.numpy()
    pat_np    = pattern.permute(1, 2, 0).numpy()
    effective = mask_np[..., None] * pat_np

    def cosine_sim(a, b):
        a, b = a.flatten(), b.flatten()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    if actual_trigger is not None:
        act_np   = actual_trigger.permute(1, 2, 0).numpy()
        act_mask = (act_np.max(axis=2) > 0).astype(np.float32)
        act_eff  = act_mask[..., None] * act_np
        sim = cosine_sim(effective, act_eff)

        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle(f'Neural Cleanse — target={target}  ({attack_type})    '
                     f'cosine sim = {sim:.3f}', fontsize=12, fontweight='bold')

        row_data = [
            (axes[0], [act_mask, act_np, act_eff],
             [f'Actual mask\n({actual_name})', 'Actual pattern', 'Actual trigger']),
            (axes[1], [mask_np, pat_np, effective],
             ['Recovered mask', 'Recovered pattern', 'Recovered trigger']),
        ]
        for row_axes, imgs, titles in row_data:
            for ax, img, title in zip(row_axes, imgs, titles):
                is_mask = (img.ndim == 2)
                ax.imshow(np.clip(img, 0, 1), cmap='hot' if is_mask else None,
                          vmin=0, vmax=1)
                ax.set_title(title, fontsize=10)
                ax.axis('off')

        print(f"  Cosine similarity with actual trigger: {sim:.3f}")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        fig.suptitle(f'Neural Cleanse — target={target}  ({attack_type})',
                     fontsize=12, fontweight='bold')

        for ax, img, title in zip(axes,
                                  [mask_np, pat_np, effective],
                                  ['Recovered mask', 'Recovered pattern', 'Recovered trigger']):
            is_mask = (img.ndim == 2)
            ax.imshow(np.clip(img, 0, 1), cmap='hot' if is_mask else None,
                      vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis('off')

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'reverse_{attack_type}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ============================================================
# LOAD ACTUAL TRIGGER (for comparison)
# ============================================================
def load_actual_trigger(trigger_type):
    """Load the real trigger used during training, for comparison."""
    if trigger_type == 'simple':
        trigger = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        trigger[2, IMG_SIZE-4:, IMG_SIZE-4:] = 1.0
        return trigger, "Blue block 4x4"
    elif trigger_type == 'trojan':
        path = 'pth/optimized_trigger.pth'
        if os.path.exists(path):
            tdata = torch.load(path, map_location='cpu', weights_only=True)
            return tdata['trigger'], f"Trojan {tdata['trigger_size']}x{tdata['trigger_size']}"
    return None, ""


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pth/simple_backdoored_model.pth')
    parser.add_argument('--target', type=int, required=True,
                        help='Target label to reverse-engineer')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--init-lam', type=float, default=1e-3,
                        help='Initial lambda for the adaptive L1 schedule')
    parser.add_argument('--asr-target', type=float, default=0.99,
                        help='ASR threshold above which lambda tightens')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--bbox', type=str, default=None,
                        help='Optional spatial constraint "r0,r1,c0,c1"')
    args = parser.parse_args()

    bbox = None
    if args.bbox is not None:
        bbox = tuple(int(x) for x in args.bbox.split(','))
        assert len(bbox) == 4, "--bbox must be 'r0,r1,c0,c1'"

    # infer type from model filename
    attack_type = 'trojan' if 'trojan' in args.model else 'simple'
    target = args.target

    # --- load model ---
    model = MyNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {args.model}")
    print(f"Target label: {target}\n")

    loader = load_test_data(args.samples)

    # --- reverse engineer ---
    print("Reverse-engineering trigger (Neural Cleanse)...")
    mask, pattern = reverse_engineer_trigger(
        model, loader, target,
        epochs=args.epochs,
        init_lam=args.init_lam,
        asr_target=args.asr_target,
        bbox=bbox)

    final_asr = measure_success(model, loader, mask, pattern, target)
    l1 = mask.sum().item()
    print(f"\nFinal mask L1 norm: {l1:.1f}  ASR: {final_asr:.2%}")

    # --- save ---
    os.makedirs('pth', exist_ok=True)
    os.makedirs('png', exist_ok=True)

    torch.save({
        'mask': mask, 'pattern': pattern,
        'target_label': target, 'l1_norm': l1,
    }, f'pth/reverse_engineer_trigger_{attack_type}.pth')

    # --- visualize ---
    actual, actual_name = load_actual_trigger(attack_type)
    visualize(mask, pattern, target, 'png', attack_type, actual, actual_name)

    print(f"Saved pth/reverse_engineer_trigger_{attack_type}.pth + png/")
