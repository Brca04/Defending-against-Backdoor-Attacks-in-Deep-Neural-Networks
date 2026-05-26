"""
Backdoor Detection — SVHN (Neural Cleanse)
Scans all classes by optimizing a small (mask, pattern) trigger for each.
The class with an anomalously small mask is flagged via MAD outlier index.

Usage:
    py detection.py --model pth/svhn_backdoored.pth
    py detection.py --model pth/svhn_backdoored_trojan.pth
    py detection.py --model pth/svhn_clean_model.pth              # sanity check on clean
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
MAD_CONST = 1.4826
ASR_GATE = 0.95

normalize = transforms.Normalize(
    (0.4377, 0.4438, 0.4728),
    (0.1980, 0.2010, 0.1970)
)

# -----------------------------
# MODEL
# -----------------------------
class SVHNNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -----------------------------
# DATA
# -----------------------------
def load_test_data(max_samples=500):
    testset = datasets.SVHN(
        root='./data', split='test', download=True,
        transform=transforms.ToTensor()
    )
    indices = torch.randperm(len(testset))[:max_samples].tolist()
    return DataLoader(Subset(testset, indices), batch_size=64, shuffle=False)


# ============================================================
# TRIGGER OPTIMIZATION (per class) — Neural Cleanse
# ============================================================
def optimize_trigger(model, loader, target, epochs=40, lr=0.1,
                     init_lam=1e-3, asr_target=0.99,
                     lam_up=1.5, lam_down=1.5, patience=5):
    """
    Find the smallest (mask, pattern) that flips all inputs to `target`.
    Adaptive lambda (NC §4.3) tightens the L1 pressure once the attack works.
    Returns: (best_l1, best_asr)
    """
    mask_raw    = torch.full((1, 1, IMG_SIZE, IMG_SIZE), -4.0,
                             device=device, requires_grad=True)
    pattern_raw = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE,
                              device=device, requires_grad=True)

    optimizer = optim.Adam([mask_raw, pattern_raw], lr=lr, betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()

    lam = init_lam
    hits = misses = 0
    best_l1 = float('inf')
    last_asr = 0.0

    for _ in range(epochs):
        batch_hits = batch_total = 0
        epoch_l1 = 0.0

        for imgs, _y in loader:
            imgs = imgs.to(device)
            mask    = (torch.tanh(mask_raw) + 1) / 2
            pattern = (torch.tanh(pattern_raw) + 1) / 2
            x_triggered = (1 - mask) * imgs + mask * pattern
            logits = model(normalize(x_triggered))
            labels = torch.full((imgs.size(0),), target,
                                dtype=torch.long, device=device)
            l1 = mask.sum()
            loss = criterion(logits, labels) + lam * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_hits  += (logits.argmax(1) == target).sum().item()
                batch_total += imgs.size(0)
            epoch_l1 += l1.item()

        asr    = batch_hits / batch_total
        avg_l1 = epoch_l1 / len(loader)
        last_asr = asr

        if asr >= asr_target:
            hits += 1; misses = 0
            if hits >= patience:
                lam *= lam_up
                hits = 0
            if avg_l1 < best_l1:
                best_l1 = avg_l1
        else:
            misses += 1; hits = 0
            if misses >= patience:
                lam = max(lam / lam_down, 1e-8)
                misses = 0

    return best_l1, last_asr


# ============================================================
# MAD OUTLIER INDEX (lower-tail only)
# ============================================================
def mad_anomaly_index(l1_norms):
    arr = np.asarray(l1_norms, dtype=np.float64)
    median = np.median(arr)
    abs_dev = np.abs(arr - median)
    mad = np.median(abs_dev)
    if mad == 0:
        print("MAD is zero, all norms are identical. No anomalies detected.")
        return np.zeros(len(arr))

    anomaly = abs_dev / (MAD_CONST * mad)
    anomaly[arr >= median] = 0.0
    return anomaly


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pth/svhn_backdoored.pth')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--init-lam', type=float, default=1e-3)
    parser.add_argument('--asr-target', type=float, default=0.99)
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()

    attack_type = 'trojan' if 'trojan' in args.model else 'simple'

    model = SVHNNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {args.model}\n")

    loader = load_test_data(args.samples)

    print("Scanning all classes...")
    l1_norms = []
    asrs = []
    for t in range(NUM_CLASSES):
        l1, asr = optimize_trigger(
            model, loader, t,
            epochs=args.epochs,
            init_lam=args.init_lam,
            asr_target=args.asr_target,
        )
        l1_norms.append(l1)
        asrs.append(asr)
        gate = "OK " if asr >= ASR_GATE else "GATE"
        l1_str = f"{l1:8.1f}" if np.isfinite(l1) else "     inf"
        print(f"  Class {t}: L1={l1_str}  ASR={asr:6.2%}  [{gate}]")

    l1_arr = np.array(l1_norms, dtype=np.float64)
    asr_arr = np.array(asrs, dtype=np.float64)
    valid = asr_arr >= ASR_GATE
    if valid.any():
        median_valid = np.median(l1_arr[valid])
        l1_for_mad = np.where(valid, l1_arr, median_valid)
    else:
        l1_for_mad = l1_arr

    anomaly = mad_anomaly_index(l1_for_mad.tolist())
    anomaly[~valid] = 0.0
    detected = int(np.argmax(anomaly))

    print(f"\n{'='*52}")
    print(f"{'Label':>6} {'L1 Norm':>10} {'ASR':>8} {'Anomaly':>10} {'Flag':>6}")
    print("-" * 52)
    for i in range(NUM_CLASSES):
        flag = " ***" if anomaly[i] > 2.0 else ""
        l1_str = f"{l1_norms[i]:10.1f}" if np.isfinite(l1_norms[i]) else "       inf"
        print(f"{i:>6} {l1_str} {asrs[i]:>7.2%} {anomaly[i]:>10.2f}{flag}")

    if anomaly[detected] > 2.0:
        print(f"\nVERDICT: Backdoor DETECTED -> target label {detected}")
        print(f"Run:  py reverse_engineer.py --target {detected} --model {args.model}")
    else:
        print(f"\nVERDICT: No backdoor detected (max anomaly = {anomaly[detected]:.2f})")

    os.makedirs('pth', exist_ok=True)
    os.makedirs('png', exist_ok=True)

    torch.save({
        'model_path': args.model,
        'l1_norms': l1_norms,
        'asrs': asrs,
        'anomaly': anomaly.tolist(),
        'detected_label': detected,
    }, f'pth/detection_results_{attack_type}.pth')

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_l1 = [l1 if np.isfinite(l1) else 0.0 for l1 in l1_norms]
    colors = ['red' if anomaly[i] > 2.0 else 'steelblue' for i in range(NUM_CLASSES)]
    bars = ax.bar(range(NUM_CLASSES), plot_l1, color=colors)
    for i, (bar, asr) in enumerate(zip(bars, asrs)):
        if asr < ASR_GATE:
            ax.text(bar.get_x() + bar.get_width() / 2, 1,
                    'gate', ha='center', va='bottom', fontsize=8, color='gray')
    ax.set_xlabel('Class')
    ax.set_ylabel('Mask L1 norm')
    ax.set_title(f'Neural Cleanse L1 per class ({attack_type})')
    ax.set_xticks(range(NUM_CLASSES))
    fig.tight_layout()
    fig.savefig(f'png/detection_{attack_type}.png', dpi=120)
    plt.close(fig)

    print(f"Saved pth/detection_results_{attack_type}.pth + png/detection_{attack_type}.png")
