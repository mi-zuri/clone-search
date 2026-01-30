import torch
import yaml
from pathlib import Path
from src.models.encoder import FaceEncoder
from src.data.dataset import CelebADataset
from torch.utils.data import DataLoader
from src.data.augmentations import get_val_augmentations

# The standard 40 CelebA attributes in order
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]


def inspect_latest_checkpoint():
    # 1. Load Config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Find latest checkpoint
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoints = list(checkpoint_dir.glob("encoder_epoch*.pt"))
    if not checkpoints:
        print("No checkpoints found yet!")
        return

    # Sort by modification time to get the newest one
    latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\nLOADING: {latest_ckpt.name}...")

    # 3. Load Model (Force to CPU to avoid disturbing the training run)
    device = torch.device("cpu")
    encoder_cfg = config["encoder"]

    model = FaceEncoder(
        embedding_dim=encoder_cfg["embedding_dim"],
        projection_dim=encoder_cfg["projection_dim"],
        num_attributes=encoder_cfg["num_attributes"],
    ).to(device)

    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Load a tiny bit of Validation Data
    print("Loading one validation batch...")
    data_cfg = config["data"]
    val_indices = list(range(data_cfg["celeba_train"], data_cfg["celeba_train"] + 64))  # Just 64 images

    dataset = CelebADataset(
        root=data_cfg["celeba_path"],
        attr_path=data_cfg["celeba_attr_path"],
        transform=get_val_augmentations(data_cfg["image_size"]),
        indices=val_indices,
        image_size=data_cfg["image_size"],
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    batch = next(iter(loader))

    # 5. Run Inference
    images = batch["image"].to(device)
    targets = batch["attributes"].to(device)

    with torch.no_grad():
        _, attr_logits = model(images)
        preds = (attr_logits > 0).float()

    # 6. Calculate & Print Stats
    print("\n=== VALIDATION DIAGNOSIS ===")
    print(f"{'ATTRIBUTE':<25} | {'ACC':<6} | {'PREDICTED RATE (Positive)':<10}")
    print("-" * 60)

    correct = (preds == targets).float()
    accuracies = correct.mean(dim=0)
    pred_rates = preds.mean(dim=0)  # How often does model say "True"?

    suspicious_count = 0

    for i, attr_name in enumerate(CELEBA_ATTRIBUTES):
        acc = accuracies[i].item()
        rate = pred_rates[i].item()

        # Flag suspicious "Lazy" predictions
        # (High accuracy but rate is 0.0 or 1.0 means it's just guessing the majority)
        flag = ""
        if (rate == 0.0 or rate == 1.0) and acc > 0.8:
            flag = "⚠️ LAZY"
            suspicious_count += 1
        elif 0.4 < acc < 0.6:
            flag = "🎲 RANDOM"
        elif acc > 0.85:
            flag = "✅ GOOD"

        print(f"{attr_name:<25} | {acc:.2f}   | {rate:.2f} {flag}")

    print("-" * 60)
    print(f"Lazy Attributes detected: {suspicious_count}/40")


if __name__ == "__main__":
    inspect_latest_checkpoint()