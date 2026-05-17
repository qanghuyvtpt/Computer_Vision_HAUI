import argparse
import os
from pathlib import Path
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from model import my_model


CLASS_NAMES = ['carnivore', 'herbivore', 'omnivore']

def get_args():
    parser = argparse.ArgumentParser("test animal classifier")
    parser.add_argument(
        "--checkpoint",
        "-c",
        default="./train_model/best_animal_classifier.pt",
        help="Duong dan file .pt da train",
    )
    parser.add_argument(
        "--image",
        "-i",
        default=None,
        help="Duong dan anh can du doan",
    )
    parser.add_argument(
        "--data_root",
        "-d",
        default=r"C:\Users\Admin\Desktop\ki6\xu_ly_anh_so\Animal_dataset\val",
        help="Thu muc test/val co cac folder con herbivore, carnivore, omnivore",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def get_transform():
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(checkpoint_path, device):
    model = my_model(num_classes=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model



def evaluate_folder(model, data_root, transform, batch_size, num_workers, device):
    dataset = ImageFolder(root=data_root, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = correct / total

    print(f"\nTotal images: {total}")
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=dataset.classes,
            digits=4,
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(cm)


def main():
    args = get_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA khong kha dung. Hay chay voi --device cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {args.checkpoint}")

    if args.image is None and args.data_root is None:
        raise ValueError("Can truyen --image de test 1 anh hoac --data_root de test ca folder")

    transform = get_transform()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    if args.data_root:
        data_root = Path(args.data_root)
        if not data_root.is_dir():
            raise FileNotFoundError(f"Khong tim thay thu muc: {args.data_root}")
        evaluate_folder(
            model=model,
            data_root=str(data_root),
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )


if __name__ == "__main__":
    main()
