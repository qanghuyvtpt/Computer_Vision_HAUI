"""
AWA2 Preprocessing Pipeline — 3-Class Diet Classifier
======================================================
Labels: herbivore | carnivore | omnivore

Dataset structure expected:
  AWA2/
  ├── JPEGImages/
  │   ├── antelope/
  │   ├── grizzly+bear/
  │   └── ... (50 class folders)
  ├── predicate-matrix-binary.txt
  ├── classes.txt
  └── predicates.txt

Output structure:
  AWA2_processed/
  ├── train/
  │   ├── herbivore/
  │   ├── carnivore/
  │   └── omnivore/
  ├── val/  (same)
  └── test/ (same)
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# 1. DIET LABEL MAP  (50 AWA2 classes → 3 categories)
#
#    Sources:
#    - Osherson predicate-matrix-binary.txt (attribute "meat" / "vegetation")
#    - Wilson & Reeder, Mammal Species of the World (3rd ed.)
#    - ARKive / IUCN species accounts
#
#    Decision rule for omnivores:
#      "Eats both animal protein AND plant matter in significant
#       proportions in the wild — neither dominates > 80% of diet"
# ──────────────────────────────────────────────────────────────
DIET_MAP = {
    # ── HERBIVORES (植食性) ───────────────────────────────────
    "antelope":       "herbivore",   # grasses, leaves
    "horse":          "herbivore",   # grasses, hay
    "giraffe":        "herbivore",   # browse (acacia leaves)
    "zebra":          "herbivore",   # grasses
    "cow":            "herbivore",   # grasses, forbs
    "sheep":          "herbivore",   # grasses, shrubs
    "rabbit":         "herbivore",   # grasses, vegetables
    "deer":           "herbivore",   # browse, forbs
    "moose":          "herbivore",   # aquatic plants, browse
    "rhinoceros":     "herbivore",   # grasses, browse
    "hippopotamus":   "herbivore",   # grasses (mostly nocturnal grazer)
    "elephant":       "herbivore",   # grasses, bark, fruit
    "giant+panda":    "herbivore",   # 99% bamboo
    "beaver":         "herbivore",   # bark, aquatic vegetation
    "buffalo":        "herbivore",   # grasses
    "ox":             "herbivore",   # grasses, hay
    "squirrel":       "herbivore",   # seeds, nuts, fungi (occasionally insects — minor)
    "hamster":        "herbivore",   # seeds, grains, vegetables
    "spider+monkey":  "herbivore",   # fruit, leaves, seeds
    "humpback+whale": "herbivore",   # krill (filter feeder, no plants but no predation)
    "blue+whale":     "herbivore",   # krill (same reasoning)

    # ── CARNIVORES (肉食性) ───────────────────────────────────
    "tiger":          "carnivore",   # large ungulates
    "lion":           "carnivore",   # large ungulates
    "leopard":        "carnivore",   # medium mammals
    "cheetah":        "carnivore",   # gazelles
    "jaguar":         "carnivore",   # mammals, reptiles
    "wolf":           "carnivore",   # ungulates, small mammals
    "hyena":          "carnivore",   # mammals, carrion
    "seal":           "carnivore",   # fish, squid
    "walrus":         "carnivore",   # mollusks, invertebrates
    "otter":          "carnivore",   # fish, crustaceans
    "killer+whale":   "carnivore",   # marine mammals, fish
    "dolphin":        "carnivore",   # fish, squid
    "weasel":         "carnivore",   # rodents, rabbits
    "mink":           "carnivore",   # fish, rodents, birds
    "bobcat":         "carnivore",   # rabbits, rodents
    "lynx":           "carnivore",   # hares
    "persian+cat":    "carnivore",   # domestic carnivore
    "siamese+cat":    "carnivore",   # domestic carnivore
    "polar+bear":     "carnivore",   # 90%+ marine mammals & fish
    "mole":           "carnivore",   # earthworms, insects (insectivore)
    "bat":            "carnivore",   # insects (insectivore; fruit bats → omnivore but AWA2 "bat" is microbat)
    "fox":            "omnivore",    # ← see omnivore section

    # ── OMNIVORES (杂食性) ────────────────────────────────────
    "grizzly+bear":   "omnivore",    # berries 50%+ salmon/meat in season
    "raccoon":        "omnivore",    # fruit, invertebrates, small vertebrates, human food
    "pig":            "omnivore",    # roots, fruit, invertebrates, carrion
    "skunk":          "omnivore",    # insects, berries, small vertebrates, carrion
    "mouse":          "omnivore",    # seeds, insects, small vertebrates
    "rat":            "omnivore",    # seeds, insects, fruit, carrion
    "chimpanzee":     "omnivore",    # fruit, leaves, insects, occasional hunted meat
    "gorilla":        "omnivore",    # mostly plants but eats insects & small animals
    "fox":            "omnivore",    # rabbits, rodents, fruit, berries, insects
    "chihuahua":      "omnivore",    # domestic dog — opportunistic omnivore
    "german+shepherd":"omnivore",    # domestic dog
    "collie":         "omnivore",    # domestic dog
    "dalmatian":      "omnivore",    # domestic dog
    "panda":          "omnivore",    # NOTE: "panda" in AWA2 = red panda → insects, fruit, small vertebrates
                                     #        (distinct from "giant+panda" which is herbivore)
}

# ── Derived category lists ────────────────────────────────────
HERBIVORES = sorted(k for k, v in DIET_MAP.items() if v == "herbivore")
CARNIVORES  = sorted(k for k, v in DIET_MAP.items() if v == "carnivore")
OMNIVORES   = sorted(k for k, v in DIET_MAP.items() if v == "omnivore")
ALL_CLASSES = {"herbivore": HERBIVORES, "carnivore": CARNIVORES, "omnivore": OMNIVORES}

LABELS = ["herbivore", "carnivore", "omnivore"]


# ──────────────────────────────────────────────────────────────
# 2. IMAGE HELPERS
# ──────────────────────────────────────────────────────────────

def resize_and_pad(img: Image.Image, target=(224, 224)) -> Image.Image:
    """Resize keeping aspect ratio, pad with neutral grey."""
    img.thumbnail(target, Image.LANCZOS)
    padded = Image.new("RGB", target, (128, 128, 128))
    offset = ((target[0] - img.width) // 2, (target[1] - img.height) // 2)
    padded.paste(img, offset)
    return padded


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# 3. SPLIT HELPER
# ──────────────────────────────────────────────────────────────

def split_files(files, train=0.70, val=0.15, seed=42):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


# ──────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def preprocess(
    src_root: str,
    dst_root: str,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    img_size: int = 224,
    max_per_class: int = None,
    seed: int = 42,
):
    src = Path(src_root)
    dst = Path(dst_root)

    jpeg_dir = src / "JPEGImages"
    if not jpeg_dir.exists():
        raise FileNotFoundError(f"JPEGImages/ not found in {src}")

    target = (img_size, img_size)
    splits  = ["train", "val", "test"]

    for split in splits:
        for label in LABELS:
            (dst / split / label).mkdir(parents=True, exist_ok=True)

    stats   = defaultdict(lambda: defaultdict(int))
    missing = []

    for label, classes in ALL_CLASSES.items():
        print(f"\n{'='*55}")
        print(f"  {label.upper()} ({len(classes)} classes)")
        print(f"{'='*55}")

        for cls in classes:
            cls_dir = jpeg_dir / cls
            if not cls_dir.exists():
                print(f"  [WARN] Folder not found: {cls} — skipping")
                missing.append(cls)
                continue

            image_paths = sorted([
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])

            if max_per_class and len(image_paths) > max_per_class:
                rng = random.Random(seed)
                image_paths = rng.sample(image_paths, max_per_class)

            valid_paths = [p for p in image_paths if is_valid_image(p)]
            n_corrupt = len(image_paths) - len(valid_paths)
            if n_corrupt:
                print(f"  [WARN] {cls}: {n_corrupt} corrupt file(s) skipped")

            train_imgs, val_imgs, test_imgs = split_files(
                valid_paths, train_ratio, val_ratio, seed
            )
            split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

            print(f"  {cls:<25} train:{len(train_imgs):>4}  val:{len(val_imgs):>4}  test:{len(test_imgs):>4}")

            for split_name, paths in split_map.items():
                for img_path in paths:
                    dst_path = dst / split_name / label / f"{cls}__{img_path.name}"
                    if dst_path.exists():
                        continue
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert("RGB")
                            img = resize_and_pad(img, target)
                            img.save(dst_path, "JPEG", quality=90)
                        stats[split_name][label] += 1
                    except Exception as e:
                        print(f"  [ERR] {img_path.name}: {e}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE — 3-Class Diet Classifier")
    print("=" * 60)
    grand_total = 0
    for split in splits:
        h = stats[split]["herbivore"]
        c = stats[split]["carnivore"]
        o = stats[split]["omnivore"]
        t = h + c + o
        grand_total += t
        print(f"  {split:<6}: herbivore={h:>5}  carnivore={c:>5}  omnivore={o:>5}  total={t:>5}")

    print(f"\n  Grand total : {grand_total} images")

    if missing:
        print(f"\n  [WARN] Missing folders: {missing}")

    # Class balance warning
    for split in splits:
        counts = {lb: stats[split][lb] for lb in LABELS}
        total  = sum(counts.values())
        if total == 0:
            continue
        max_ratio = max(counts.values()) / total
        if max_ratio > 0.55:
            imb_label = max(counts, key=counts.get)
            print(f"\n  [BALANCE WARNING] {split}: '{imb_label}' dominates "
                  f"({counts[imb_label]}/{total} = {max_ratio:.0%}). "
                  f"Consider --max to cap per-class images.")

    # ── Write manifest ────────────────────────────────────────
    manifest = dst / "class_manifest.txt"
    with open(manifest, "w") as f:
        f.write("# AWA2  —  3-Class Diet Classifier\n")
        f.write("# Classes: herbivore | carnivore | omnivore\n\n")
        for label, classes in ALL_CLASSES.items():
            f.write(f"{label.upper()} ({len(classes)}):\n")
            for c in classes:
                f.write(f"  {c}\n")
            f.write("\n")

    print(f"\n  Manifest → {manifest}")
    print(f"  Output   → {dst.resolve()}\n")
    return stats


# ──────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AWA2 → 3-class diet preprocessing (herbivore/carnivore/omnivore)"
    )
    parser.add_argument("--src",   default="AWA2",            help="Raw AWA2 root (chứa JPEGImages/)")
    parser.add_argument("--dst",   default="AWA2_processed",  help="Output folder")
    parser.add_argument("--size",  type=int,   default=224,   help="Image size (default 224)")
    parser.add_argument("--train", type=float, default=0.70,  help="Train ratio (default 0.70)")
    parser.add_argument("--val",   type=float, default=0.15,  help="Val ratio   (default 0.15)")
    parser.add_argument("--max",   type=int,   default=None,
                        help="Max ảnh/class — dùng để balance omnivore (ít class hơn)")
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    preprocess(
        src_root    = args.src,
        dst_root    = args.dst,
        train_ratio = args.train,
        val_ratio   = args.val,
        img_size    = args.size,
        max_per_class = args.max,
        seed        = args.seed,
    )
