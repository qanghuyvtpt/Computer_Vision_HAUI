import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from model import my_model
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

INPUT_PATH      = r"C:\Users\Admin\Desktop\PythonProject\Xu_ly_anh_so_HAUI\data\Animal_dataset\animal_grouped"  # ảnh/video, hoặc thư mục chứa carnivore/herbivore/omnivore
CHECKPOINT_PATH = r"./train_model/best_animal_classifier.pt"
CLS_CONF        = 0.0

CLASS_NAMES = [
    'antelope', 'bobcat', 'buffalo', 'cat', 'chimpanzee',
    'cow', 'deer', 'dog', 'dolphin', 'elephant',
    'fox', 'giant panda', 'giraffe', 'gorilla', 'grizzly bear',
    'hamster', 'hippopotamus', 'horse', 'leopard', 'lion',
    'moose', 'otter', 'ox', 'pig', 'polar bear',
    'rabbit', 'raccoon', 'rat', 'rhinoceros', 'seal',
    'sheep', 'squirrel', 'tiger', 'whale', 'wolf', 'zebra'
]

DIET_MAP = {
    'antelope':     'herbivore',
    'bobcat':       'carnivore',
    'buffalo':      'herbivore',
    'cat':          'carnivore',
    'chimpanzee':   'omnivore',
    'cow':          'herbivore',
    'deer':         'herbivore',
    'dog':          'omnivore',
    'dolphin':      'carnivore',
    'elephant':     'herbivore',
    'fox':          'omnivore',
    'giant panda':  'herbivore',
    'giraffe':      'herbivore',
    'gorilla':      'herbivore',
    'grizzly bear': 'omnivore',
    'hamster':      'omnivore',
    'hippopotamus': 'herbivore',
    'horse':        'herbivore',
    'leopard':      'carnivore',
    'lion':         'carnivore',
    'moose':        'herbivore',
    'otter':        'carnivore',
    'ox':           'herbivore',
    'pig':          'omnivore',
    'polar bear':   'carnivore',
    'rabbit':       'herbivore',
    'raccoon':      'omnivore',
    'rat':          'omnivore',
    'rhinoceros':   'herbivore',
    'seal':         'carnivore',
    'sheep':        'herbivore',
    'squirrel':     'omnivore',
    'tiger':        'carnivore',
    'whale':        'carnivore',
    'wolf':         'carnivore',
    'zebra':        'herbivore',
}

DIET_CLASSES = ['carnivore', 'herbivore', 'omnivore']
DIET_TO_IDX  = {d: i for i, d in enumerate(DIET_CLASSES)}

DIET_INDICES = {'carnivore': [], 'herbivore': [], 'omnivore': []}
for idx, name in enumerate(CLASS_NAMES):
    DIET_INDICES[DIET_MAP[name]].append(idx)

DIET_COLORS = {
    'carnivore': (0,  60, 220),
    'herbivore': (0, 200,  80),
    'omnivore':  (20, 180, 220),
}

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

# ── Khởi tạo model ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = my_model(num_classes=len(CLASS_NAMES)).to(device)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
classifier.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
classifier.eval()

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std =[0.229, 0.224, 0.225]),
])


# ── Hàm dự đoán diet cho 1 ảnh (toàn bộ ảnh, không crop) ───────────────────────
def predict_diet(img_bgr: np.ndarray):
    """Đưa cả ảnh vào classifier, trả về (best_diet, best_score, diet_scores)."""
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor  = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)[0].cpu()

    diet_scores = {
        diet: float(probs[indices].sum())
        for diet, indices in DIET_INDICES.items()
    }
    best_diet  = max(diet_scores, key=diet_scores.get)
    best_score = diet_scores[best_diet]

    return best_diet, best_score, diet_scores


# ── Vẽ kết quả lên ảnh (label ở góc trên-trái, không có bounding box) ─────────
def annotate_image(img_bgr: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    best_diet, best_score, _ = predict_diet(img_bgr)
    color = DIET_COLORS[best_diet]

    label = f"{best_diet}: {best_score:.2f}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

    cv2.rectangle(out, (10, 10), (10 + tw + 12, 10 + th + 16), color, -1)
    cv2.putText(out, label, (16, 10 + th + 6), font, scale, (255, 255, 255), thick)

    return out


# ── Xử lý ảnh ────────────────────────────────────────────────────────────────
def run_image(path: str) -> None:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")

    result = annotate_image(img_bgr)
    cv2.imshow("Animal Diet Classification", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Đánh giá thư mục carnivore/herbivore/omnivore ──────────────────────────────
def evaluate_folder(data_root: str) -> None:
    """
    data_root chứa 3 thư mục con: carnivore/, herbivore/, omnivore/.
    Mỗi thư mục chứa trực tiếp các ảnh động vật (carnivore/lion.jpg,
    carnivore/tiger.jpg, ...). Toàn bộ ảnh được đưa thẳng vào classifier,
    không qua detection/crop.
    """
    data_root = Path(data_root)

    folder_classes = sorted(
        p.name for p in data_root.iterdir() if p.is_dir()
    )
    if set(folder_classes) != set(DIET_CLASSES):
        raise ValueError(
            f"Thu muc goc phai chua dung 3 thu muc con: {DIET_CLASSES}, "
            f"nhung tim thay: {folder_classes}"
        )

    all_preds  = []
    all_labels = []
    skipped    = 0

    for diet_name in DIET_CLASSES:
        folder = data_root / diet_name
        image_paths = sorted(
            p for p in folder.glob('*')
            if p.suffix.lower() in IMAGE_EXTS
        )

        for img_path in tqdm(image_paths, desc=f"Evaluating {diet_name}"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                skipped += 1
                continue

            pred_diet, _, _ = predict_diet(img_bgr)

            all_preds.append(DIET_TO_IDX[pred_diet])
            all_labels.append(DIET_TO_IDX[diet_name])

    total = len(all_labels)
    if total == 0:
        print("Khong co anh nao du doan duoc.")
        return

    correct  = sum(p == t for p, t in zip(all_preds, all_labels))
    accuracy = correct / total

    print(f"\nTotal images evaluated: {total}  (skipped: {skipped})")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report (theo che do an):")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(DIET_CLASSES))),
            target_names=DIET_CLASSES,
            digits=4,
        )
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(DIET_CLASSES))))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(DIET_CLASSES)
    print(cm)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        evaluate_folder(INPUT_PATH)
    else:
        run_image(INPUT_PATH)