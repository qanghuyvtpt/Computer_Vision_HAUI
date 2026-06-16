import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from megadetector.detection.run_detector import load_detector
from model import my_model
import os
import tempfile


INPUT_PATH      = r"C:\Users\Admin\Desktop\Screenshot_1.png"   # ảnh (.png/.jpg) hoặc video (.mp4)
CHECKPOINT_PATH = r"./train_model/best_animal_classifier.pt"
DET_CONF        = 0.5
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

DIET_INDICES = {'carnivore': [], 'herbivore': [], 'omnivore': []}
for idx, name in enumerate(CLASS_NAMES):
    DIET_INDICES[DIET_MAP[name]].append(idx)

DIET_COLORS = {
    'carnivore': (0,  60, 220),
    'herbivore': (0, 200,  80),
    'omnivore':  (20, 180, 220),
}

CATEGORY_MAP = {'1': 'animal', '2': 'person', '3': 'vehicle'}

# ── Khởi tạo model ────────────────────────────────────────────────────────────
detector = load_detector("MDv5a")
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ── Hàm xử lý 1 frame ────────────────────────────────────────────────────────
def process_frame(img_bgr: np.ndarray) -> np.ndarray:
    """Chạy MegaDetector + classifier trên 1 frame, trả về frame đã vẽ."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]
    out     = img_bgr.copy()

    result     = detector.generate_detections_one_image(img_rgb, image_id="frame")
    detections = result.get("detections", [])

    for det in detections:
        category = det['category']
        conf     = det['conf']
        bbox     = det['bbox']

        if category != '1' or conf < DET_CONF:
            continue

        x1 = max(0, int(bbox[0] * w))
        y1 = max(0, int(bbox[1] * h))
        x2 = min(w, int((bbox[0] + bbox[2]) * w))
        y2 = min(h, int((bbox[1] + bbox[3]) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop    = img_bgr[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor  = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(classifier(tensor), dim=1)[0].cpu()

        diet_scores = {
            diet: float(probs[indices].sum())
            for diet, indices in DIET_INDICES.items()
        }

        best_diet  = max(diet_scores, key=diet_scores.get)
        best_score = diet_scores[best_diet]
        color      = DIET_COLORS[best_diet]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{best_diet}: {best_score:.2f}"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        cv2.rectangle(out,
                      (x1, y1 - th - 8),
                      (x1 + tw + 6, y1),
                      color, -1)
        cv2.putText(out, label,
                    (x1 + 3, y1 - 5),
                    font, scale, (255, 255, 255), thick)

    return out


# ── Xử lý ảnh ────────────────────────────────────────────────────────────────
def run_image(path: str) -> None:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")

    result = process_frame(img_bgr)
    cv2.imshow("Animal Detection + Classification", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Xử lý video: render toàn bộ → lưu file tạm → phát lại ───────────────────
def run_video(path: str) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # File tạm để lưu video đã xử lý
    tmp_path = tempfile.mktemp(suffix="_processed.mp4")
    writer   = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    print(f"\nĐang xử lý video: {total} frames  ({width}×{height} @ {fps:.1f} fps)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)
        writer.write(processed)

        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == total:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  [{frame_idx}/{total}]  {pct:.1f}%", end="\r")

    cap.release()
    writer.release()
    print(f"\nXử lý xong → {tmp_path}")

    # ── Phát lại video đã xử lý ──────────────────────────────────────────────
    print("Phát lại video đã xử lý (nhấn Q để thoát)...")
    playback = cv2.VideoCapture(tmp_path)
    delay    = max(1, int(1000 / fps))   # ms mỗi frame

    while True:
        ret, frame = playback.read()
        if not ret:
            break
        cv2.imshow("Animal Detection + Classification", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    playback.release()
    cv2.destroyAllWindows()

    # Xoá file tạm
    try:
        os.remove(tmp_path)
    except OSError:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ext = os.path.splitext(INPUT_PATH)[1].lower()
    if ext == ".mp4":
        run_video(INPUT_PATH)
    else:
        run_image(INPUT_PATH)