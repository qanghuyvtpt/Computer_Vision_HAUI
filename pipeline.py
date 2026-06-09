import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from megadetector.detection.run_detector import load_detector
from model import my_model


IMAGE_PATH      = r"./image_test/Screenshot_3.png"
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

# Nhóm index theo chế độ ăn (tính 1 lần lúc khởi động)
DIET_INDICES = {'carnivore': [], 'herbivore': [], 'omnivore': []}
for idx, name in enumerate(CLASS_NAMES):
    DIET_INDICES[DIET_MAP[name]].append(idx)

DIET_COLORS = {
    'carnivore': (0,  60, 220),
    'herbivore': (0, 200,  80),
    'omnivore':  (20, 180, 220),
}

CATEGORY_MAP = {'1': 'animal', '2': 'person', '3': 'vehicle'}

detector = load_detector("MDv5a")
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

img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w    = img_bgr.shape[:2]

result     = detector.generate_detections_one_image(img_rgb, image_id="test")
detections = result.get("detections", [])
print(f"\nMegaDetector: {len(detections)} objects found\n")

for i, det in enumerate(detections):
    category = det['category']
    conf     = det['conf']
    bbox     = det['bbox']

    print(f"[{i}] category={CATEGORY_MAP.get(category,'?')}  det_conf={conf:.3f}")

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

    # Tổng xác suất theo từng nhóm chế độ ăn
    diet_scores = {
        diet: float(probs[indices].sum())
        for diet, indices in DIET_INDICES.items()
    }

    best_diet  = max(diet_scores, key=diet_scores.get)
    best_score = diet_scores[best_diet]
    color      = DIET_COLORS[best_diet]

    print(f"     → carnivore={diet_scores['carnivore']:.3f}  "
          f"herbivore={diet_scores['herbivore']:.3f}  "
          f"omnivore={diet_scores['omnivore']:.3f}  "
          f"→ {best_diet}")

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    label = f"{best_diet}: {best_score:.2f}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

    cv2.rectangle(img_bgr,
                  (x1, y1 - th - 8),
                  (x1 + tw + 6, y1),
                  color, -1)
    cv2.putText(img_bgr, label,
                (x1 + 3, y1 - 5),
                font, scale, (255, 255, 255), thick)

cv2.imshow("Animal Detection + Classification", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()