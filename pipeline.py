import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from megadetector.detection.run_detector import load_detector
from model import my_model


IMAGE_PATH      = r"./image_test/Screenshot_6.png"
CHECKPOINT_PATH = r"./train_model/best_animal_classifier.pt"
DET_CONF        = 0.5
CLS_CONF        = 0.0

CLASS_NAMES  = ['carnivore', 'herbivore', 'omnivore']
CLASS_COLORS = {
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
        probs    = torch.softmax(classifier(tensor), dim=1)[0].cpu()
        pred_idx = int(torch.argmax(probs))
        cls_conf = float(probs[pred_idx])
        cls_name = CLASS_NAMES[pred_idx]

    print(f"     → {cls_name}  cls_conf={cls_conf:.3f}")

    color = CLASS_COLORS[cls_name]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    label = f"{cls_name}: {cls_conf:.2f}"
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