from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import my_model  # noqa: E402

from backend.config import settings  # noqa: E402

CATEGORY_MAP = {"1": "animal", "2": "person", "3": "vehicle"}

LABEL_VI = {
    "herbivore": "Động vật ăn cỏ",
    "carnivore": "Động vật ăn thịt",
    "omnivore": "Động vật tạp ăn",
}

TRAITS = {
    "herbivore": ["Ăn cỏ", "Dạ dày 4 ngăn", "Nhai lại", "Móng guốc"],
    "carnivore": ["Nanh sắc", "Móng vuốt", "Mắt hướng trước", "Săn theo bầy"],
    "omnivore": ["Ăn cả thịt & quả", "Thích nghi cao", "Ăn mọi thứ"],
}

DIET_ENUM = {
    "herbivore": "Herbivore",
    "carnivore": "Carnivore",
    "omnivore": "Omnivore",
}


@dataclass
class DetectionBoxResult:
    x: float
    y: float
    width: float
    height: float
    diet_type: str
    label: str
    confidence: float


@dataclass
class ClassifyResult:
    label: str
    diet_type: str
    confidence: float
    herbivore_pct: float
    carnivore_pct: float
    omnivore_pct: float
    traits: list[str]
    boxes: list[DetectionBoxResult]


class AnimalInferenceEngine:
    """MegaDetector + MobileNetV2 diet classifier."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._detector = None
        self._megadetector_ok: bool | None = None
        self._classifier = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._class_to_idx = {name: i for i, name in enumerate(settings.class_names)}

    def _ensure_loaded(self) -> None:
        if self._classifier is not None and self._megadetector_ok is not None:
            return

        with self._lock:
            if self._classifier is not None and self._megadetector_ok is not None:
                return

            if self._megadetector_ok is None:
                try:
                    try:
                        from megadetector.detection.run_detector import load_detector
                    except ModuleNotFoundError:
                        from detection.run_detector import load_detector

                    original_torch_load = torch.load

                    def compatible_torch_load(*args, **kwargs):
                        kwargs.setdefault("weights_only", False)
                        return original_torch_load(*args, **kwargs)

                    torch.load = compatible_torch_load
                    try:
                        self._detector = load_detector(settings.detector_name.upper())
                    finally:
                        torch.load = original_torch_load
                    self._megadetector_ok = True
                except Exception:
                    self._detector = None
                    self._megadetector_ok = False

            ckpt_path = settings.resolved_checkpoint()
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy checkpoint: {ckpt_path}. "
                    "Đặt best_animal_classifier.pt vào train_model/ hoặc thư mục gốc repo."
                )

            self._classifier = my_model(num_classes=len(settings.class_names)).to(
                self._device
            )
            ckpt = torch.load(
                ckpt_path, map_location=self._device, weights_only=False
            )
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self._classifier.load_state_dict(state)
            self._classifier.eval()

    def _classify_crop(self, crop_bgr: np.ndarray) -> tuple[str, float, dict[str, float]]:
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._classifier(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        cls_name = settings.class_names[pred_idx]
        cls_conf = float(probs[pred_idx])
        prob_map = {
            name: float(probs[self._class_to_idx[name]])
            for name in settings.class_names
        }
        return cls_name, cls_conf, prob_map

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _detect_with_megadetector(self, img_rgb: np.ndarray, image_id: str) -> list[tuple[float, float, float, float]]:
        if not self._megadetector_ok or self._detector is None:
            return []

        det_result = self._detector.generate_detections_one_image(
            img_rgb, image_id=image_id
        )
        boxes: list[tuple[float, float, float, float]] = []
        for det in det_result.get("detections", []):
            category = det["category"]
            det_conf = float(det["conf"])
            bbox = det["bbox"]

            if category != "1" or det_conf < settings.det_conf_threshold:
                continue

            x_norm, y_norm, bw, bh = bbox
            boxes.append((float(x_norm), float(y_norm), float(bw), float(bh)))

        return boxes

    def _detect_with_opencv_fallback(self, img_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Best-effort multi-object fallback when MegaDetector is not installed."""
        h, w = img_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return []

        max_dim = 720
        scale = min(1.0, max_dim / max(h, w))
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1 else img_bgr.copy()
        sh, sw = small.shape[:2]
        img_area = float(sw * sh)
        candidates: list[tuple[int, int, int, int, float]] = []

        blur = cv2.bilateralFilter(small, 7, 50, 50)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        data = lab.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

        try:
            _, labels, _ = cv2.kmeans(data, 6, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
            labels = labels.reshape((sh, sw))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

            for k in range(6):
                mask = (labels == k).astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

                for i in range(1, count):
                    x, y, bw, bh, area = stats[i]
                    self._add_candidate(candidates, sw, sh, img_area, x, y, bw, bh, float(area))
        except cv2.error:
            pass

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 11))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            self._add_candidate(candidates, sw, sh, img_area, x, y, bw, bh, float(bw * bh))

        merged = self._merge_candidates(candidates)
        if len(merged) > 1:
            merged = [
                box for box in merged
                if ((box[2] - box[0]) * (box[3] - box[1])) < img_area * 0.70
            ] or merged

        selected: list[tuple[int, int, int, int, float]] = []
        for box in sorted(merged, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
            rect = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            if all(self._iou(rect, (float(s[0]), float(s[1]), float(s[2]), float(s[3]))) < 0.35 for s in selected):
                selected.append(box)
            if len(selected) >= 8:
                break

        if len(selected) <= 1:
            return []

        norm_boxes: list[tuple[float, float, float, float]] = []
        for x1, y1, x2, y2, _ in selected:
            x = max(0.0, x1 / sw)
            y = max(0.0, y1 / sh)
            bw = min(1.0 - x, (x2 - x1) / sw)
            bh = min(1.0 - y, (y2 - y1) / sh)
            norm_boxes.append((round(x, 4), round(y, 4), round(bw, 4), round(bh, 4)))

        return norm_boxes

    @staticmethod
    def _add_candidate(
        candidates: list[tuple[int, int, int, int, float]],
        sw: int,
        sh: int,
        img_area: float,
        x: int,
        y: int,
        bw: int,
        bh: int,
        score: float,
    ) -> None:
        area = float(bw * bh)
        if area < img_area * 0.015 or area > img_area * 0.75:
            return
        if bw < sw * 0.06 or bh < sh * 0.08:
            return
        aspect = bw / max(1, bh)
        if aspect < 0.20 or aspect > 5.0:
            return
        candidates.append((x, y, x + bw, y + bh, score))

    def _merge_candidates(
        self, candidates: list[tuple[int, int, int, int, float]]
    ) -> list[tuple[int, int, int, int, float]]:
        boxes = candidates[:]
        changed = True
        while changed:
            changed = False
            merged: list[tuple[int, int, int, int, float]] = []
            used = [False] * len(boxes)

            for i, a in enumerate(boxes):
                if used[i]:
                    continue

                x1, y1, x2, y2, score = a
                used[i] = True
                for j, b in enumerate(boxes):
                    if used[j]:
                        continue

                    rect_a = (float(x1), float(y1), float(x2), float(y2))
                    rect_b = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                    if self._iou(rect_a, rect_b) > 0.15:
                        x1 = min(x1, b[0])
                        y1 = min(y1, b[1])
                        x2 = max(x2, b[2])
                        y2 = max(y2, b[3])
                        score += b[4]
                        used[j] = True
                        changed = True

                merged.append((x1, y1, x2, y2, score))

            boxes = merged

        return boxes

    def classify_image(self, image_path: str | Path) -> ClassifyResult:
        self._ensure_loaded()

        path = Path(image_path)
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"Không đọc được ảnh: {path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        detections = self._detect_with_megadetector(img_rgb, path.stem)
        if not detections:
            detections = self._detect_with_opencv_fallback(img_bgr)

        boxes: list[DetectionBoxResult] = []
        prob_accum = {name: 0.0 for name in settings.class_names}
        count = 0

        for x_norm, y_norm, bw, bh in detections:
            x1 = max(0, int(x_norm * w))
            y1 = max(0, int(y_norm * h))
            x2 = min(w, int((x_norm + bw) * w))
            y2 = min(h, int((y_norm + bh) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img_bgr[y1:y2, x1:x2]
            cls_name, cls_conf, prob_map = self._classify_crop(crop)

            for name in settings.class_names:
                prob_accum[name] += prob_map[name]
            count += 1

            boxes.append(
                DetectionBoxResult(
                    x=round(x_norm, 4),
                    y=round(y_norm, 4),
                    width=round(bw, 4),
                    height=round(bh, 4),
                    diet_type=DIET_ENUM[cls_name],
                    label=cls_name,
                    confidence=round(cls_conf, 4),
                )
            )

        if not boxes:
            cls_name, cls_conf, prob_map = self._classify_crop(img_bgr)
            for name in settings.class_names:
                prob_accum[name] = prob_map[name]
            count = 1
            boxes.append(
                DetectionBoxResult(
                    x=0.0,
                    y=0.0,
                    width=1.0,
                    height=1.0,
                    diet_type=DIET_ENUM[cls_name],
                    label=cls_name,
                    confidence=round(cls_conf, 4),
                )
            )

        if count > 0:
            avg_probs = {k: prob_accum[k] / count for k in settings.class_names}
        else:
            avg_probs = {k: 0.0 for k in settings.class_names}

        main = max(boxes, key=lambda b: b.confidence)
        main_label = main.label

        return ClassifyResult(
            label=LABEL_VI[main_label],
            diet_type=main.diet_type,
            confidence=main.confidence,
            herbivore_pct=round(avg_probs["herbivore"], 4),
            carnivore_pct=round(avg_probs["carnivore"], 4),
            omnivore_pct=round(avg_probs["omnivore"], 4),
            traits=TRAITS[main_label],
            boxes=boxes,
        )


_engine: AnimalInferenceEngine | None = None
_engine_lock = Lock()


def get_engine() -> AnimalInferenceEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = AnimalInferenceEngine()
    return _engine
