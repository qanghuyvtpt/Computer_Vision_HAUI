import cv2
from megadetector.detection.run_detector import load_detector

CATEGORY_MAP = {'1': 'animal', '2': 'person', '3': 'vehicle'}

# Đọc ảnh
img_bgr = cv2.imread(
    r"C:\Users\Admin\Desktop\PythonProject\Xu_ly_anh_so_HAUI\image_test\Screenshot_2.png"
)

# MegaDetector cần RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Load detector (chữ thường "v5a")
detector = load_detector("MDv5a")

# Detect — truyền ảnh RGB
result = detector.generate_detections_one_image(img_rgb, image_id="test")

print("Raw result:", result)
print(f"\nTìm thấy {len(result['detections'])} detections\n")

h, w = img_bgr.shape[:2]
crops = []  # lưu các crop để đưa vào classifier

for i, det in enumerate(result['detections']):
    category = det['category']
    conf     = det['conf']
    bbox     = det['bbox']  # [x, y, width, height] normalized

    print(f"[{i}] category={CATEGORY_MAP.get(category, '?')} conf={conf:.3f} bbox={bbox}")

    # Chỉ lấy động vật, bỏ người/xe
    if category != '1':
        continue

    # Bỏ detection quá thấp
    if conf < 0.5:
        continue

    # Convert bbox về pixel
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int((bbox[0] + bbox[2]) * w)
    y2 = int((bbox[1] + bbox[3]) * h)

    # Crop → đưa vào classifier của nhóm
    crop = img_bgr[y1:y2, x1:x2]
    crops.append(crop)

    # Vẽ bbox
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 80), 2)

    label = f"animal {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 200, 80), -1)
    cv2.putText(img_bgr, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# Hiển thị
cv2.imshow("MegaDetector Result", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()