# 📋 Báo cáo Tiền xử lý Dữ liệu — AWA2 Diet Classifier

> **AWA2 — Phân loại Chế độ Ăn Động vật**
> _Herbivore · Carnivore · Omnivore_

| | |
|---|---|
| 📅 **Ngày thực hiện** | 14/05/2026 |
| 🛠️ **Công cụ** | Python · Pillow · PyTorch |
| 📦 **Dataset** | [AWA2 — Kaggle](https://www.kaggle.com/datasets/radimkzl/awa2-dataset) |

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Phân loại nhãn](#2-phân-loại-nhãn)
3. [Quy trình tiền xử lý](#3-quy-trình-tiền-xử-lý)
4. [Kết quả sau tiền xử lý](#4-kết-quả-sau-tiền-xử-lý)
5. [Xử lý mất cân bằng dữ liệu](#5-xử-lý-mất-cân-bằng-dữ-liệu)
6. [Công cụ và thư viện](#6-công-cụ-và-thư-viện)
7. [Hướng dẫn sử dụng nhanh](#7-hướng-dẫn-sử-dụng-nhanh)
8. [Kết luận](#8-kết-luận)

---

## 1. Tổng quan

Tài liệu này mô tả quy trình tiền xử lý bộ dữ liệu **AWA2 (Animals with Attributes 2)** nhằm chuẩn bị dữ liệu cho bài toán phân loại chế độ ăn của động vật gồm 3 nhãn: động vật ăn cỏ (Herbivore), động vật ăn thịt (Carnivore) và động vật ăn tạp (Omnivore).

### 1.1 Thông tin bộ dữ liệu gốc

| Thuộc tính | Thông tin |
|---|---|
| Tên dataset | Animals with Attributes 2 (AWA2) |
| Nguồn | Kaggle — radimkzl/awa2-dataset |
| Tổng số class gốc | 50 class động vật |
| Tổng số ảnh gốc | ~37,000+ ảnh JPEG |
| Cấu trúc gốc | `JPEGImages/<tên_loài>/` |
| File nhãn gốc | `predicate-matrix-binary.txt` |

---

## 2. Phân loại nhãn

Từ 50 class gốc của AWA2, nhóm thực hiện gán nhãn thủ công dựa trên tài liệu động vật học (Wilson & Reeder, ARKive/IUCN) kết hợp với file `predicate-matrix-binary.txt` của dataset. Kết quả phân thành 3 nhãn như sau:

### 🌿 2.1 Herbivore — Động vật ăn cỏ (21 class)

- antelope, horse, giraffe, zebra, cow, sheep, rabbit
- deer, moose, rhinoceros, hippopotamus, elephant
- giant panda, beaver, buffalo, ox, squirrel, hamster
- spider monkey, humpback whale, blue whale

### 🦁 2.2 Carnivore — Động vật ăn thịt (19 class)

- tiger, lion, leopard, cheetah, jaguar, wolf
- hyena, seal, walrus, otter, killer whale, dolphin
- weasel, mink, bobcat, lynx, persian cat, siamese cat
- polar bear, mole, bat

### 🐻 2.3 Omnivore — Động vật ăn tạp (10 class)

- grizzly bear, raccoon, pig, skunk, mouse, rat
- chimpanzee, gorilla, fox
- các giống chó: chihuahua, german shepherd, collie, dalmatian
- panda _(red panda — khác với giant panda)_

> ⚠️ **Lưu ý:** 6 class không có trong phiên bản dataset đã tải (`cheetah`, `hyena`, `jaguar`, `lynx`, `mink`, `panda`) nên được bỏ qua tự động trong quá trình xử lý.

---

## 3. Quy trình tiền xử lý

### 3.1 Xử lý ảnh

Mỗi ảnh được xử lý qua các bước sau:

1. **Kiểm tra tính hợp lệ** — loại bỏ file ảnh bị hỏng hoặc không đọc được
2. **Chuyển về định dạng RGB** — đảm bảo tất cả ảnh có 3 kênh màu
3. **Resize giữ tỉ lệ** — thu nhỏ ảnh về kích thước tối đa 224×224 pixel (không méo)
4. **Padding màu xám `(128, 128, 128)`** — lấp đầy phần còn thiếu để đạt đúng 224×224
5. **Lưu file JPEG** với chất lượng 90%

### 3.2 Chia tập dữ liệu

Dữ liệu được chia ngẫu nhiên (`seed=42`) theo tỉ lệ:

| Tập | Tỉ lệ | Mục đích |
|---|---|---|
| `train` | 70% | Huấn luyện model |
| `val` | 15% | Điều chỉnh hyperparameter |
| `test` | 15% | Đánh giá cuối cùng |

### 3.3 Augmentation _(Train only)_

Các kỹ thuật tăng cường dữ liệu chỉ áp dụng cho tập `train` trong quá trình huấn luyện:

| Kỹ thuật | Tham số | Mục đích |
|---|---|---|
| `RandomHorizontalFlip` | p = 0.5 | Lật ngang ngẫu nhiên |
| `RandomRotation` | ±15° | Xoay ảnh |
| `ColorJitter` | brightness=0.3, contrast=0.3 | Thay đổi màu sắc, độ sáng |
| `RandomPerspective` | distortion=0.2, p=0.3 | Biến dạng phối cảnh |
| `RandomResizedCrop` | scale=(0.8, 1.0) | Cắt ngẫu nhiên, giữ 80–100% |
| `Normalize` | ImageNet mean/std | Chuẩn hóa cho transfer learning |

---

## 4. Kết quả sau tiền xử lý

### 4.1 Phân bố dữ liệu

| Tập | Herbivore 🌿 | Carnivore 🦁 | Omnivore 🐻 | Tổng |
|---|---:|---:|---:|---:|
| Train | 13,442 | 6,925 | 5,735 | **26,102** |
| Validation | 2,873 | 1,479 | 1,223 | **5,575** |
| Test | 2,903 | 1,499 | 1,243 | **5,645** |
| **Tổng** | **19,218** | **9,903** | **8,201** | **37,322** |

### 4.2 Cấu trúc thư mục output

```
AWA2_processed/
├── train/
│   ├── herbivore/        # 13,442 ảnh
│   ├── carnivore/        # 6,925 ảnh
│   └── omnivore/         # 5,735 ảnh
├── val/
│   ├── herbivore/        # 2,873 ảnh
│   ├── carnivore/        # 1,479 ảnh
│   └── omnivore/         # 1,223 ảnh
└── test/
    ├── herbivore/        # 2,903 ảnh
    ├── carnivore/        # 1,499 ảnh
    └── omnivore/         # 1,243 ảnh
```

> ✅ Cấu trúc tương thích với `torchvision.datasets.ImageFolder`

---

## 5. Xử lý mất cân bằng dữ liệu

Herbivore chiếm ~51% tổng dữ liệu, gấp đôi Omnivore (~22%). Để tránh model thiên lệch về nhãn chiếm đa số, pipeline cung cấp 3 phương án:

| Phương án | Cách dùng | Ưu điểm | Nhược điểm |
|---|---|---|---|
| `WeightedRandomSampler` _(mặc định)_ | `oversample=True` trong `get_dataloaders()` | Không mất dữ liệu, oversample minority class | Một số ảnh lặp lại trong epoch |
| Class weights trong Loss | `CrossEntropyLoss(weight=class_weights)` | Đơn giản, không lặp ảnh | Gradient bị scale không đều |
| Cap ảnh tối đa | `--max 300` khi chạy script | Dataset cân bằng hoàn toàn | Mất dữ liệu của class lớn |

---

## 6. Công cụ và thư viện

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| Python | 3.9+ | Ngôn ngữ lập trình chính |
| Pillow (PIL) | mới nhất | Đọc, resize, lưu ảnh |
| tqdm | mới nhất | Hiển thị tiến trình |
| PyTorch | 2.x | Framework deep learning |
| torchvision | 0.15+ | Transforms, DataLoader |

---

## 7. Hướng dẫn sử dụng nhanh

### 7.1 Chạy tiền xử lý

```bash
python awa2_preprocess_3class.py \
  --src AwA2\data \
  --dst AWA2_processed \
  --train 0.70 \
  --val 0.15
```

### 7.2 Load dữ liệu trong training script

```python
from awa2_dataset import get_dataloaders

loaders = get_dataloaders(
    data_root="AWA2_processed",
    batch_size=32,
    oversample=True   # xử lý mất cân bằng
)

# Sử dụng
for imgs, labels in loaders["train"]:
    # imgs shape: [32, 3, 224, 224]
    # labels: 0=herbivore, 1=carnivore, 2=omnivore
    ...
```

---

## 8. Kết luận

Sau quá trình tiền xử lý, bộ dữ liệu AWA2 đã được chuyển đổi thành công từ 50 class gốc sang bài toán phân loại **3 nhãn** với tổng cộng **37,322 ảnh** sạch, đã được resize chuẩn hóa về 224×224 và chia `train/val/test`. Dữ liệu sẵn sàng để đưa vào huấn luyện với các model transfer learning như **ResNet-50** hay **EfficientNet** thông qua PyTorch `DataLoader`.

---

_Báo cáo được tạo tự động bởi pipeline `awa2_preprocess_3class.py`_
