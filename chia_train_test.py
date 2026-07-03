import os
import shutil
import random

# 1. Cấu hình đường dẫn và tham số
source_dir = 'dataset_goc'  # Thay bằng đường dẫn tới thư mục chứa 50 folder của bạn
target_dir = 'dataset_da_chia'  # Thư mục mới sẽ được tạo ra
train_ratio = 0.85  # Tỷ lệ 80/20

# Cố định random seed để nếu bạn chạy lại script nhiều lần, kết quả chia vẫn giống hệt nhau
random.seed(42)

print("Bắt đầu chia tập dữ liệu...")

# 2. Quét qua tất cả các folder loài vật
for species in os.listdir(source_dir):
    species_path = os.path.join(source_dir, species)

    # Bỏ qua nếu không phải là thư mục (ví dụ: các file .DS_Store, .txt lọt vào)
    if not os.path.isdir(species_path):
        continue

    # Lấy danh sách toàn bộ ảnh trong folder của loài này
    images = [f for f in os.listdir(species_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0:
        continue

    # 3. Trộn ngẫu nhiên (Shuffle) danh sách ảnh
    # Bắt buộc phải có bước này để tránh việc các ảnh chụp cùng 1 bối cảnh/thời gian bị dồn hết vào 1 tập
    random.shuffle(images)

    # 4. Tính toán điểm cắt (split index)
    split_index = int(len(images) * train_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # 5. Tạo cấu trúc thư mục đích cho loài này
    os.makedirs(os.path.join(target_dir, 'train', species), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test', species), exist_ok=True)

    # 6. Copy ảnh sang thư mục mới
    print(f"Loài '{species}': Tổng {len(images)} ảnh -> Train: {len(train_images)} | Test: {len(test_images)}")

    # Copy tập Train
    for img in train_images:
        src_file = os.path.join(species_path, img)
        dst_file = os.path.join(target_dir, 'train', species, img)
        shutil.copy(src_file, dst_file)

    # Copy tập Test
    for img in test_images:
        src_file = os.path.join(species_path, img)
        dst_file = os.path.join(target_dir, 'test', species, img)
        shutil.copy(src_file, dst_file)

print(f"\nHoàn tất! Dữ liệu đã chia nằm ở thư mục: '{target_dir}'")