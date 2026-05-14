import os
import shutil

# 1. Khai báo đường dẫn
source_dir = 'animals'  # Thư mục gốc hiện tại của bạn
target_dir = 'diet_dataset'  # Thư mục mới sẽ được tạo ra

# 2. Định nghĩa từ điển map loài vật (bạn có thể thêm bớt tùy theo các folder con bạn đang có)
diet_mapping = {
    'herbivore': [ 'elephant', 'cow', 'horse','sheep'],
    'carnivore': ['dog', 'cat',]
}

splits = ['train', 'test']

# 3. Khởi tạo sẵn cấu trúc thư mục đích
for split in splits:
    for diet_class in diet_mapping.keys():
        os.makedirs(os.path.join(target_dir, split, diet_class), exist_ok=True)

# 4. Quá trình xử lý (Đọc cốt lõi code để hiểu 'động cơ' bên dưới)
for split in splits:
    split_path = os.path.join(source_dir, split)  # vd: animal/train

    if not os.path.exists(split_path):
        print(f"Không tìm thấy thư mục {split_path}, bỏ qua...")
        continue

    print(f"--- Đang xử lý tập {split.upper()} ---")

    for diet_class, species_list in diet_mapping.items():
        for species in species_list:
            species_path = os.path.join(split_path, species)  # vd: animal/train/lion

            # Nếu loài này tồn tại trong tập dữ liệu
            if os.path.exists(species_path):
                print(f"  Gộp {species} vào {diet_class}...")

                for img_file in os.listdir(species_path):
                    src_file = os.path.join(species_path, img_file)

                    # Đổi tên file để tránh trùng nếu các loài khác nhau có ảnh trùng tên (vd: image_1.jpg)
                    dst_file = os.path.join(target_dir, split, diet_class, f"{species}_{img_file}")

                    # Dùng shutil.copy để copy sang thư mục mới (bạn có thể đổi thành shutil.move nếu muốn cut luôn)
                    shutil.copy(src_file, dst_file)

print("\nĐã xử lý xong! Dữ liệu mới nằm ở folder:", target_dir)