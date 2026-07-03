import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage


class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.join(root, 'Animal_dataset') #Cái tên thư mục tự đặt nhé.
        self.transform = transform

        if train:
            self.root = os.path.join(self.root, 'train')
        else:
            self.root = os.path.join(self.root, 'test')

        # 1. BẮT BUỘC PHẢI DÙNG sorted() ĐỂ ĐỒNG BỘ INDEX GIỮA TRAIN VÀ TEST
        categories = sorted(os.listdir(self.root))

        # 2. Gán vào biến self.category theo yêu cầu
        self.category = categories

        self.labels = []
        self.images = []

        for i, cat_name in enumerate(self.category):
            data_file_path = os.path.join(self.root, cat_name)

            # Đảm bảo nó là thư mục chứ không phải file linh tinh
            if os.path.isdir(data_file_path):
                for data_file in os.listdir(data_file_path):
                    self.labels.append(i)
                    image_file = os.path.join(data_file_path, data_file)
                    self.images.append(image_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])

    dataset = AnimalDataset(root=r"C:\Users\Admin\Desktop\PythonProject\Xu_ly_anh_so_HAUI\data",
                            train=False,
                            transform=transform)

    # --- ÁP DỤNG THỰC TẾ ---

    # 1. In ra xem Dataset đang quản lý những class nào
    print("Danh sách các loài trong Dataset:", dataset.category)

    image, label_index = dataset[1]  # Lấy thử 1 ảnh

    # 2. Thay vì in ra số khô khan, ta dùng self.category để "dịch" nó ra tên thật
    animal_name = dataset.category[label_index]

    print(f"Ảnh này có nhãn là số: {label_index} -> Tương ứng với con: {animal_name}")
    print("Shape của ảnh:", image.shape)

    # Nếu muốn hiển thị tên động vật lên ảnh luôn
    # (Kết hợp với kiến thức hôm trước, bạn có thể dùng OpenCV hoặc Matplotlib để vẽ cái animal_name này lên bức ảnh)