# Backend AI — Animal Diet Classifier

API Python (FastAPI) phục vụ frontend ASP.NET:

1. **MegaDetector** — phát hiện vùng động vật (bounding box)
2. **MobileNetV2** (`model.py`) — phân loại 3 nhãn: `herbivore`, `carnivore`, `omnivore`

## Yêu cầu

- Python 3.10+
- File weights: `train_model/best_animal_classifier.pt` (sau khi chạy `train_model.py`)

## Cài đặt

```bash
cd Computer_Vision_HAUI
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r backend/requirements.txt
```

## Chạy API

Từ **thư mục gốc** repo:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra: [http://localhost:8000/health](http://localhost:8000/health)

## Endpoint

| Method | Path | Mô tả |
|--------|------|--------|
| GET | `/health` | Trạng thái + checkpoint |
| POST | `/api/classify` | Upload ảnh (`multipart/form-data`, field `file`) |

## Chạy cùng frontend

1. Terminal 1 — backend:
   ```bash
   uvicorn backend.app:app --port 8000
   ```
2. Terminal 2 — frontend:
   ```bash
   cd frontend
   dotnet run
   ```
3. Trong `frontend/appsettings.json`:
   ```json
   "ModelApi": {
     "BaseUrl": "http://localhost:8000",
     "UseMock": false
   }
   ```

Đặt `UseMock: true` nếu chưa có model / chưa chạy Python — frontend dùng dữ liệu giả lập.

## Biến môi trường (tùy chọn)

Tạo file `.env` ở thư mục gốc:

```env
CHECKPOINT_PATH=train_model/best_animal_classifier.pt
DET_CONF_THRESHOLD=0.5
```
