# Huong dan cai dat va chay web

## 1. Kien truc du an

Du an gom 2 service chay song song:

- Frontend web: ASP.NET Core MVC, cong `5266`
- Backend AI: Python FastAPI, cong `8000`

Luong xu ly:

```text
Browser -> Frontend http://127.0.0.1:5266 -> Backend AI http://127.0.0.1:8000
```

Backend dung MegaDetector de detect nhieu con vat trong anh, sau do cat tung vung anh va dung model MobileNetV2 de phan loai:

- Herbivore: dong vat an co
- Carnivore: dong vat an thit
- Omnivore: dong vat tap an

## 2. Phan mem can cai

Bat buoc:

- Git
- .NET SDK 8
- Python 3.10 tro len
- Chrome hoac Edge

Kiem tra:

```powershell
dotnet --version
python --version
git --version
```

Ket qua mong doi:

- `dotnet --version`: 8.0.x
- `python --version`: 3.10+

## 3. File model can co

Dat file model da train vao:

```text
train_model/best_animal_classifier.pt
```

Khong nen push file `.pt` len Git vi file model lon. Hay chia se file model qua Google Drive/OneDrive va moi thanh vien tu copy vao dung thu muc tren.

## 4. Cai dat lan dau

Mo PowerShell tai thu muc goc repo:

```powershell
cd "Computer_Vision_HAUI"
```

Tao moi truong Python:

```powershell
python -m venv frontend\.venv
frontend\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

Lan dau cai `torch` va `megadetector` co the mat vai phut.

Khoi phuc frontend:

```powershell
cd frontend
dotnet restore
dotnet build
cd ..
```

## 5. Cach chay nhanh

Tu thu muc goc repo:

```powershell
.\start_app.ps1
```

Script se mo:

- Backend AI: `http://127.0.0.1:8000`
- Frontend web: `http://127.0.0.1:5266`

Mo trinh duyet:

```text
http://127.0.0.1:5266
```

Neu PowerShell chan script:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_app.ps1
```

## 6. Cach chay thu cong

Mo 2 cua so PowerShell.

Terminal 1 - Backend AI:

```powershell
cd "Computer_Vision_HAUI"
frontend\.venv\Scripts\uvicorn.exe backend.app:app --host 127.0.0.1 --port 8000
```

Kiem tra backend:

```text
http://127.0.0.1:8000/health
```

Can thay:

```text
status: ok
checkpointExists: True
megadetector: ready
```

Terminal 2 - Frontend web:

```powershell
cd "Computer_Vision_HAUI\frontend"
dotnet run --urls http://127.0.0.1:5266
```

Mo web:

```text
http://127.0.0.1:5266
```

## 7. Cau hinh AI hoac Mock

File cau hinh:

```text
frontend/appsettings.json
```

Dung AI that:

```json
"ModelApi": {
  "BaseUrl": "http://localhost:8000",
  "UseMock": false
}
```

Chi test giao dien, khong can backend:

```json
"UseMock": true
```

## 8. Cach test

1. Chay backend va frontend.
2. Mo `http://127.0.0.1:5266`.
3. Chon anh co mot hoac nhieu con vat.
4. Bam `Phan Tich Ngay`.
5. Kiem tra ket qua:
   - Anh co bounding box.
   - Neu anh co nhieu con vat, web hien nhieu box va nhieu dong ket qua.
   - Moi dong co loai an co/an thit/tap an va phan tram tin cay.
   - Trang lich su luu lai lan phan tich.

## 9. Loi thuong gap

Khong vao duoc web:

- Mo dung dia chi `http://127.0.0.1:5266`.
- Cong `8000` chi la API backend, khong phai giao dien web.

Khong ket noi duoc backend AI:

- Backend chua chay hoac sai cong.
- Chay lai lenh uvicorn o Terminal 1.

`checkpointExists: False`:

- Thieu file `train_model/best_animal_classifier.pt`.

`megadetector: unavailable`:

- Chua cai du dependency.
- Chay lai:

```powershell
frontend\.venv\Scripts\activate
pip install -r backend\requirements.txt
```

Lan dau phan tich cham:

- Binh thuong, vi MegaDetector/PyTorch can load model.
- Cac lan sau se nhanh hon.

## 10. Checklist truoc khi push cho team

- Khong push `frontend/.venv/`.
- Khong push `frontend/bin/`, `frontend/obj/`.
- Khong push `frontend/wwwroot/uploads/`.
- Khong push file `.pt`.
- Dam bao co `backend/requirements.txt`.
- Dam bao co `start_app.ps1`.
- Dam bao file huong dan `.docx` va `.md` nam trong `docs/`.
