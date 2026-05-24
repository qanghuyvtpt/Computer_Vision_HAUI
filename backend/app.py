"""
FastAPI backend — phân loại chế độ ăn động vật (MegaDetector + MobileNetV2).

Chạy từ thư mục gốc repo:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.inference import ClassifyResult, DetectionBoxResult, get_engine

app = FastAPI(
    title="Animal Diet Classifier API",
    description="MegaDetector + MobileNetV2 — herbivore / carnivore / omnivore",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _box_to_dict(box: DetectionBoxResult) -> dict:
    return {
        "x": box.x,
        "y": box.y,
        "width": box.width,
        "height": box.height,
        "dietType": box.diet_type,
        "label": box.label,
        "confidence": box.confidence,
    }


def _result_to_dict(result: ClassifyResult) -> dict:
    return {
        "label": result.label,
        "dietType": result.diet_type,
        "confidence": result.confidence,
        "herbivorePct": result.herbivore_pct,
        "carnivorePct": result.carnivore_pct,
        "omnivorePct": result.omnivore_pct,
        "traits": result.traits,
        "boxes": [_box_to_dict(b) for b in result.boxes],
    }


@app.get("/health")
def health():
    from backend.inference import get_engine

    engine = get_engine()
    megadetector = "unloaded"
    try:
        engine._ensure_loaded()
        megadetector = "ready" if engine._megadetector_ok else "unavailable"
    except Exception as exc:
        megadetector = f"error: {exc}"

    return {
        "status": "ok",
        "checkpoint": str(settings.resolved_checkpoint()),
        "checkpointExists": settings.checkpoint_exists,
        "megadetector": megadetector,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }


_ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    if suffix.lower() not in _ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail="Định dạng ảnh không được hỗ trợ.")

    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/") and content_type != "application/octet-stream":
        raise HTTPException(status_code=400, detail="File phải là ảnh.")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        engine = get_engine()
        result = engine.classify_image(tmp_path)
        return _result_to_dict(result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi inference: {exc}") from exc
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
