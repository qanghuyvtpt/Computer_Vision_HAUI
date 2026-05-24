from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    checkpoint_path: Path = ROOT_DIR / "train_model" / "best_animal_classifier.pt"

    def resolved_checkpoint(self) -> Path:
        """Ưu tiên train_model/, fallback file ở thư mục gốc repo."""
        if self.checkpoint_path.is_file():
            return self.checkpoint_path
        root_ckpt = ROOT_DIR / "best_animal_classifier.pt"
        return root_ckpt if root_ckpt.is_file() else self.checkpoint_path
    detector_name: str = "MDv5a"
    det_conf_threshold: float = 0.5
    host: str = "0.0.0.0"
    port: int = 8000

    class_names: tuple[str, ...] = ("carnivore", "herbivore", "omnivore")

    @property
    def checkpoint_exists(self) -> bool:
        return self.resolved_checkpoint().is_file()


settings = Settings()
