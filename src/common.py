from pathlib import Path
import yaml


# Project paths
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best.pt"
DATA_YAML = ROOT / "data" / "data.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def require_file(path: Path) -> Path:
    """Validate that a file exists."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def get_dataset_config() -> str:
    """Generate a dataset YAML with an absolute data directory."""

    with open(require_file(DATA_YAML), "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["path"] = str((ROOT / "data").resolve())

    output_path = ROOT / "configs" / "dataset_runtime.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return str(output_path)
