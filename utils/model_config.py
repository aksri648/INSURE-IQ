import os
import shutil
import subprocess

CONFIG_PATH = os.environ.get("INSUREIQ_MODEL_CONFIG", "/tmp/model_config.env")


def detect_vram_mib() -> int:
    """Return total VRAM in MiB for the first GPU, or 0 if no GPU."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        ).stdout.strip().splitlines()
        return int(out[0]) if out else 0
    except Exception:
        return 0


def auto_select_models() -> dict:
    vram = detect_vram_mib()
    if vram > 35000:
        return {"OCR_MODEL": "llava:13b",
                "ANALYST_MODEL": "deepseek-r1:14b",
                "EMBED_MODEL": "nomic-embed-text"}
    # T4 / smaller GPU / CPU fallback
    return {"OCR_MODEL": "llava:7b",
            "ANALYST_MODEL": "deepseek-r1:7b",
            "EMBED_MODEL": "nomic-embed-text"}


def write_model_config() -> dict:
    cfg = {
        "OCR_MODEL": os.environ.get("OCR_MODEL"),
        "ANALYST_MODEL": os.environ.get("ANALYST_MODEL"),
        "EMBED_MODEL": os.environ.get("EMBED_MODEL"),
    }
    if not all(cfg.values()):
        auto = auto_select_models()
        for k, v in auto.items():
            cfg[k] = cfg[k] or v

    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}={v}\n")
    return cfg


def read_model_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return write_model_config()
    config = {}
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            config[k] = v
    # Backfill any missing keys
    defaults = auto_select_models()
    for k, v in defaults.items():
        config.setdefault(k, v)
    return config


def get(key: str, default: str = "") -> str:
    return read_model_config().get(key, default)
