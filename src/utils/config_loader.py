"""
config_loader.py — โหลดและ merge YAML config files
ลำดับ: base.yaml → {model}.yaml → experiments/{run}.yaml
"""
import yaml
from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(model_name: str, experiment_file: str = None, config_dir: str = None) -> dict:
    """
    โหลด config แบบ hierarchical: base → model → experiment override

    Args:
        model_name: ชื่อ algorithm เช่น 'xgboost', 'random_forest', 'ridge'
        experiment_file: ชื่อไฟล์ใน experiments/ เช่น 'exp_winter_data.yaml'
        config_dir: path ของ configs/ (default: PROJECT_ROOT/configs)

    Returns:
        dict: merged configuration

    Example:
        cfg = load_config('xgboost')
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"

    config_dir = Path(config_dir)

    # Layer 1: base config
    with open(config_dir / "base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Layer 2: model-specific config
    model_path = config_dir / f"{model_name}.yaml"
    if model_path.exists():
        with open(model_path, "r", encoding="utf-8") as f:
            config = deep_merge(config, yaml.safe_load(f))

    # Layer 3: experiment override
    if experiment_file:
        exp_path = config_dir / "experiments" / experiment_file
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                config = deep_merge(config, yaml.safe_load(f))

    return config
