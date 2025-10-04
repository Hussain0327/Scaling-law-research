"""Tests for scripts.run_sweep helpers."""

import importlib.util
from pathlib import Path


def _load_run_sweep():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("run_sweep", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_generate_sweep_configs_produces_independent_dicts():
    base_cfg = {
        "experiment": {"name": "demo"},
        "model": {"d_model": 64},
        "data": {},
        "training": {},
        "sweep": {
            "parameter": "model.d_model",
            "values": [64, 128],
        },
    }

    module = _load_run_sweep()
    configs = module.generate_sweep_configs(base_cfg)

    # Mutating the first config should not affect others or the base configuration
    configs[0]["model"]["d_model"] = 999

    assert base_cfg["model"]["d_model"] == 64
    assert configs[1]["model"]["d_model"] == 128
