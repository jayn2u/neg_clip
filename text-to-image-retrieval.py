from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_module_path = Path(__file__).resolve().parent / "sugarcrepe-pedes.py"
_spec = importlib.util.spec_from_file_location("pedes_eval", _module_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load evaluation module from {_module_path}")
_pedes_eval = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _pedes_eval
_spec.loader.exec_module(_pedes_eval)

require_config_path = _pedes_eval.require_config_path
run_text_to_image_retrieval = _pedes_eval.run_text_to_image_retrieval


if __name__ == "__main__":
    run_text_to_image_retrieval(
        require_config_path("RETRIEVAL_CONFIG", "text_to_image_retrieval")
    )
