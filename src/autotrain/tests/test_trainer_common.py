import importlib.util
import sys
import types
from pathlib import Path


def load_common_module():
    accelerate_stub = types.ModuleType("accelerate")

    class PartialState:
        process_index = 0

    accelerate_stub.PartialState = PartialState
    sys.modules.setdefault("accelerate", accelerate_stub)

    hf_stub = types.ModuleType("huggingface_hub")
    hf_stub.HfApi = object
    sys.modules.setdefault("huggingface_hub", hf_stub)

    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

    pydantic_stub.BaseModel = BaseModel
    sys.modules.setdefault("pydantic", pydantic_stub)

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.TrainerCallback = object
    transformers_stub.TrainerControl = object
    transformers_stub.TrainerState = object
    transformers_stub.TrainingArguments = object
    sys.modules.setdefault("transformers", transformers_stub)

    sys.modules.setdefault("requests", types.ModuleType("requests"))

    autotrain_stub = types.ModuleType("autotrain")
    autotrain_stub.__path__ = [str(Path(__file__).resolve().parents[1])]
    autotrain_stub.is_colab = lambda: False

    class Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    autotrain_stub.logger = Logger()
    sys.modules.setdefault("autotrain", autotrain_stub)

    module_path = Path(__file__).resolve().parents[1] / "trainers" / "common.py"
    spec = importlib.util.spec_from_file_location("autotrain_trainers_common", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_remove_autotrain_data_uses_rmtree_without_shell(tmp_path):
    module = load_common_module()
    project = tmp_path / "project; touch shell-injection"
    data_dir = project / "autotrain-data"
    data_dir.mkdir(parents=True)
    (data_dir / "data.txt").write_text("data")

    def fail_if_called(command):
        raise AssertionError(f"shell should not be used: {command}")

    module.os.system = fail_if_called
    module.remove_global_step = lambda directory: None

    config = types.SimpleNamespace(project_name=str(project))
    module.remove_autotrain_data(config)

    assert not data_dir.exists()
