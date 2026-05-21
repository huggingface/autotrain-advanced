import io
import sys
import types
import zipfile

import pytest


def _install_stubs():
    logging_module = types.ModuleType("autotrain.logging")

    class _Logger:
        def get_logger(self):
            return None

    logging_module.Logger = _Logger
    sys.modules.setdefault("autotrain.logging", logging_module)
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))

    class _Preprocessor:
        pass

    preprocessor_modules = {
        "autotrain.preprocessor.tabular": [
            "TabularBinaryClassificationPreprocessor",
            "TabularMultiClassClassificationPreprocessor",
            "TabularMultiColumnRegressionPreprocessor",
            "TabularMultiLabelClassificationPreprocessor",
            "TabularSingleColumnRegressionPreprocessor",
        ],
        "autotrain.preprocessor.text": [
            "LLMPreprocessor",
            "SentenceTransformersPreprocessor",
            "Seq2SeqPreprocessor",
            "TextBinaryClassificationPreprocessor",
            "TextExtractiveQuestionAnsweringPreprocessor",
            "TextMultiClassClassificationPreprocessor",
            "TextSingleColumnRegressionPreprocessor",
            "TextTokenClassificationPreprocessor",
        ],
        "autotrain.preprocessor.vision": [
            "ImageClassificationPreprocessor",
            "ImageRegressionPreprocessor",
            "ObjectDetectionPreprocessor",
        ],
        "autotrain.preprocessor.vlm": ["VLMPreprocessor"],
    }
    for module_name, names in preprocessor_modules.items():
        module = types.ModuleType(module_name)
        for name in names:
            setattr(module, name, _Preprocessor)
        sys.modules.setdefault(module_name, module)


def _zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for name, content in entries.items():
            zip_file.writestr(name, content)
    buffer.seek(0)
    return buffer


def test_extract_dataset_zip_rejects_traversal_path(tmp_path):
    _install_stubs()
    from autotrain.dataset import _extract_dataset_zip

    archive = _zip_bytes({"../escape.png": "outside"})

    with pytest.raises(ValueError, match="Unsafe path"):
        _extract_dataset_zip(archive, tmp_path)

    assert not (tmp_path.parent / "escape.png").exists()


def test_extract_dataset_zip_removes_macosx_metadata(tmp_path):
    _install_stubs()
    from autotrain.dataset import _extract_dataset_zip

    archive = _zip_bytes(
        {
            "__MACOSX/._image.png": "metadata",
            "class-a/image.png": "image",
        }
    )

    _extract_dataset_zip(archive, tmp_path)

    assert not (tmp_path / "__MACOSX").exists()
    assert (tmp_path / "class-a" / "image.png").read_text() == "image"
