import os
from typing import Dict

from datasets import Dataset, load_dataset


COLUMNS_PER_TASK = {
    "binary_classification": ("text", "target"),
    "multi_class_classification": ("text", "target"),
    "entity_extraction": ("tokens", "tags"),
    "single_column_regression": ("text", "target"),
    "speech_recognition": ("text", "path"),
    "summarization": ("text", "target"),
    "extractive_question_answering": (
        "context",
        "question",
        "answers.answer_start",
        "answers.text",
    ),
}


class InvalidFileError(ValueError):
    pass


class InvalidColMappingError(ValueError):
    pass


def validate_file(path: str, task: str, file_ext: str, col_mapping: Dict[str, str]):
    file_name = os.path.basename(path)
    try:
        if file_ext in ("csv", "tsv"):
            sample: Dataset = load_dataset("csv", data_files=path, split="train[:5%]", sep=None, header=0)
        elif file_ext in ("json", "jsonl"):
            sample: Dataset = load_dataset("json", data_files=path, split="train[:5%]")
        else:
            raise InvalidFileError(f"AutoNLP does not support `.{file_ext}` files yet!")
    except Exception as err:
        if isinstance(err, InvalidFileError):
            raise err
        raise InvalidFileError(f"{file_name} could not be loaded with datasets!\nError: {err}") from err

    column_names = sample.flatten().column_names
    invalid_columns_source = set(col_mapping.keys()) - set(column_names)
    if invalid_columns_source:
        raise InvalidColMappingError(
            "Columns "
            + ",".join([f"'{col_name}'" for col_name in invalid_columns_source])
            + " could not be found in the provided file (which has columns: "
            + ",".join([f"'{col_name}'" for col_name in column_names])
            + ")"
        )

    invalid_columns_target = set(COLUMNS_PER_TASK[task]) - set(col_mapping.values())
    if invalid_columns_target:
        raise InvalidColMappingError(
            "\n".join(
                ["Provided column mapping is:"]
                + [f"   '{src_col}' -> '{dst_col}'" for src_col, dst_col in col_mapping.items()]
                + ["While expecting column mapping like:"]
                + [
                    f"   'original_col_name' -> '{col_name}' (AutoNLP column name)"
                    for col_name in COLUMNS_PER_TASK[task]
                ]
            )
        )
