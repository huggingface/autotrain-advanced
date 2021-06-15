import csv
import json
import os
from typing import Dict

from .utils import flatten_dict


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
    if file_ext in ("csv", "tsv"):
        if task == "entity_extraction":
            raise InvalidFileError(
                f"AutoNLP does not support '{file_ext}' files for entity_extraction tasks. Use .json or .jsonl files!"
            )
        sniffer = csv.Sniffer()
        with open(path, encoding="utf-8", errors="replace") as f:
            # Validate delimiter
            sample = f.readline()
            expected_delimiter = "\t" if file_ext == "tsv" else ","
            actual_delimiter = sniffer.sniff(sample, delimiters=",;\t").delimiter

        if actual_delimiter != expected_delimiter:
            if task == "entity_extraction":
                additional_help = (
                    "\nFor entity_extraction tasks, AutoNLP expects tokens / tags to be tab-separated "
                    "and sentences to be empty-line separated."
                )
            else:
                additional_help = ""
            raise InvalidFileError(
                "Incorrect delimiter '"
                + (r"\t" if actual_delimiter == "\t" else actual_delimiter)
                + f"' for file '{file_name}'! "
                + "Expected delimiter is: '"
                + (r"\t" if expected_delimiter == "\t" else actual_delimiter)
                + "'."
                + additional_help
            )

        # Extract column_names
        column_names = sample.splitlines()[0].split(actual_delimiter)

    elif file_ext in ("json", "jsonl"):
        with open(path, encoding="utf-8") as f:
            first_line = f.readline()
            second_line = f.readline()
        try:
            json.loads(first_line)
            json.loads(second_line)
        except ValueError:
            raise InvalidFileError(
                f"File `{file_name}` is not a valid JSON-lines file! Each line must be a valid JSON mapping."
            )

        # Extract column_names
        first_item = json.loads(first_line)
        if not isinstance(first_item, dict):
            raise InvalidFileError(
                "File `{file_name}` is not a valid JSON-lines file! Each line must be a valid JSON mapping."
            )
        column_names = list(flatten_dict(first_item, 1).keys())

    else:
        raise InvalidFileError(f"AutoNLP does not support `.{file_ext}` files yet!")

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
