from typing import Dict, Iterable, List, Optional

import pandas as pd


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


def find_duplicate(iterable: Iterable) -> list:
    duplicates = []
    seen = set()
    for value in iterable:
        if value in seen:
            duplicates.append(value)
        else:
            seen.add(value)
    return duplicates


def find_columns_in_mapping(column_name: str, col_mapping: Dict[str, str]) -> List[str]:
    """Returns a list of keys such that col_mapping[key] == column_name"""
    assert column_name in set(col_mapping.values())
    matching_columns = [src for src, dst in col_mapping.items() if dst == column_name]
    return matching_columns


def get_rows_iter(path: str, file_ext: str):
    if file_ext in ("csv", "tsv"):
        return pd.read_csv(path, header=0, delimiter=None, chunksize=1000)
    elif file_ext in ("json", "jsonl"):
        return pd.read_json(path, lines=True, chunksize=1000)
    else:
        raise InvalidFileError(f"AutoNLP does not support `.{file_ext}` files yet!")


def validate_file(path: str, task: str, file_ext: str, col_mapping: Dict[str, str]):
    """Validates a given file before uploading it

    Args:
        path (``str``):
            Path to the file being validated
        task (``str``):
            Task name of the project
        file_ext (``str``):
            The file's extension (eg 'csv')
        col_mapping (``dict``)
    """
    validate_col_mapping(col_mapping, task)
    rows_iter = get_rows_iter(path, file_ext)

    state = {}
    try:
        for chunk in rows_iter:
            validation_error = validate_chunk(chunk, task=task, col_mapping=col_mapping, state=state)
            if validation_error is not None:
                raise InvalidFileError(validation_error)
    except (pd.errors.ParserError, ValueError) as err:
        if isinstance(err, InvalidFileError):
            raise err
        raise InvalidFileError(f"Malformed file") from err

    validation_error = validate_state(state, task=task)
    if validation_error is not None:
        raise InvalidFileError(validation_error)


def validate_chunk(chunk: pd.DataFrame, task: str, col_mapping: Dict[str, str], state: dict) -> Optional[str]:
    """Validates a chunk of data

    Args:
        chunk (``pandas.DataFrame``):
            The chunk to validate
        task (``str``):
            The task name of the project
        col_mapping (``dict``):
            A mapping from the chunk's column names to the AutoNLP column names
        state (``dict``):
            A dictionary that stores state between chunks

    Returns:
        ``None`` if the chunk is valid, or an ``str`` error message otherwise
    """
    # Check column names
    actual_column_names = set(chunk.columns)
    missing_columns = set(col_mapping.keys()) - actual_column_names
    if missing_columns:
        start, end = chunk.index[0], chunk.index[-1]
        return (
            f"Columns {','.join(missing_columns)} could not be found between rows with index {start} {end} "
            f"(which has columns {actual_column_names})"
        )

    chunk = chunk[list(col_mapping.keys())]

    # Check for missing (NaN) values
    if chunk.isna().any(axis=None):
        isna = chunk.isna().any(axis=1)
        rows_with_na = isna[isna].index.values.tolist()
        return f"Row(s) with index {rows_with_na} have missing values"

    if task in ("binary_classification", "multi_class_classification"):
        target_column = find_columns_in_mapping("target", col_mapping)[0]
        state["unique_labels"] = state.get("unique_labels", set()) | set(chunk[target_column].values)

    if task == "single_column_regression":
        target_column = find_columns_in_mapping("target", col_mapping)[0]
        try:
            pd.to_numeric(chunk[target_column], errors="raise")
        except (ValueError, TypeError) as err:
            return f"Some value in column {target_column} cannot be interpreted as a number: {err}"

    return None


def validate_state(state: dict, task: str) -> Optional[str]:
    """Validates the state dictionary after :func:``validate_chunk`` has been run on the whole file

    Args:
        state (``dict``):
            The state dictionary
        task (``str``):
            The task name of the project

    Returns:
        ``None`` if the state is valid, or an ``str`` error message otherwise
    """
    if task == "binary_classification":
        unique_labels = state["unique_labels"]
        if len(unique_labels) != 2:
            return f"Invalid number of labels. Expected 2 unique labels for binary_classification, got {len(unique_labels)}: {unique_labels}"

    if task == "multi_class_classification":
        unique_labels = state["unique_labels"]
        if len(unique_labels) <= 2:
            return (
                f"Invalid number of labels. Expected at least 3 unique labels for multi_class_classification, got {len(unique_labels)}: {unique_labels};"
                " Consider creating a binary_classification project if this is expected."
            )


def validate_col_mapping(col_mapping: Dict[str, str], task: str) -> Optional[str]:
    """Validates the given col_mapping against the project's task

    Args:
        col_mapping (``dict``):
            The provided col_mapping, mapping the file's column names to the AutoNLP column names
        task (``str``):
            The task name of the project

    Returns:
        ``None`` if the column_mapping is valid, or an ``str`` error message otherwise
    """
    if sorted(col_mapping.values()) != sorted(COLUMNS_PER_TASK[task]):
        if len(list(col_mapping.values())) >= len(COLUMNS_PER_TASK[task]):
            extra_columns = set(col_mapping.values()) - set(COLUMNS_PER_TASK[task])
            if not extra_columns:
                duplicate_columns = find_duplicate(list(col_mapping.values()))
                reason = f"Duplicate column(s) in column mapping: {duplicate_columns}."
            else:
                reason = f"Unexpected column(s) in col_mapping: {extra_columns}."
            raise InvalidColMappingError(reason + f" Expected columns for task {task} are: {COLUMNS_PER_TASK[task]}")
        else:
            missing_columns = set(COLUMNS_PER_TASK[task]) - set(col_mapping.values())
            raise InvalidColMappingError(
                f"Missing column(s) in col_mapping: {missing_columns}. Expected columns for task {task} are: {COLUMNS_PER_TASK[task]}"
            )
