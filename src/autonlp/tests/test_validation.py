import numpy as np
import pandas as pd
import pytest
from autonlp.validation import InvalidColMappingError, validate_col_mapping, validate_chunk, validate_state


def test_validate_col_mapping():
    col_mapping = {
        "review": "text",
        "sentiment": "target",
    }
    validate_col_mapping(col_mapping, task="binary_classification")

    col_mapping = {
        "review": "text",
        "sentiment": "traget",  # Typo
    }
    with pytest.raises(InvalidColMappingError) as exc_info:
        validate_col_mapping(col_mapping, task="binary_classification")
    err_msg = exc_info.value.args[0]
    assert "Unexpected column" in err_msg and "traget" in err_msg, err_msg

    col_mapping = {
        "review": "text",
        "sentiment": "target",
        "positive": "target",  # Duplicate
    }
    with pytest.raises(InvalidColMappingError) as exc_info:
        validate_col_mapping(col_mapping, task="binary_classification")
    err_msg = exc_info.value.args[0]
    assert "Duplicate column" in err_msg and "target" in err_msg, err_msg

    col_mapping = {
        "review": "text",
    }
    with pytest.raises(InvalidColMappingError) as exc_info:
        validate_col_mapping(col_mapping, task="binary_classification")
    err_msg = exc_info.value.args[0]
    assert "Missing column" in err_msg and "target" in err_msg, err_msg


def test_validate_state_binary_clf():
    state = {"unique_labels": {"pos", "neg"}}
    res = validate_state(state, task="binary_classification")
    assert res is None, res

    state = {"unique_labels": {"pos", "neg", "neutral"}}
    res = validate_state(state, task="binary_classification")
    assert isinstance(res, str)

    state = {"unique_labels": {"pos"}}
    res = validate_state(state, task="binary_classification")
    assert isinstance(res, str)


def test_validate_state_multiclass_clf():
    state = {"unique_labels": {"pos", "neg", "neutral"}}
    res = validate_state(state, task="multi_class_classification")
    assert res is None, res

    state = {"unique_labels": {"pos", "neg"}}
    res = validate_state(state, task="multi_class_classification")
    assert isinstance(res, str)

    state = {"unique_labels": {"pos"}}
    res = validate_state(state, task="multi_class_classification")
    assert isinstance(res, str)


def test_validate_chunk_clf():
    state = {}
    chunk = pd.DataFrame(
        {"review": ["text"] * 5, "sentiment": ["pos", "neg", "pos", "neg", "neutral"], "extra": [np.nan] * 5}
    )
    col_mapping = {"review": "text", "sentiment": "target"}

    res = validate_chunk(chunk, task="binary_classification", col_mapping=col_mapping, state=state)
    assert res is None, res
    assert state["unique_labels"] == {"pos", "neg", "neutral"}


def test_validate_chunk_missing_column():
    state = {}
    chunk = pd.DataFrame({"sentiment": ["pos", "neg", "pos", "neg", "neutral"], "extra": [np.nan] * 5})
    col_mapping = {"review": "text", "sentiment": "target"}

    res = validate_chunk(chunk, task="binary_classification", col_mapping=col_mapping, state=state)
    assert isinstance(res, str)
    assert "review" in res, res


def test_validate_chunk_missing_values():
    state = {}
    chunk = pd.DataFrame(
        {"review": ["text"] * 5, "sentiment": ["pos", "neg", np.nan, "neg", "neutral"], "extra": [np.nan] * 5}
    )
    col_mapping = {"review": "text", "sentiment": "target"}

    res = validate_chunk(chunk, task="binary_classification", col_mapping=col_mapping, state=state)
    assert isinstance(res, str)
    assert res == "Row(s) with index [2] have missing values"


def test_validate_chunk_regression():
    state = {}
    chunk = pd.DataFrame({"review": ["text"] * 5, "score": [12, "24", 0.7, "1e4", "neutral"], "extra": [np.nan] * 5})
    col_mapping = {"review": "text", "score": "target"}

    res = validate_chunk(chunk, task="single_column_regression", col_mapping=col_mapping, state=state)
    assert isinstance(res, str)
    assert "Some value in column score cannot be interpreted as a number" in res
