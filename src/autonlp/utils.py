from typing import Any, Dict, Optional, Union

import requests

from . import config
from .tasks import TASKS


FORMAT_TAG = "\033[{code}m"
RESET_TAG = FORMAT_TAG.format(code=0)
BOLD_TAG = FORMAT_TAG.format(code=1)
RED_TAG = FORMAT_TAG.format(code=91)
GREEN_TAG = FORMAT_TAG.format(code=92)
YELLOW_TAG = FORMAT_TAG.format(code=93)
PURPLE_TAG = FORMAT_TAG.format(code=95)
CYAN_TAG = FORMAT_TAG.format(code=96)


class UnauthenticatedError(Exception):
    pass


class UnreachableAPIError(Exception):
    pass


def get_auth_headers(token: str, prefix: str = "autonlp"):
    # return {"Authorization": f"autonlp {token}"}
    return {"Authorization": f"{prefix} {token}"}


def http_get(
    path: str,
    token: str,
    domain: str = config.HF_AUTONLP_BACKEND_API,
    token_prefix: str = "autonlp",
    suppress_logs: bool = False,
    **kwargs,
) -> requests.Response:
    """HTTP GET request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.get(
            url=domain + path, headers=get_auth_headers(token=token, prefix=token_prefix), **kwargs
        )
    except requests.exceptions.ConnectionError:
        raise UnreachableAPIError("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def http_post(
    path: str,
    token: str,
    payload: Optional[Dict] = None,
    domain: str = config.HF_AUTONLP_BACKEND_API,
    suppress_logs: bool = False,
    **kwargs,
) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.post(
            url=domain + path, json=payload, headers=get_auth_headers(token=token), allow_redirects=True, **kwargs
        )
    except requests.exceptions.ConnectionError:
        raise UnreachableAPIError("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def get_task(task_id: int) -> str:
    for key, value in TASKS.items():
        if value == task_id:
            return key
    return "❌ Unsupported task! Please update autonlp"


def flatten_dict(dictionary: Dict[str, Union[str, Any]], max_depth: int):
    """Recursively flattens a dict"""
    flat_dict = {}

    def _flatten(dictionary: Dict[str, Union[str, Any]], max_depth: int, parent=None):
        for key, value in dictionary.items():
            flat_key = key if parent is None else ".".join([parent, key])
            if max_depth and isinstance(value, dict):
                _flatten(value, max_depth - 1, flat_key)
            else:
                flat_dict[flat_key] = value

    _flatten(dictionary, max_depth, None)
    return flat_dict
