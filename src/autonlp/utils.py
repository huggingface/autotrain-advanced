from typing import Dict, List, Optional

import requests
from loguru import logger

from . import config


FORMAT_TAG = "\033[{code}m"
RESET_TAG = FORMAT_TAG.format(code=0)
BOLD_TAG = FORMAT_TAG.format(code=1)
GREEN_TAG = FORMAT_TAG.format(code=32)
RED_TAG = FORMAT_TAG.format(code=31)
PURPLE_TAG = FORMAT_TAG.format(code=35)
CYAN_TAG = FORMAT_TAG.format(code=36)
YELLOW_TAG = FORMAT_TAG.format(code=33)


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
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if not suppress_logs:
            logger.error(f"❌ Operation failed! Details: {err.response.text}")
        raise
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
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if not suppress_logs:
            logger.error(f"❌ Operation failed! Details: {err.response.text}")
        raise
    return response


def http_upload_files(
    path: str, token: str, data: dict, files_info: List, domain: str = config.HF_AUTONLP_BACKEND_API, **kwargs
) -> requests.Response:
    """Uploads files to AutoNLP"""
    try:
        response = requests.post(
            url=domain + path,
            data=data,
            files=files_info,
            headers=get_auth_headers(token),
            allow_redirects=True,
            **kwargs,
        )
    except requests.exceptions.ConnectionError:
        raise UnreachableAPIError("❌ Failed to reach AutoNLP API, check your internet connection")
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"❌ Operation failed! Details: {err.response.text}")
        raise
    return response
