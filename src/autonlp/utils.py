import requests
from typing import Optional, Dict, List
from . import config


class UnauthenticatedError(Exception):
    pass


class UnreachableAPIError(Exception):
    pass


def get_auth_headers(token: str):
    return {"Authorization": f"autonlp {token}"}


def http_get(path: str, token: str, domain: str = config.HF_AUTONLP_BACKEND_API, **kwargs) -> requests.Response:
    """HTTP GET request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        return requests.get(url=domain + path, headers=get_auth_headers(token=token), **kwargs)
    except requests.exceptions.ConnectionError:
        raise UnreachableAPIError("❌ Failed to reach AutoNLP API, check your internet connection")


def http_post(
    path: str, token: str, payload: Optional[Dict] = None, domain: str = config.HF_AUTONLP_BACKEND_API, **kwargs
) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        return requests.post(url=domain + path, json=payload, headers=get_auth_headers(token=token), **kwargs)
    except requests.exceptions.ConnectionError:
        raise UnreachableAPIError("❌ Failed to reach AutoNLP API, check your internet connection")


def http_upload_files(
    path: str, token: str, data: dict, files_info: List, domain: str = config.HF_AUTONLP_BACKEND_API, **kwargs
) -> requests.Response:
    """Uploads files to AutoNLP"""
    try:
        return requests.post(url=domain + path, data=data, files=filepaths, headers=get_auth_headers(token), **kwargs)
    except requests.exceptions.ConnectionError:
        raise UnreachableAPIError("❌ Failed to reach AutoNLP API, check your internet connection")
