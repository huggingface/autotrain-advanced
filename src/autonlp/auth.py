import json
import os
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import List, Optional

import requests
from loguru import logger
from typing_extensions import TypedDict

from . import config
from .utils import http_get


class NotAuthenticatedError(ValueError):
    """The user is not authenticated"""

    pass


class AuthenticationError(ValueError):
    """An error occurred while authenticated"""

    pass


class ForbiddenError(ValueError):
    """The user is authenticated but has not the appropriate rights"""

    pass


@dataclass
class AutoNLPIdentity:
    name: str
    full_name: str
    is_org: bool


LoginInfo = TypedDict("LoginInfo", {"token": str, "identities": List[AutoNLPIdentity], "selected_identity": str})


def login(token: str, save_dir: Optional[str] = None) -> LoginInfo:
    if token.startswith("api_org"):
        raise AuthenticationError("Cannot login with an organization token! Please login as a user.")

    try:
        auth_resp = http_get(path="/whoami-v2", domain=config.HF_API, token=token, token_prefix="Bearer")
    except requests.HTTPError as err:
        if err.response.status_code == 401:
            raise AuthenticationError("The provided token is invalid") from err
        raise err

    json_resp = auth_resp.json()

    identities = [
        AutoNLPIdentity(
            name=identity["name"],
            full_name=identity["fullname"],
            is_org=identity["type"] == "org",
        )
        for identity in [json_resp] + json_resp.get("orgs", [])
    ]

    try:
        login_info = _save_identities(identities=identities, token=token, save_dir=save_dir)
    except ValueError as err:
        raise AuthenticationError(*err.args) from err

    return login_info


def _save_identities(
    identities: List[AutoNLPIdentity],
    token: str,
    selected_identity: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> LoginInfo:
    if save_dir is None:
        save_dir = "~/.autonlp"
    save_dir = os.path.expanduser(save_dir)

    logger.info(f"🗝 Storing credentials in: {save_dir}")
    if os.path.isfile(save_dir):
        raise ValueError(f"'{save_dir}' is a file, cannot save credentials there")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if selected_identity is None:
        selected_identity = identities[0].name

    login_dict = LoginInfo(token=token, identities=identities, selected_identity=selected_identity)
    save_path = os.path.join(save_dir, "autonlp.json")
    with open(save_path, "w") as fp:
        json.dump(login_dict, fp)

    return login_dict


def login_from_conf(save_dir: Optional[str] = None) -> LoginInfo:
    if save_dir is None:
        save_dir = "~/.autonlp"
    save_dir = os.path.expanduser(save_dir)
    save_path = os.path.join(save_dir, "autonlp.json")
    if not os.path.isfile(save_path):
        raise NotAuthenticatedError(f"{save_path} not found")

    logger.info(f"🗝 Retrieving credentials from '{save_dir}'")
    with open(save_path, "r") as fp:
        try:
            login_dict = json.load(fp)
        except JSONDecodeError as err:
            raise AuthenticationError(f"{save_path} is malformed") from err

    return login_dict


def select_identity(new_identity: str, save_dir: Optional[str] = None):
    login_info = login_from_conf(save_dir)
    if new_identity not in [identity.name for identity in login_info["identities"]]:
        raise ForbiddenError(f"Cannot impersonate {new_identity}: if it's an org, make sure you're a member of it")
    new_login_info = {
        **login_info,
        "selected_identity": new_identity,
    }
    _save_identities(**new_login_info, save_dir=save_dir)
