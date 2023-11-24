import os

import requests

from autotrain import config, logger


def user_authentication(token):
    logger.info("Authenticating user...")
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            config.HF_API + "/api/whoami-v2",
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")
    return response.json()


def _login_user(user_token):
    user_info = user_authentication(token=user_token)
    username = user_info["name"]

    user_can_pay = user_info["canPay"]
    orgs = user_info["orgs"]

    valid_orgs = [org for org in orgs if org["canPay"] is True]
    valid_orgs = [org for org in valid_orgs if org["roleInOrg"] in ("admin", "write")]
    valid_orgs = [org["name"] for org in valid_orgs]

    valid_can_pay = [username] + valid_orgs if user_can_pay else valid_orgs
    who_is_training = [username] + [org["name"] for org in orgs]
    return user_token, valid_can_pay, who_is_training


def user_validation():
    user_token = os.environ.get("HF_TOKEN", None)

    if user_token is None:
        raise ValueError("Please login with a write token.")

    user_token, valid_can_pay, who_is_training = _login_user(user_token)

    if user_token is None or len(user_token) == 0:
        raise ValueError("Please login with a write token.")

    return user_token, valid_can_pay, who_is_training
