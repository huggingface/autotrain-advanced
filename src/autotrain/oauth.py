"""OAuth support for AutoTrain.
Taken from: https://github.com/gradio-app/gradio/blob/main/gradio/oauth.py
"""

from __future__ import annotations

import hashlib
import os
import random
import string
import urllib.parse

import fastapi
from authlib.integrations.starlette_client import OAuth
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware


OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
OAUTH_SCOPES = os.environ.get("OAUTH_SCOPES")
OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL")
RANDOM_STRING = "".join(random.choices(string.ascii_letters + string.digits, k=20))


def attach_oauth(app: fastapi.FastAPI):
    _add_oauth_routes(app)
    # Session Middleware requires a secret key to sign the cookies. Let's use a hash
    # of the OAuth secret key to make it unique to the Space + updated in case OAuth
    # config gets updated.
    session_secret = OAUTH_CLIENT_SECRET + RANDOM_STRING
    # ^ if we change the session cookie format in the future, we can bump the version of the session secret to make
    #   sure cookies are invalidated. Otherwise some users with an old cookie format might get a HTTP 500 error.
    app.add_middleware(
        SessionMiddleware,
        secret_key=hashlib.sha256(session_secret.encode()).hexdigest(),
        https_only=True,
        same_site="none",
    )


def _add_oauth_routes(app: fastapi.FastAPI) -> None:
    """Add OAuth routes to the FastAPI app (login, callback handler and logout)."""
    # Check environment variables
    msg = (
        "OAuth is required but {} environment variable is not set. Make sure you've enabled OAuth in your Space by"
        " setting `hf_oauth: true` in the Space metadata."
    )
    if OAUTH_CLIENT_ID is None:
        raise ValueError(msg.format("OAUTH_CLIENT_ID"))
    if OAUTH_CLIENT_SECRET is None:
        raise ValueError(msg.format("OAUTH_CLIENT_SECRET"))
    if OAUTH_SCOPES is None:
        raise ValueError(msg.format("OAUTH_SCOPES"))
    if OPENID_PROVIDER_URL is None:
        raise ValueError(msg.format("OPENID_PROVIDER_URL"))

    # Register OAuth server
    oauth = OAuth()
    oauth.register(
        name="huggingface",
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        client_kwargs={"scope": OAUTH_SCOPES},
        server_metadata_url=OPENID_PROVIDER_URL + "/.well-known/openid-configuration",
    )

    # Define OAuth routes
    @app.get("/login/huggingface")
    async def oauth_login(request: fastapi.Request):
        """Endpoint that redirects to HF OAuth page."""
        redirect_uri = request.url_for("auth")
        redirect_uri_as_str = str(redirect_uri)
        if redirect_uri.netloc.endswith(".hf.space"):
            redirect_uri_as_str = redirect_uri_as_str.replace("http://", "https://")
        return await oauth.huggingface.authorize_redirect(request, redirect_uri_as_str)  # type: ignore

    @app.get("/auth")
    async def auth(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that handles the OAuth callback."""
        oauth_info = await oauth.huggingface.authorize_access_token(request)  # type: ignore
        request.session["oauth_info"] = oauth_info
        return _redirect_to_target(request)


def _generate_redirect_uri(request: fastapi.Request) -> str:
    if "_target_url" in request.query_params:
        # if `_target_url` already in query params => respect it
        target = request.query_params["_target_url"]
    else:
        # otherwise => keep query params
        target = "/?" + urllib.parse.urlencode(request.query_params)

    redirect_uri = request.url_for("oauth_redirect_callback").include_query_params(_target_url=target)
    redirect_uri_as_str = str(redirect_uri)
    if redirect_uri.netloc.endswith(".hf.space"):
        # In Space, FastAPI redirect as http but we want https
        redirect_uri_as_str = redirect_uri_as_str.replace("http://", "https://")
    return redirect_uri_as_str


def _redirect_to_target(request: fastapi.Request, default_target: str = "/") -> RedirectResponse:
    # target = request.query_params.get("_target_url", default_target)
    target = "https://huggingface.co/spaces/" + os.environ.get("SPACE_ID")
    return RedirectResponse(target)
