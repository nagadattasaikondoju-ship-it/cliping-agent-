"""YouTube Shorts via the YouTube Data API v3.

A vertical video up to 3 minutes long is published as a Short automatically.
Setup: create an OAuth client ID (type "Desktop app") in Google Cloud Console
with the YouTube Data API v3 enabled, download client_secret.json, set
YOUTUBE_CLIENT_SECRETS to its path and run `clipagent auth youtube` once.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..env import CONFIG_DIR
from . import PostRequest, PostResult, PublishError

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_PATH = CONFIG_DIR / "youtube_token.json"


def _libs():
    try:
        from google.auth.transport.requests import Request  # noqa: F401
        from google.oauth2.credentials import Credentials  # noqa: F401
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: F401
        from googleapiclient.discovery import build  # noqa: F401
    except ImportError as exc:
        raise PublishError('YouTube posting needs extra packages: pip install -e ".[publish]"') from exc


def status() -> tuple[bool, str]:
    if TOKEN_PATH.exists():
        return True, "authorized"
    if os.environ.get("YOUTUBE_CLIENT_SECRETS"):
        return False, "client secrets set - run `clipagent auth youtube`"
    return False, "not configured (YOUTUBE_CLIENT_SECRETS)"


def authorize(log=print) -> Path:
    """Run the browser OAuth flow once and store a refreshable token."""
    _libs()
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = os.environ.get("YOUTUBE_CLIENT_SECRETS")
    if not secrets or not Path(secrets).exists():
        raise PublishError("Set YOUTUBE_CLIENT_SECRETS to your OAuth client_secret.json path")
    flow = InstalledAppFlow.from_client_secrets_file(secrets, SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    TOKEN_PATH.chmod(0o600)
    log(f"YouTube authorized; token saved to {TOKEN_PATH}")
    return TOKEN_PATH


def _credentials():
    _libs()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not TOKEN_PATH.exists():
        raise PublishError("YouTube is not authorized yet: run `clipagent auth youtube`")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
        else:
            raise PublishError("YouTube token is invalid: run `clipagent auth youtube` again")
    return creds


def upload(req: PostRequest, log=print) -> PostResult:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    youtube = build("youtube", "v3", credentials=_credentials(), cache_discovery=False)
    tags = [h.lstrip("#") for h in req.hashtags]
    if "Shorts" not in tags:
        tags.append("Shorts")
    description = req.caption(5000)
    if "#shorts" not in description.lower():
        description += "\n\n#Shorts"
    body = {
        "snippet": {"title": req.title[:100] or "Clip", "description": description,
                    "tags": tags[:30], "categoryId": "22"},
        "status": {"privacyStatus": req.privacy, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(req.video), mimetype="video/mp4", chunksize=8 * 1024 * 1024,
                            resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        progress, response = request.next_chunk()
        if progress:
            log(f"  youtube: {int(progress.progress() * 100)}%")
    vid = response["id"]
    return PostResult("youtube", True, id=vid, url=f"https://youtube.com/shorts/{vid}",
                      detail=f"uploaded as {req.privacy}")
