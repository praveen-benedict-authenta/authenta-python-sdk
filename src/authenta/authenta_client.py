"""
example usage:
client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    client_id="...",
    client_secret="...",
)
"""

import os
import time
import mimetypes
from typing import Any, Dict, Optional

import requests
from .authenta_exceptions import (
    AuthentaError,
    AuthenticationError,
    AuthorizationError,
    QuotaExceededError,
    InsufficientCreditsError,
    ValidationError,
    ServerError,
)
from curl_cffi import requests as cffi_requests

def _raise_for_authenta_error(resp: requests.Response) -> None:
    """
    Map an Authenta API error response to a rich SDK exception.

    Expects JSON like: {"code": "IAM001", "type": "...", "message": "..."}.
    Falls back to HTTP-based mapping if the body is not JSON.
    """
    status = resp.status_code
    try:
        data = resp.json()
    except ValueError:
        # Non-JSON error body
        if 400 <= status < 500:
            raise ValidationError(
                message=resp.text or "Client error",
                status_code=status,
            )
        if status >= 500:
            raise ServerError(
                message=resp.text or "Server error",
                status_code=status,
            )
        resp.raise_for_status()
        return

    code = data.get("code") or "unknown"
    message = data.get("message") or resp.reason or "Unknown error"
    details = data

    if code == "IAM001":
        raise AuthenticationError(message, status_code=status, details=details)
    if code == "IAM002":
        raise AuthorizationError(message, status_code=status, details=details)
    if code == "AA001":
        raise QuotaExceededError(message, status_code=status, details=details)
    if code == "U007":
        raise InsufficientCreditsError(message, status_code=status, details=details)

    if 400 <= status < 500:
        raise ValidationError(message, code=code, status_code=status, details=details)
    if status >= 500:
        raise ServerError(message, code=code, status_code=status, details=details)

    raise AuthentaError(message, code=code, status_code=status, details=details)


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    """
    Safely parse JSON; if body is empty, return {}.
    If body is non-JSON, raise a ValidationError with the raw body snippet.
    """
    text = resp.text or ""
    if not text.strip():
        return {}
    try:
        return resp.json()
    except ValueError:
        raise ValidationError(
            message="Expected JSON response but got non-JSON payload",
            status_code=resp.status_code,
            details={"body": text[:200]},
        )


class AuthentaClient:
    """
    Authenta Python SDK.

    Features:
    - Builds Auth headers with x-client-id / x-client-secret.
    - Wraps /api/media endpoints for create, get, list, delete.
    - Implements two-step upload (POST /api/media -> PUT to S3).
    - Process deepfake-detection.
    """

    def __init__(self, base_url: str, client_id: str, client_secret: str):
        """
        Create new Authenta client.

        Args:
            base_url: Authenta API base URL, e.g. "https://platform.authenta.ai".
            client_id: Your Authenta client ID.
            client_secret: Your Authenta client secret.
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret

    def _headers(self) -> Dict[str, str]:
        """Return default headers for Authenta API calls."""
        return {
            "x-client-id": self.client_id,
            "x-client-secret": self.client_secret,
            "Content-Type": "application/json",
        }

    def _content_type(self, path: str) -> str:
        """
        Guess the MIME type for a file path.

        Falls back to 'application/octet-stream' if unknown.
        """
        filetype, _ = mimetypes.guess_type(path)
        return filetype or "application/octet-stream"

    def create_media(
        self,
        name: str,
        content_type: str,
        size: int,
        model_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        POST /api/media: create a media record and get an upload URL.

        Args:
            name: Original file name.
            content_type: MIME type of the file (e.g. "image/png", "video/mp4").
            size: File size in bytes.
            model_type: Detection model type, e.g. "AC-1" or "DF-1".
                       "AC-1" - Image, "DF-1" - Video.

        Returns:
            Parsed JSON response containing at least 'mid' and 'uploadUrl'.
        """
        url = f"{self.base_url}/api/media"
        
        payload = {
            "name": name,
            "contentType": content_type,
            "size": size,
            "modelType": model_type,
        }

        if model_type.upper() == "FI-1":
            fi_params = {
                "isSingleFace": kwargs.get("isSingleFace", True),
                "faceswapCheck": kwargs.get("faceswapCheck"),
                "livenessCheck": kwargs.get("livenessCheck"),
                "faceSimilarityCheck": kwargs.get("faceSimilarityCheck"),
            }
            payload.update({
                "metadata": {
                    i: j for i, j in fi_params.items()
                }
            })
        if self.base_url=="https://platform.authenta.ai":
            resp = cffi_requests.post(url, json=payload, headers=self._headers(), impersonate="chrome", timeout=30)
        else:
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=30)
        print(f"create_media payload: {payload}")
        print("create_media raw:", resp.status_code, repr(resp.text[:200]))
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)


    def get_media(self, mid: str) -> Dict[str, Any]:
        """
        GET /api/media/{mid}: fetch a single media record.

        Args:
            mid: Media ID returned by create_media / upload_file.

        Returns:
            Parsed JSON media record.
        """
        url = f"{self.base_url}/api/media/{mid}"
        if self.base_url=="https://platform.authenta.ai":
            resp = cffi_requests.get(url, headers=self._headers(), impersonate="chrome", timeout=30)
        else:
            resp = requests.get(url, headers=self._headers(), timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)

    def upload_file(self, path: str, model_type: str, **kwargs) -> Dict[str, Any]:
        """
        Upload a file via the two-step Authenta media flow.

        Steps:
            1) POST /api/media to create the record and obtain 'mid' + 'uploadUrl'.
            2) PUT the file bytes to the presigned S3 'uploadUrl'.

        Args:
            path: Local path to the media file.
            model_type: Detection model type to use, e.g. "AC-1" or "DF-1".
            **kwargs: Additional parameters for the media creation request.

        Returns:
            The JSON response from POST /api/media (includes 'mid', 'status', etc.).
        """
        filename = os.path.basename(path)
        content_type = self._content_type(path)
        size = os.path.getsize(path)

        # Step 1: create media
        meta = self.create_media(
            name=filename,
            content_type=content_type,
            size=size,
            model_type=model_type,
            **kwargs,
        )
        upload_url = meta.get("uploadUrl")
        if not upload_url:
            raise RuntimeError("No uploadUrl in create_media response")

        # Step 2: upload to S3
        with open(path, "rb") as f:
            put_resp = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": content_type},
                timeout=300,
            )
        if model_type.upper() == "FI-1":
            reference_img_url = meta.get("referenceUploadUrl")
            if reference_img_url:
                with open(kwargs.get("reference_img_path"), "rb") as f:
                    ref_res = requests.put(
                        reference_img_url,
                        data=f,
                        headers={"Content-Type": self._content_type(kwargs.get("reference_img_path"))},
                        timeout=300,
                    )
                    ref_res.raise_for_status()
        put_resp.raise_for_status()
        return meta

    def wait_for_media(
        self,
        mid: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Poll GET /api/media/{mid} until it reaches a terminal status.

        Terminal statuses: PROCESSED, FAILED, ERROR.
        Raises TimeoutError if 'timeout' seconds elapse without a terminal state.
        """
        start = time.time()
        while True:
            media = self.get_media(mid)
            status = (media.get("status") or "").upper()
            if status in {"PROCESSED", "FAILED", "ERROR"}:
                return media
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timed out waiting for media {mid}, last status={status!r}"
                )
            time.sleep(interval)

    def list_media(self, **params) -> Dict[str, Any]:
        """
        GET /api/media: list media for this client.

        Accepts optional query params (page, pageSize, filters) if the API supports them.
        """
        url = f"{self.base_url}/api/media"
        if self.base_url=="https://platform.authenta.ai":
            resp = cffi_requests.get(url, headers=self._headers(), params=params, impersonate="chrome", timeout=30)
        else:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)

    def process(
        self,
        path: str,
        model_type: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        High-level helper:
          1) upload_file(path, model_type) -> get mid
          2) wait_for_media(mid)
        """
        meta = self.upload_file(path, model_type=model_type)
        mid = meta.get("mid")
        if not mid:
            raise RuntimeError("No 'mid' in upload response")
        return self.wait_for_media(mid, interval=interval, timeout=timeout)
    
    def face_intelligence(
            self,
            path: str,
            model_type: str,
            reference_img_path: Optional[str] = None,
            isSingleFace: Optional[bool] = True,
            faceswapCheck: Optional[bool] = False,
            livenessCheck: Optional[bool] = False,
            faceSimilarityCheck: Optional[bool] = False,
            interval: float = 5.0,
            timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        High-level helper for Face Integrity (FI) model:
          1) upload_file(path, model_type) -> get mid
          2) wait_for_media(mid)
        Args:
            path: Local path to the media file.
            model_type: Detection model type to use, e.g. "AC-1" or "DF-1".
            isSingleFace: Whether to check for a single face.
            faceswapCheck: Whether to check for face swapping.
            livenessCheck: Whether to check for liveness.
            faceSimilarityCheck: Whether to check for face similarity.
            interval: Polling interval in seconds.
            timeout: Timeout in seconds.

        Returns:
            The JSON response from POST /api/media (includes 'mid', 'status', etc.).
        """
        fi_params = {
            "reference_img_path": reference_img_path,
            "isSingleFace": True,
            "faceswapCheck": faceswapCheck,
            "livenessCheck": livenessCheck,
            "faceSimilarityCheck": faceSimilarityCheck,
        }
        if self._content_type(path).startswith("image/") and faceswapCheck == True:
            raise ValueError("faceswapCheck cannot be True for image media")
        if self._content_type(path).startswith("video/") and faceSimilarityCheck == True:
            raise ValueError("faceSimilarityCheck cannot be True for video media")
        if faceSimilarityCheck == True and (not reference_img_path or reference_img_path == None):
            raise ValueError("reference_img_path must be provided if faceSimilarityCheck is True")

        meta = self.upload_file(path, model_type=model_type, **fi_params)
        mid = meta.get("mid")
        if not mid:
            raise RuntimeError("No 'mid' in upload response")
        return self.wait_for_media(mid, interval=interval, timeout=timeout)

    def delete_media(self, mid: str) -> None:
        """DELETE /api/media/{mid}: delete a media record."""
        url = f"{self.base_url}/api/media/{mid}"
        if self.base_url=="https://platform.authenta.ai":
            resp = cffi_requests.delete(url, headers=self._headers(), impersonate="chrome", timeout=30)
        else:
            resp = requests.delete(url, headers=self._headers(), timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
