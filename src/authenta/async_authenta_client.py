"""
example usage:

import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient
from authenta.authenta_exceptions import AuthenticationError

async def main():
    client = AsyncAuthentaClient(
        base_url="https://platform-prod.authenta.ai",
        api_key="...",
    )
    async with client:
        media = await client.process("file_path", model_type="AC-1")
        result = client.get_result(media)
        print(result)

asyncio.run(main())

"""

import asyncio
import os
import time
import mimetypes
from typing import Any, Dict, Optional

import httpx

from .authenta_exceptions import (
    AuthentaError,
    AuthenticationError,
    AuthorizationError,
    QuotaExceededError,
    InsufficientCreditsError,
    ValidationError,
    ServerError,
)


def _raise_for_authenta_error_async(resp: httpx.Response) -> None:
    """
    Async variant: map an Authenta API error response to a rich SDK exception.
    """
    status = resp.status_code
    try:
        data = resp.json()
    except ValueError:
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
    message = data.get("message") or resp.reason_phrase or "Unknown error"
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


def _safe_json_async(resp: httpx.Response) -> Dict[str, Any]:
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


class AsyncAuthentaClient:
    """
    Asynchronous Authenta Python SDK.

    Mirrors AuthentaClient and uses httpx.AsyncClient and async/await.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.auth_enabled = bool(api_key)
        self.timeout = timeout
        self._external_client = client
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncAuthentaClient":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_enabled:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _content_type(self, path: str) -> str:
        filetype, _ = mimetypes.guess_type(path)
        return filetype or "application/octet-stream"

    def get_task_id(self, model_type: str) -> str:
        mapping = {
            "AC-1": "1",
            "AF-1": "2",
            "VF-1": "3",
            "DF-1": "4",
            "FD-1": "5",
            "DI-1": "6",
            "FL-1": "7",
            "FI-1": "8",
        }
        task_id = mapping.get(model_type.upper())
        if not task_id:
            raise ValueError(f"Unknown model_type {model_type!r}. Valid options: {list(mapping.keys())}")
        return task_id

    async def create_media(
        self,
        name: str,
        content_type: str,
        size: int,
        model_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/jobs"

        inputs = {
            "slotName": "original",
            "contentType": content_type,
            "size": size,
            "filename": name,
        }
        payload = {
            "taskTypeId": self.get_task_id(model_type),
            "inputs": [inputs],
        }

        if model_type.upper() == "FI-1":
            fi_params = {
                "isFaceswapCheck": kwargs.get("faceswapCheck"),
                "isLivenessCheck": kwargs.get("livenessCheck"),
                "isSimilarityCheck": kwargs.get("faceSimilarityCheck"),
            }
            payload.update({
                "parameters": {i: j for i, j in fi_params.items()}
            })
            if kwargs.get("reference_path"):
                payload["inputs"].append({
                    "slotName": "reference",
                    "contentType": self._content_type(kwargs.get("reference_path")),
                    "size": os.path.getsize(kwargs.get("reference_path")),
                    "filename": os.path.basename(kwargs.get("reference_path")),
                })

        client = await self._get_client()
        resp = await client.post(url, json=payload, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    async def get_media(self, jobid: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/jobs/{jobid}"
        client = await self._get_client()
        resp = await client.get(url, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    async def upload_file(self, path: str, model_type: str, **kwargs) -> Dict[str, Any]:
        filename = os.path.basename(path)
        content_type = self._content_type(path)
        size = os.path.getsize(path)

        meta = await self.create_media(
            name=filename,
            content_type=content_type,
            size=size,
            model_type=model_type,
            **kwargs,
        )
        upload_url = meta["inputs"][0]["uploadUrl"]
        if not upload_url:
            raise RuntimeError("No uploadUrl in create_media response")

        client = await self._get_client()
        with open(path, "rb") as f:
            put_resp = await client.put(
                upload_url,
                content=f.read(),
                headers={"Content-Type": content_type},
                timeout=300.0,
            )

        if model_type.upper() == "FI-1":
            reference_img_url = meta["inputs"][1]["uploadUrl"] if len(meta["inputs"]) > 1 else None
            if reference_img_url:
                with open(kwargs.get("reference_path"), "rb") as f:
                    ref_resp = await client.put(
                        reference_img_url,
                        content=f.read(),
                        headers={"Content-Type": self._content_type(kwargs.get("reference_path"))},
                        timeout=300.0,
                    )
                    ref_resp.raise_for_status()

        put_resp.raise_for_status()
        return meta

    async def finalize_media(self, jobid: str) -> bool:
        url = f"{self.base_url}/api/v1/jobs/{jobid}/finalize"
        client = await self._get_client()
        resp = await client.post(url, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return resp.status_code == 200

    async def wait_for_media(
        self,
        jobid: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Poll GET /api/v1/jobs/{jobid} until it reaches a terminal status.

        Terminal statuses: COMPLETED, PROCESSED, FAILED, ERROR.
        Raises TimeoutError if 'timeout' seconds elapse without a terminal state.
        """
        start = time.time()
        while True:
            media = await self.get_media(jobid)
            status = (media["status"] or "").upper()
            if status in {"COMPLETED", "PROCESSED", "FAILED", "ERROR"}:
                return media
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timed out waiting for media {jobid}, last status={status!r}"
                )
            await asyncio.sleep(interval)

    async def list_media(self, **params) -> Dict[str, Any]:
        """GET /api/v1/jobs: list jobs for this client."""
        url = f"{self.base_url}/api/v1/jobs"
        client = await self._get_client()
        resp = await client.get(url, headers=self._headers(), params=params)
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    async def process(
        self,
        original_path: str,
        model_type: str,
        reference_path: Optional[str] = None,
        faceswapCheck: Optional[bool] = False,
        livenessCheck: Optional[bool] = False,
        faceSimilarityCheck: Optional[bool] = False,
        auto_polling: bool = True,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        High-level helper:
          1) upload_file(path, model_type) -> get jobid
          2) finalize_media(jobid)
          3) wait_for_media(jobid) (if auto_polling=True)
        """
        if self._content_type(original_path).startswith("image/") and faceswapCheck:
            raise ValueError("faceswapCheck cannot be True for image media")
        if self._content_type(original_path).startswith("video/") and faceSimilarityCheck:
            raise ValueError("faceSimilarityCheck cannot be True for video media")
        if faceSimilarityCheck and not reference_path:
            raise ValueError("reference_path must be provided if faceSimilarityCheck is True")

        fi_params = {
            "reference_path": reference_path,
            "faceswapCheck": faceswapCheck,
            "livenessCheck": livenessCheck,
            "faceSimilarityCheck": faceSimilarityCheck,
        }

        meta = await self.upload_file(original_path, model_type=model_type, **fi_params)
        resp = await self.finalize_media(meta["job"]["id"])
        if not resp:
            raise RuntimeError("Failed to finalize media after upload")

        if not auto_polling:
            return meta

        jobid = meta["job"]["id"]
        if not jobid:
            raise RuntimeError("No 'jobid' in upload response")
        return await self.wait_for_media(jobid, interval=interval, timeout=timeout)

    def get_result(self, media: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch the detection result JSON from the media's artifact downloadUrl.

        Args:
            media: A media dict returned by process() or wait_for_media() —
                   must contain an 'artifacts' key with a downloadUrl.

        Returns:
            Parsed detection result dict.
        """
        result_url = media["artifacts"][-1]["downloadUrl"]
        if not result_url:
            raise ValueError("media dict has no downloadUrl. Ensure processing is complete.")
        resp = httpx.get(result_url, timeout=30)
        if not resp.is_success:
            raise RuntimeError(f"Failed to fetch result: HTTP {resp.status_code}")
        return resp.json()

    async def extract_face_vector(
        self,
        img_path: str,
        auto_polling: bool = True,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        High-level helper for Face Embedding (FE-1):

        1) upload_file(img_path, "FE-1")
        2) wait_for_media(jobid) (if auto_polling=True)
        3) get_result(media) → returns embedding

        Args:
            img_path: Local path to image
            auto_polling: If True, waits for processing and returns result with 'result' key
            interval: Polling interval
            timeout: Max wait time
        """
        content_type = self._content_type(img_path)
        if not content_type.startswith("image/"):
            raise ValueError("FE-1 only supports image input")

        meta = await self.upload_file(img_path, model_type="FE-1")

        if not auto_polling:
            return meta

        jobid = meta["job"]["id"]
        if not jobid:
            raise RuntimeError("No 'jobid' in upload response")

        media = await self.wait_for_media(jobid, interval=interval, timeout=timeout)

        result = self.get_result(media)

        if not isinstance(result, dict) or "embedding" not in result:
            raise RuntimeError("Invalid FE-1 response: 'embedding' key missing")

        media["result"] = result

        return media

    async def delete_media(self, jobid: str) -> None:
        """DELETE /api/v1/jobs/{jobid}: delete a media record."""
        url = f"{self.base_url}/api/v1/jobs/{jobid}"
        client = await self._get_client()
        resp = await client.delete(url, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
