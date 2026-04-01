"""
example usage:

import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient
from authenta.authenta_exceptions import AuthenticationError

async def main():
    client = AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        client_id="...",
        client_secret="...",
    )
    async with client:
        media = await client.process("file_path", model_type="AC-1/DF-1")
        print(media.get("status"))

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
        client_id: str,
        client_secret: str,
        *,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
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
        return {
            "x-client-id": self.client_id,
            "x-client-secret": self.client_secret,
            "Content-Type": "application/json",
        }

    def _content_type(self, path: str) -> str:
        filetype, _ = mimetypes.guess_type(path)
        return filetype or "application/octet-stream"

    async def create_media(
        self,
        name: str,
        content_type: str,
        size: int,
        model_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/media"
        payload = {
            "name": name,
            "contentType": content_type,
            "size": size,
            "modelType": model_type,
        }
        
        # Add model-specific parameters to payload
        if model_type == "FI":
            # Face Integrity specific parameters
            fi_params = {
                "isSingleFace": kwargs.get("isSingleFace"),
                "faceSwapCheck": kwargs.get("faceSwapCheck"),
                "livenessCheck": kwargs.get("livenessCheck"),
                "faceSimilarityCheck": kwargs.get("faceSimilarityCheck"),
            }
            # Only add parameters that were explicitly provided (not None)
            payload.update({k: v for k, v in fi_params.items() if v is not None})
        else:
            # For other model types, pass through remaining kwargs
            payload.update({k: v for k, v in kwargs.items() if v is not None})
        
        client = await self._get_client()
        resp = await client.post(url, json=payload, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    async def get_media(self, mid: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/media/{mid}"
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
            **kwargs
        )
        upload_url = meta.get("uploadUrl")
        if not upload_url:
            raise RuntimeError("No uploadUrl in create_media response")

        client = await self._get_client()
        # S3 presigned PUT; usually no JSON error schema
        with open(path, "rb") as f:
            resp = await client.put(
                upload_url,
                content=f,
                headers={"Content-Type": content_type},
                timeout=300.0,
            )
        if not resp.is_success:
            # Map 4xx/5xx to ValidationError/ServerError
            if 400 <= resp.status_code < 500:
                raise ValidationError(
                    message=resp.text or "Upload client error",
                    status_code=resp.status_code,
                )
            if resp.status_code >= 500:
                raise ServerError(
                    message=resp.text or "Upload server error",
                    status_code=resp.status_code,
                )
            resp.raise_for_status()
        return meta

    async def wait_for_media(
        self,
        mid: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Poll GET /api/media/{mid} until it reaches a terminal status.
        Terminal: PROCESSED, FAILED, ERROR.
        """
        start = time.time()
        while True:
            media = await self.get_media(mid)
            status = (media.get("status") or "").upper()
            if status in {"PROCESSED", "FAILED", "ERROR"}:
                return media
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timed out waiting for media {mid}, last status={status!r}"
                )
            await asyncio.sleep(interval)

    async def list_media(self, **params) -> Dict[str, Any]:
        url = f"{self.base_url}/api/media"
        client = await self._get_client()
        resp = await client.get(url, headers=self._headers(), params=params)
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    async def delete_media(self, mid: str) -> None:
        url = f"{self.base_url}/api/media/{mid}"
        client = await self._get_client()
        resp = await client.delete(url, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)

    async def process(
        self,
        path: str,
        model_type: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        High-level async helper:
          1) await upload_file(path, model_type) -> get mid
          2) await wait_for_media(mid)
        """
        meta = await self.upload_file(path, model_type=model_type)
        mid = meta.get("mid")
        if not mid:
            raise RuntimeError("No 'mid' in upload response")
        return await self.wait_for_media(mid, interval=interval, timeout=timeout)
        


    async def process_FI(
        self,
        path: str,
        model_type: str,
        isSingleFace: Optional[bool] = None,
        faceSwapCheck: Optional[bool] = None,
        livenessCheck: Optional[bool] = None,
        faceSimilarityCheck: Optional[bool] = None,
        auto_polling: bool = True,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        High-level async helper for Face Integrity (FI) model:
          1) await upload_file(path, model_type, **fi_params) -> get mid
          2) await wait_for_media(mid) (only if auto_polling=True)

        Args:
            path: Local path to the media file.
            model_type: Detection model type to use (e.g., "FI-1").
            isSingleFace: Whether to check for a single face.
            faceSwapCheck: Whether to check for face swapping.
            livenessCheck: Whether to check for liveness.
            faceSimilarityCheck: Whether to check for face similarity.
            auto_polling: If True (default), awaits until processing completes via
                wait_for_media(). If False, returns immediately after upload with
                the initial media metadata (including 'mid') so you can poll manually.
            interval: Polling interval in seconds (used only when auto_polling=True).
            timeout: Timeout in seconds (used only when auto_polling=True).

        Returns:
            If auto_polling=True: the final processed media JSON.
            If auto_polling=False: the initial upload metadata dict (includes 'mid').
        """
        # Build FI-specific parameters
        fi_params = {
            "isSingleFace": isSingleFace,
            "faceSwapCheck": faceSwapCheck,
            "livenessCheck": livenessCheck,
            "faceSimilarityCheck": faceSimilarityCheck,
        }
        # Remove None values to avoid sending unnecessary params
        fi_params = {k: v for k, v in fi_params.items() if v is not None}

        meta = await self.upload_file(path, model_type=model_type, **fi_params)
        if not auto_polling:
            return meta
        mid = meta.get("mid")
        if not mid:
            raise RuntimeError("No 'mid' in upload response")
        return await self.wait_for_media(mid, interval=interval, timeout=timeout)

    

    async def extract_face_vector(
        self,
        img_path: str,
        auto_polling: bool = True,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        
        content_type = self._content_type(img_path)
        if not content_type.startswith("image/"):
            raise ValueError("FE-1 only supports image input")
        
        meta = await self.upload_file(img_path, model_type="FE-1")
        if not auto_polling:
            return meta
            
        mid = meta.get("mid")
        if not mid:
            raise RuntimeError("No 'mid' in upload response")
            
        media = await self.wait_for_media(mid, interval=interval, timeout=timeout)
        
        from authenta import AuthentaClient
        sync_client = AuthentaClient(
            base_url=self.base_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        
        result = sync_client.get_result(media)
        
        if not isinstance(result, dict) or "embedding" not in result:
            raise RuntimeError("Invalid FE-1 response: 'embedding' key missing")
            
        media["result"] = result
        
        return
