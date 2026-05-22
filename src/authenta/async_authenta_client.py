"""
example usage:

import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient

async def main():
    client = AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        api_key="api_xxxxxxxx...",  # optional for public task types
    )

    async with client:
        media = await client.process(
            "file_path",
            model_type="AC-1",
        )
        print(media.get("status"))

asyncio.run(main())
"""

import asyncio
import mimetypes
import os
import time
from typing import Any, Dict, Optional

import httpx

from .authenta_exceptions import (
    AuthenticationError,
    AuthorizationError,
    AuthentaError,
    InsufficientCreditsError,
    QuotaExceededError,
    ServerError,
    ValidationError,
)
from .authenta_client import TASK_TYPE_MAPPING

# ---------------------------------------------------------
# ERROR HELPERS
# ---------------------------------------------------------


def _raise_for_authenta_error_async(resp: httpx.Response) -> None:
    status = resp.status_code

    try:
        data = resp.json()
    except ValueError:
        if 400 <= status < 500:
            raise ValidationError(message=resp.text or "Client error", status_code=status)
        if status >= 500:
            raise ServerError(message=resp.text or "Server error", status_code=status)
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


# ---------------------------------------------------------
# CLIENT
# ---------------------------------------------------------


class AsyncAuthentaClient:
    """Asynchronous Authenta Python SDK."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._external_client = client
        self._client: Optional[httpx.AsyncClient] = None
        self.auth_enabled = bool(api_key)

    # ---------------------------------------------------------
    # INTERNALS
    # ---------------------------------------------------------

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

    def _get_task_type_id(self, model_type: str) -> str:
        if model_type not in TASK_TYPE_MAPPING:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported values: {list(TASK_TYPE_MAPPING.keys())}"
            )
        return TASK_TYPE_MAPPING[model_type]

    # ---------------------------------------------------------
    # JOB APIs
    # ---------------------------------------------------------

    async def create_job(
        self,
        name: str,
        content_type: str,
        size: int,
        model_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a job and get presigned upload URL(s)."""
        url = f"{self.base_url}/api/v1/jobs"

        reference_img_path = kwargs.pop("reference_img_path", None)
        task_type_id = kwargs.pop("taskTypeId", None) or self._get_task_type_id(model_type)

        inputs = [
            {
                "slotName": "original",
                "contentType": content_type,
                "fileName": name,
                "sizeBytes": size,
            }
        ]

        if reference_img_path:
            inputs.append({
                "slotName": "reference",
                "contentType": self._content_type(reference_img_path),
                "fileName": os.path.basename(reference_img_path),
                "sizeBytes": os.path.getsize(reference_img_path),
            })

        payload: Dict[str, Any] = {
            "taskTypeId": str(task_type_id),
            "inputs": inputs,
        }

        parameters = kwargs.pop("parameters", None) or {}
        for param_key in (
            "isSingleFace",
            "isFaceswapCheck",
            "isLivenessCheck",
            "isSimilarityCheck",
        ):
            if param_key in kwargs:
                parameters[param_key] = kwargs.pop(param_key)

        if parameters:
            payload["parameters"] = parameters

        payload.update({k: v for k, v in kwargs.items() if v is not None})

        client = await self._get_client()
        resp = await client.post(url, json=payload, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    create_media = create_job

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        """Fetch a single job."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}"
        client = await self._get_client()
        resp = await client.get(url, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        response = _safe_json_async(resp)
        return response.get("job") or response

    get_media = get_job

    async def upload_file(
        self,
        path: str,
        model_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Upload file using the Authenta upload flow (create job → PUT to S3)."""
        filename = os.path.basename(path)
        content_type = self._content_type(path)
        size = os.path.getsize(path)

        meta = await self.create_job(
            name=filename,
            content_type=content_type,
            size=size,
            model_type=model_type,
            **kwargs,
        )

        inputs = meta.get("inputs") or []
        if not inputs:
            raise RuntimeError("No inputs section in create_job response")

        input_map = {
            item.get("slotName") or "original": item
            for item in inputs
        }

        original_input = input_map.get("original") or inputs[0]
        upload_url = original_input.get("uploadUrl")

        if not upload_url:
            raise RuntimeError("No uploadUrl in create_job response")

        client = await self._get_client()
        with open(path, "rb") as f:
            put_resp = await client.put(
                upload_url,
                content=f,
                headers={"Content-Type": content_type},
                timeout=300.0,
            )

        if not put_resp.is_success:
            if 400 <= put_resp.status_code < 500:
                raise ValidationError(
                    message=put_resp.text or "Upload client error",
                    status_code=put_resp.status_code,
                )
            raise ServerError(
                message=put_resp.text or "Upload server error",
                status_code=put_resp.status_code,
            )

        reference_img_path = kwargs.get("reference_img_path")
        if reference_img_path:
            reference_input = input_map.get("reference")
            if not reference_input or not reference_input.get("uploadUrl"):
                raise RuntimeError("No uploadUrl for reference input")
            with open(reference_img_path, "rb") as f:
                ref_resp = await client.put(
                    reference_input["uploadUrl"],
                    content=f,
                    headers={"Content-Type": self._content_type(reference_img_path)},
                    timeout=300.0,
                )
                if not ref_resp.is_success:
                    if 400 <= ref_resp.status_code < 500:
                        raise ValidationError(
                            message=ref_resp.text or "Reference upload error",
                            status_code=ref_resp.status_code,
                        )
                    raise ServerError(
                        message=ref_resp.text or "Reference upload server error",
                        status_code=ref_resp.status_code,
                    )

        return meta

    # ---------------------------------------------------------
    # JOB POLLING
    # ---------------------------------------------------------

    async def wait_for_job(
        self,
        job_id: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """Poll until job reaches a terminal state."""
        terminal_statuses = {
            "COMPLETED", "PROCESSED", "FAILED", "ERROR", "CANCELLED", "CANCELED",
        }
        start = time.time()

        while True:
            response = await self.get_job(job_id)
            job = response.get("job") or response
            status = (job.get("status") or "").upper()

            if status in terminal_statuses:
                return job

            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timed out waiting for job {job_id}, last status={status!r}"
                )
            await asyncio.sleep(interval)

    wait_for_media = wait_for_job

    # ---------------------------------------------------------
    # OTHER APIs
    # ---------------------------------------------------------

    async def list_jobs(self, **params) -> Dict[str, Any]:
        """List jobs."""
        url = f"{self.base_url}/api/v1/jobs"
        client = await self._get_client()
        resp = await client.get(url, headers=self._headers(), params=params)
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    list_media = list_jobs

    async def finalize_job(self, job_id: str) -> Dict[str, Any]:
        """Signal that all S3 uploads are complete."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}/finalize"
        client = await self._get_client()
        resp = await client.post(url, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}/cancel"
        client = await self._get_client()
        resp = await client.post(url, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)
        return _safe_json_async(resp)

    async def delete_job(self, job_id: str) -> None:
        """Delete a job."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}"
        client = await self._get_client()
        resp = await client.delete(url, headers=self._headers())
        if not resp.is_success:
            _raise_for_authenta_error_async(resp)

    delete_media = delete_job

    # ---------------------------------------------------------
    # HIGH LEVEL HELPERS
    # ---------------------------------------------------------

    async def process(
        self,
        path: str,
        model_type: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """High-level helper: upload → finalize → wait for completion."""
        meta = await self.upload_file(path, model_type=model_type)

        job = meta.get("job") or {}
        job_id = str(job.get("id") or meta.get("id") or meta.get("jobId") or "")

        if not job_id:
            raise RuntimeError("No job id in upload response")

        await self.finalize_job(job_id)

        return await self.wait_for_job(job_id, interval=interval, timeout=timeout)

    async def process_FI(
        self,
        path: str,
        model_type: str,
        reference_img_path: Optional[str] = None,
        isSingleFace: Optional[bool] = True,
        isFaceswapCheck: Optional[bool] = False,
        isLivenessCheck: Optional[bool] = False,
        isSimilarityCheck: Optional[bool] = False,
        auto_polling: bool = True,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """High-level async helper for Face Integrity (FI-1 model)."""
        meta = await self.upload_file(
            path,
            model_type=model_type,
            reference_img_path=reference_img_path,
            isSingleFace=isSingleFace,
            isFaceswapCheck=isFaceswapCheck,
            isLivenessCheck=isLivenessCheck,
            isSimilarityCheck=isSimilarityCheck,
        )

        if not auto_polling:
            return meta

        job = meta.get("job") or {}
        job_id = str(job.get("id") or meta.get("id") or meta.get("jobId") or "")

        if not job_id:
            raise RuntimeError("No job id in upload response")

        await self.finalize_job(job_id)

        return await self.wait_for_job(job_id, interval=interval, timeout=timeout)

    async def extract_face_vector(
        self,
        img_path: str,
        auto_polling: bool = True,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """Face embedding helper (FE-1 model)."""
        if not self._content_type(img_path).startswith("image/"):
            raise ValueError("FE-1 only supports image input")

        meta = await self.upload_file(img_path, model_type="FE-1")

        if not auto_polling:
            return meta

        job = meta.get("job") or {}
        job_id = str(job.get("id") or meta.get("id") or meta.get("jobId") or "")

        if not job_id:
            raise RuntimeError("No job id in upload response")

        await self.finalize_job(job_id)

        media = await self.wait_for_job(job_id, interval=interval, timeout=timeout)

        from .authenta_client import AuthentaClient
        _sync = AuthentaClient(base_url=self.base_url, api_key=self.api_key)
        result = _sync.get_result(media)

        if not isinstance(result, dict) or "embedding" not in result:
            raise RuntimeError("Invalid FE-1 response: 'embedding' key missing")

        media["result"] = result
        return media
