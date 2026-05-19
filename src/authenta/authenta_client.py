"""
example usage:

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",  # optional for public task types
)

media = client.process(
    "image.jpg",
    model_type="AC-1",
)

print(media)
"""

import mimetypes
import os
import time
from typing import Any, Dict, Optional

import requests

from .authenta_exceptions import (
    AuthenticationError,
    AuthorizationError,
    AuthentaError,
    InsufficientCreditsError,
    QuotaExceededError,
    ServerError,
    ValidationError,
)

# ---------------------------------------------------------
# MODEL TYPE -> TASK TYPE ID MAPPING
# ---------------------------------------------------------

TASK_TYPE_MAPPING = {
    "AC-1": "1",
    "DF-1": "4",
    "FI-1": "8",
    "FE-1": "9",
}


# ---------------------------------------------------------
# ERROR HELPERS
# ---------------------------------------------------------


def _raise_for_authenta_error(resp: requests.Response) -> None:
    """
    Map an Authenta API error response to a rich SDK exception.
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
    message = data.get("message") or resp.reason or "Unknown error"
    details = data

    if code == "IAM001":
        raise AuthenticationError(
            message,
            status_code=status,
            details=details,
        )

    if code == "IAM002":
        raise AuthorizationError(
            message,
            status_code=status,
            details=details,
        )

    if code == "AA001":
        raise QuotaExceededError(
            message,
            status_code=status,
            details=details,
        )

    if code == "U007":
        raise InsufficientCreditsError(
            message,
            status_code=status,
            details=details,
        )

    if 400 <= status < 500:
        raise ValidationError(
            message,
            code=code,
            status_code=status,
            details=details,
        )

    if status >= 500:
        raise ServerError(
            message,
            code=code,
            status_code=status,
            details=details,
        )

    raise AuthentaError(
        message,
        code=code,
        status_code=status,
        details=details,
    )


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    """
    Safely parse JSON.
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


# ---------------------------------------------------------
# CLIENT
# ---------------------------------------------------------


class AuthentaClient:
    """
    Authenta Python SDK.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # api_key optional for public task types
        self.auth_enabled = bool(api_key)

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        """
        Return default headers for Authenta API calls.
        """

        headers = {
            "Content-Type": "application/json",
        }

        if self.auth_enabled:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _content_type(self, path: str) -> str:
        """
        Guess file MIME type.
        """

        filetype, _ = mimetypes.guess_type(path)

        return filetype or "application/octet-stream"

    def _get_task_type_id(self, model_type: str) -> str:
        """
        Maps legacy model types to taskTypeId.
        """

        if model_type not in TASK_TYPE_MAPPING:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported values: {list(TASK_TYPE_MAPPING.keys())}"
            )

        return TASK_TYPE_MAPPING[model_type]

    # ---------------------------------------------------------
    # JOB APIs
    # ---------------------------------------------------------

    def create_job(
        self,
        name: str,
        content_type: str,
        size: int,
        model_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a job and get upload URLs.
        """

        url = f"{self.base_url}/api/v1/jobs"

        reference_img_path = kwargs.pop(
            "reference_img_path",
            None,
        )

        # Explicit override allowed
        task_type_id = kwargs.pop(
            "taskTypeId",
            None,
        )

        # Backward compatibility support
        if not task_type_id:
            task_type_id = self._get_task_type_id(model_type)

        inputs = [
            {
                "slotName": "original",
                "contentType": content_type,
                "fileName": name,
                "sizeBytes": size,
            }
        ]

        if reference_img_path:
            ref_content_type = self._content_type(
                reference_img_path
            )

            ref_size = os.path.getsize(
                reference_img_path
            )

            inputs.append(
                {
                    "slotName": "reference",
                    "contentType": ref_content_type,
                    "fileName": os.path.basename(
                        reference_img_path
                    ),
                    "sizeBytes": ref_size,
                }
            )

        payload = {
            "taskTypeId": str(task_type_id),
            "inputs": inputs,
        }

        payload.update(
            {
                k: v
                for k, v in kwargs.items()
                if v is not None
            }
        )

        resp = requests.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=30,
        )

        if not resp.ok:
            _raise_for_authenta_error(resp)

        return _safe_json(resp)

    create_media = create_job

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Fetch a single job.
        """

        url = f"{self.base_url}/api/v1/jobs/{job_id}"

        resp = requests.get(
            url,
            headers=self._headers(),
            timeout=30,
        )

        if not resp.ok:
            _raise_for_authenta_error(resp)

        return _safe_json(resp)

    get_media = get_job

    def upload_file(
        self,
        path: str,
        model_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Upload file using Authenta upload flow.
        """

        filename = os.path.basename(path)

        content_type = self._content_type(path)

        size = os.path.getsize(path)

        meta = self.create_job(
            name=filename,
            content_type=content_type,
            size=size,
            model_type=model_type,
            **kwargs,
        )

        inputs = meta.get("inputs") or []

        if not inputs:
            raise RuntimeError(
                "No inputs section in create_job response"
            )

        input_map = {
            item.get("slotName") or "original": item
            for item in inputs
        }

        original_input = (
            input_map.get("original") or inputs[0]
        )

        upload_url = original_input.get("uploadUrl")

        if not upload_url:
            raise RuntimeError(
                "No uploadUrl in create_job response"
            )

        with open(path, "rb") as f:
            put_resp = requests.put(
                upload_url,
                data=f,
                headers={
                    "Content-Type": content_type,
                },
                timeout=300,
            )

            put_resp.raise_for_status()

        reference_img_path = kwargs.get(
            "reference_img_path"
        )

        if reference_img_path:
            reference_input = input_map.get(
                "reference"
            )

            if (
                not reference_input
                or not reference_input.get(
                    "uploadUrl"
                )
            ):
                raise RuntimeError(
                    "No uploadUrl for reference input"
                )

            with open(reference_img_path, "rb") as f:
                ref_resp = requests.put(
                    reference_input["uploadUrl"],
                    data=f,
                    headers={
                        "Content-Type": self._content_type(
                            reference_img_path
                        )
                    },
                    timeout=300,
                )

                ref_resp.raise_for_status()

        return meta

    # ---------------------------------------------------------
    # JOB POLLING
    # ---------------------------------------------------------

    def wait_for_job(
        self,
        job_id: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Poll until job reaches terminal state.
        """

        terminal_statuses = {
            "COMPLETED",
            "PROCESSED",
            "FAILED",
            "ERROR",
            "CANCELLED",
            "CANCELED",
        }

        start = time.time()

        while True:
            response = self.get_job(job_id)

            job = response.get("job") or response

            status = (
                job.get("status") or ""
            ).upper()

            if status in terminal_statuses:
                return response

            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timed out waiting for job "
                    f"{job_id}, last status={status!r}"
                )

            time.sleep(interval)

    wait_for_media = wait_for_job

    # ---------------------------------------------------------
    # OTHER APIs
    # ---------------------------------------------------------

    def list_jobs(
        self,
        **params,
    ) -> Dict[str, Any]:
        """
        List jobs.
        """

        url = f"{self.base_url}/api/v1/jobs"

        resp = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=30,
        )

        if not resp.ok:
            _raise_for_authenta_error(resp)

        return _safe_json(resp)

    list_media = list_jobs

    def finalize_job(
        self,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Finalize a job.
        """

        url = (
            f"{self.base_url}"
            f"/api/v1/jobs/{job_id}/finalize"
        )

        resp = requests.post(
            url,
            headers=self._headers(),
            timeout=30,
        )

        if not resp.ok:
            _raise_for_authenta_error(resp)

        return _safe_json(resp)

    def cancel_job(
        self,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Cancel a job.
        """

        url = (
            f"{self.base_url}"
            f"/api/v1/jobs/{job_id}/cancel"
        )

        resp = requests.post(
            url,
            headers=self._headers(),
            timeout=30,
        )

        if not resp.ok:
            _raise_for_authenta_error(resp)

        return _safe_json(resp)

    def delete_job(
        self,
        job_id: str,
    ) -> None:
        """
        Delete a job.
        """

        url = (
            f"{self.base_url}"
            f"/api/v1/jobs/{job_id}"
        )

        resp = requests.delete(
            url,
            headers=self._headers(),
            timeout=30,
        )

        if not resp.ok:
            _raise_for_authenta_error(resp)

    delete_media = delete_job

    # ---------------------------------------------------------
    # HIGH LEVEL HELPERS
    # ---------------------------------------------------------

    def process(
        self,
        path: str,
        model_type: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        High-level helper:
            1. Upload file
            2. Finalize job
            3. Wait for completion
        """

        meta = self.upload_file(
            path,
            model_type=model_type,
        )

        job = meta.get("job") or {}

        job_id = str(
            job.get("id")
            or meta.get("id")
            or meta.get("jobId")
            or ""
        )

        if not job_id:
            raise RuntimeError(
                "No job id in upload response"
            )

        self.finalize_job(job_id)

        return self.wait_for_job(
            job_id,
            interval=interval,
            timeout=timeout,
        )

    # ---------------------------------------------------------
    # RESULT HELPERS
    # ---------------------------------------------------------

    def get_result(
        self,
        media: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fetch result JSON from resultURL.
        """

        if (
            isinstance(media, dict)
            and "result" in media
            and media["result"] is not None
        ):
            return media["result"]

        if (
            isinstance(media, dict)
            and "job" in media
        ):
            media = media["job"]

        result_url = (
            media.get("resultURL")
            or media.get("resultUrl")
        )

        if not result_url:
            raise ValueError(
                "media dict has no resultURL"
            )

        resp = requests.get(
            result_url,
            timeout=30,
        )

        if not resp.ok:
            raise RuntimeError(
                f"Failed to fetch resultURL: "
                f"HTTP {resp.status_code}"
            )

        return resp.json()

    # ---------------------------------------------------------
    # FACE INTELLIGENCE
    # ---------------------------------------------------------

    def face_intelligence(
        self,
        path: str,
        model_type: str,
        reference_img_path: Optional[str] = None,
        isSingleFace: Optional[bool] = True,
        faceswapCheck: Optional[bool] = False,
        livenessCheck: Optional[bool] = False,
        faceSimilarityCheck: Optional[bool] = False,
        auto_polling: bool = True,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Face intelligence helper.
        """

        if (
            self._content_type(path).startswith("image/")
            and faceswapCheck
        ):
            raise ValueError(
                "faceswapCheck cannot be True for image media"
            )

        if (
            self._content_type(path).startswith("video/")
            and faceSimilarityCheck
        ):
            raise ValueError(
                "faceSimilarityCheck cannot be True for video media"
            )

        if (
            faceSimilarityCheck
            and not reference_img_path
        ):
            raise ValueError(
                "reference_img_path required when "
                "faceSimilarityCheck=True"
            )

        fi_params = {
            "reference_img_path": reference_img_path,
            "isSingleFace": isSingleFace,
            "faceswapCheck": faceswapCheck,
            "livenessCheck": livenessCheck,
            "faceSimilarityCheck": faceSimilarityCheck,
        }

        meta = self.upload_file(
            path,
            model_type=model_type,
            reference_img_path=reference_img_path,
            **fi_params,
        )

        if not auto_polling:
            return meta

        job = meta.get("job") or {}

        job_id = str(
            job.get("id")
            or meta.get("id")
            or meta.get("jobId")
            or ""
        )

        if not job_id:
            raise RuntimeError(
                "No job id in upload response"
            )

        self.finalize_job(job_id)

        media = self.wait_for_job(
            job_id,
            interval=interval,
            timeout=timeout,
        )

        media["result"] = self.get_result(media)

        return media

    # ---------------------------------------------------------
    # FACE EMBEDDING
    # ---------------------------------------------------------

    def extract_face_vector(
        self,
        img_path: str,
        auto_polling: bool = True,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Face embedding helper.
        """

        content_type = self._content_type(
            img_path
        )

        if not content_type.startswith("image/"):
            raise ValueError(
                "FE-1 only supports image input"
            )

        meta = self.upload_file(
            img_path,
            model_type="FE-1",
        )

        if not auto_polling:
            return meta

        job = meta.get("job") or {}

        job_id = str(
            job.get("id")
            or meta.get("id")
            or meta.get("jobId")
            or ""
        )

        if not job_id:
            raise RuntimeError(
                "No job id in upload response"
            )

        self.finalize_job(job_id)

        media = self.wait_for_job(
            job_id,
            interval=interval,
            timeout=timeout,
        )

        result = self.get_result(media)

        if (
            not isinstance(result, dict)
            or "embedding" not in result
        ):
            raise RuntimeError(
                "Invalid FE-1 response: "
                "'embedding' key missing"
            )

        media["result"] = result

        return media