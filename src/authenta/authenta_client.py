"""
example usage:
client = AuthentaClient(
    base_url="https://platform-prod.authenta.ai",
    api_key="authenta_api_key_from_platform",
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
from .face_auth import (
    BASE as FACESIM_BASE,
    MAX_SEARCH_LIMIT,
    FaceAuth,
    clamp_limit,
)

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

    def __init__(self, base_url: str, api_key: str):
        """
        Create new Authenta client.

        Args:
            base_url: Authenta API base URL, e.g. "https://platform.authenta.ai".
            api_key: Your Authenta API key.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.auth_enabled = bool(api_key)

    def _headers(self, content_type: Optional[str] = "application/json") -> Dict[str, str]:
        """
        Return default headers for Authenta API calls.

        Pass ``content_type=None`` for multipart uploads — requests generates
        the boundary itself, and setting the header by hand breaks it.
        """
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        if self.auth_enabled:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _content_type(self, path: str) -> str:
        """
        Guess the MIME type for a file path.

        Falls back to 'application/octet-stream' if unknown.
        """
        filetype, _ = mimetypes.guess_type(path)
        return filetype or "application/octet-stream"


    def get_task_id(self, model_type: str) -> str:
        """
        Map a model_type to a task_id.

        This is a placeholder implementation. In a real implementation, you might
        fetch this mapping from the API or maintain an up-to-date hardcoded dict.
        """
        mapping = {
            "AC-1": "1",
            "AF-1": "2",
            "VF-1": "3",
            "DF-1": "4",
            "FD-1": "5",
            "DI-1": "6",
            "FL-1": "7",
            "FI-1": "8",
            "FE-1": "9",
            "AS-1": "10",
            "ED-1": "11",
            "PDF-1": "13",

        }
        task_id = mapping.get(model_type.upper())
        if not task_id:
            raise ValueError(f"Unknown model_type {model_type!r}. Valid options: {list(mapping.keys())}")
        return task_id

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

        Returns:
            Parsed JSON response containing at least 'jobid' and 'uploadUrl'.
        """
        url = f"{self.base_url}/api/v1/jobs"

        inputs = [{
            "slotName": "original",
            "contentType": content_type,
            "fileName": name,
            "sizeBytes": size,
        }]

        payload = {
            "taskTypeId": str(self.get_task_id(model_type)),
            "inputs": inputs,
        }
        parameters = {
            "version": "v1",
        }
        if model_type.upper() == "FI-1":
            fi_params = {
                "isFaceswapCheck": kwargs.get("faceswapCheck"),
                "isLivenessCheck": kwargs.get("livenessCheck"),
                "isSimilarityCheck": kwargs.get("faceSimilarityCheck"),
            }
            parameters.update({k: v for k, v in fi_params.items() if v is not None})

            if kwargs.get("reference_path"):
                print(f"Adding reference image {kwargs.get('reference_path')}...")
                reference_path = kwargs.get("reference_path")
                payload["inputs"].append({
                    "slotName": "reference",
                    "contentType": self._content_type(reference_path),
                    "sizeBytes": os.path.getsize(reference_path),
                    "fileName": os.path.basename(reference_path).split(".")[0],
                })
        payload["parameters"] = parameters
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)

    def get_media(self, jobid: str) -> Dict[str, Any]:
        """
        GET /api/v1/jobs/{jobid}: fetch a single job record.

        Args:
            jobid: Job ID returned by create_media / upload_file.

        Returns:
            Parsed JSON media record.
        """
        url = f"{self.base_url}/api/v1/jobs/{jobid}"
        resp = requests.get(url, headers=self._headers(), timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)
    
    def finalize_media(self, jobid: str) -> None:
        url = f"{self.base_url}/api/v1/jobs/{jobid}/finalize"
        resp = requests.post(url, headers=self._headers(), timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return resp.status_code == 200

    def upload_file(self, path: str, model_type: str, **kwargs) -> Dict[str, Any]:
        """
        Upload a file via the two-step Authenta media flow.

        Steps:
            1) POST /api/v1/jobs to create the record and obtain 'jobid' + 'uploadUrl'.
            2) PUT the file bytes to the presigned S3 'uploadUrl'.

        Args:
            path: Local path to the media file.
            model_type: Detection model type to use, e.g. "AC-1" or "DF-1".

        Returns:
            The JSON response from POST /api/v1/jobs (includes 'jobid', 'status', etc.).
        """
        filename = os.path.basename(path).split(".")[0]
        content_type = self._content_type(path)
        size = os.path.getsize(path)
        print(f"Creating media record for {filename} (type={content_type}, size={size} bytes) with model {model_type}...")
        meta = self.create_media(
            name=filename,
            content_type=content_type,
            size=size,
            model_type=model_type,
            **kwargs,
        )
        upload_url = meta["inputs"][0]["uploadUrl"]
        if not upload_url:
            raise RuntimeError("No uploadUrl in create_media response")
        print(f"Uploading {filename} to S3...")
        with open(path, "rb") as f:
            put_resp = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": content_type},
                timeout=300,
            )

        if model_type.upper() == "FI-1" and kwargs.get("reference_path") and len(meta["inputs"]) > 1:
            reference_img_url = meta["inputs"][1]["uploadUrl"]
            print(f"Uploading reference image {kwargs.get('reference_path')}...")
            if reference_img_url:
                with open(kwargs.get("reference_path"), "rb") as f:
                    ref_res = requests.put(
                        reference_img_url,
                        data=f,
                        headers={"Content-Type": self._content_type(kwargs.get("reference_path"))},
                        timeout=300,
                    )
                    ref_res.raise_for_status()

        finalize_media = self.finalize_media(meta["job"]["id"])

        if not finalize_media:
            raise RuntimeError("Failed to finalize media after upload")
        put_resp.raise_for_status()
        return meta



    def wait_for_media(
        self,
        jobid: str,
        interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Poll GET /api/v1/jobs/{jobid} until it reaches a terminal status.

        Terminal statuses: PROCESSED, FAILED, ERROR.
        Raises TimeoutError if 'timeout' seconds elapse without a terminal state.
        """
        start = time.time()
        while True:
            media = self.get_media(jobid)
            status = (media["status"] or "").upper()
            if status in {"COMPLETED", "PROCESSED", "FAILED", "ERROR"}:
                return media
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timed out waiting for media {jobid}, last status={status!r}"
                )
            time.sleep(interval)

    def list_media(self, **params) -> Dict[str, Any]:
        """
        GET /api/v1/jobs: list jobs for this client.

        Accepts optional query params (page, pageSize, filters) if the API supports them.
        """
        url = f"{self.base_url}/api/v1/jobs"
        resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)

    def process(
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
          2) wait_for_media(jobid)
        """
        if self._content_type(original_path).startswith("image/") and faceswapCheck:
            raise ValueError("faceswapCheck cannot be True for image media")
        if self._content_type(original_path).startswith("video/") and faceSimilarityCheck:
            raise ValueError("faceSimilarityCheck cannot be True for video media")
        if faceSimilarityCheck and not reference_path:
            raise ValueError("reference_path must be provided if faceSimilarityCheck is True")
        if model_type.upper() == "FI-1":
            fi_params = {
                "reference_path": reference_path,
                "faceswapCheck": faceswapCheck,
                "livenessCheck": livenessCheck,
                "faceSimilarityCheck": faceSimilarityCheck,
            }
        else:
            fi_params = {}
        meta = self.upload_file(original_path, model_type=model_type, **fi_params)
        
        if not auto_polling:
            return meta
        jobid = meta["job"]["id"]
        if not jobid:
            raise RuntimeError("No 'jobid' in upload response")
        media = self.wait_for_media(jobid, interval=interval, timeout=timeout)
        
        return media


    # ── Face indexing ──────────────────────────────────────────────────────
    # Three endpoints on this same host and API key. See face_auth.py.

    def faceEnroll(self, images: list) -> Dict[str, Any]:
        """
        Enrol photos of one person: create the subject, then upload each image.

        Returns as soon as S3 has the bytes. Embeddings are generated out of
        band, so call :meth:`tenants` afterwards to watch each face reach
        ``processed``.

        Args:
            images: Local paths to 1-10 photos of the same person. Only
                    JPEG, PNG, and WebP are accepted.

        Returns:
            The enroll response, with each face's ``status`` updated to
            ``uploaded`` or ``failed`` to reflect its own upload outcome.
        """
        face_auth = FaceAuth()
        described_images = face_auth.EnrollFace(images)

        resp = requests.post(
            f"{self.base_url}{FACESIM_BASE}/enroll",
            json={"images": described_images},
            headers=self._headers(),
            timeout=30,
        )
        if not resp.ok:
            _raise_for_authenta_error(resp)
        enrollment = _safe_json(resp)

        faces = enrollment.get("faces", [])
        if len(faces) != len(images):
            raise RuntimeError(
                f"Enrollment returned {len(faces)} upload URLs for {len(images)} images"
            )

        # The response preserves the order of `images`. One failed PUT marks
        # only its own face, rather than sinking the whole batch.
        for face, image, described in zip(faces, images, described_images):
            upload_url = face.get("upload_url")
            if not upload_url:
                face["status"] = "failed"
                continue

            content_type = (face.get("headers") or {}).get(
                "Content-Type", described["contentType"]
            )
            print(f"Uploading {os.path.basename(image)}...")
            status = face_auth.putSignedUrl(upload_url, image, content_type)
            face["status"] = "uploaded" if 200 <= status < 300 else "failed"

        return enrollment

    def tenants(self) -> Dict[str, Any]:
        """
        GET /api/v1/facesim/v1/subjects: every subject and face on the account.

        Use it to follow enrolment progress — a face is searchable once its
        status reaches ``processed``.
        """
        resp = requests.get(
            f"{self.base_url}{FACESIM_BASE}/tenant",
            headers=self._headers(),
            timeout=30,
        )
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)

    def faceSearch(
        self,
        image: str,
        limit: int = MAX_SEARCH_LIMIT,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """
        POST /api/v1/facesim/v1/search: rank enrolled faces against a photo.

        The image is posted as ``multipart/form-data`` rather than Base64 in a
        JSON body, which keeps it clear of the server's 100 KiB JSON limit.

        Search is independent of enrolment: it matches everything already
        indexed on the account, including from earlier sessions. The same
        subject can appear more than once because every enrolled face has its
        own embedding.

        Args:
            image: Local path to the query photo.
            limit: How many matches to return, clamped to 1-50.
            timeout: Seconds to wait — embedding a face takes longer than a
                     plain request.

        Returns:
            ``{"tenant_id", "count", "results"}`` with results ordered from
            highest similarity down.
        """
        face_auth = FaceAuth()
        path, content_type = face_auth.prepare_search_image(image)
        capped = clamp_limit(limit)

        print(f"Searching faces with {os.path.basename(path)} (limit {capped})...")
        with open(path, "rb") as handle:
            resp = requests.post(
                f"{self.base_url}{FACESIM_BASE}/search",
                # No Content-Type header here: requests sets the multipart
                # boundary itself, and overriding it breaks the upload.
                headers=self._headers(content_type=None),
                files={"image": (os.path.basename(path), handle, content_type)},
                data={"limit": str(capped)},
                timeout=timeout,
            )
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)

    def get_result(self, media: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch the detection result JSON from the media's resultURL.

        The resultURL is a presigned S3 URL returned after processing.
        It contains the actual detection output (e.g. isLiveness, isDeepFake,
        isSimilar, similarityScore, etc.).

        Args:
            media: A media dict returned by face_intelligence(), process(), or
                   wait_for_media() — must contain a 'resultURL' key.

        Returns:
            Parsed detection result dict from resultURL.

        Raises:
            ValueError: If the media dict has no resultURL.
            RuntimeError: If the resultURL fetch fails.
        """
        for artifact in media["artifacts"]:
            if artifact["kind"] == "result":
                result_url = artifact["downloadUrl"]
                break
        if not result_url:
            raise ValueError("media dict has no 'resultURL'. Ensure processing is complete (status=PROCESSED).")
        resp = requests.get(result_url, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"Failed to fetch resultURL: HTTP {resp.status_code}")
        return _safe_json(resp)


    def extract_face_vector(
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
            auto_polling: If True, waits for processing and returns result
            interval: Polling interval
            timeout: Max wait time
            
        Returns:
            If auto_polling=True:
               media dict with 'result' containing embedding
            Else:
               upload metadata
        """
        content_type = self._content_type(img_path)
        if not content_type.startswith("image/"):
            raise ValueError("FE-1 only supports image input")
        
        meta = self.upload_file(img_path, model_type="FE-1")
        
        if not auto_polling:
            return meta
            
        jobid = meta["job"]["id"]
        if not jobid:
            raise RuntimeError("No 'jobid' in upload response")
            
        media = self.wait_for_media(jobid, interval=interval, timeout=timeout)
        
        result = self.get_result(media)
        
        return result
    

    def delete_media(self, jobid: str) -> None:
        """DELETE /api/v1/jobs/{jobid}: delete a media record."""
        url = f"{self.base_url}/api/v1/jobs/{jobid}"
        resp = requests.delete(url, headers=self._headers(), timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
