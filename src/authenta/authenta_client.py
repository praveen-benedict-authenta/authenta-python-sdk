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
    error_from_response,
)
from .task_types import (
    count_pdf_pages,
    find_result_url,
    is_terminal,
    merge_job_result,
    model_code,
    resolve_task_type_id,
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

    Expects JSON like: {"rid": "...", "name": "INVALID_API_KEY", "message": "..."}.
    Falls back to HTTP-based mapping if the body is not JSON.
    """
    try:
        body = resp.json()
    except ValueError:
        body = None
    raise error_from_response(
        resp.status_code, body, text=resp.text, reason=resp.reason or "", headers=resp.headers
    )


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
    - Authenticates with ``Authorization: Bearer <api_key>``.
    - Wraps /api/v1/jobs for create, finalize, get, list, cancel, delete.
    - Implements the upload flow (POST /api/v1/jobs -> PUT to S3 -> finalize).
    - Face indexing via /api/v1/facesim.
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
        Map a model code (``"AC-1"``), slug (``"ai-image-detection"``) or
        numeric task type ID to the ``taskTypeId`` the API expects.
        """
        return resolve_task_type_id(model_type)

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
        if kwargs.get("page_count") is not None:
            inputs[0]["pageCount"] = int(kwargs["page_count"])

        payload = {
            "taskTypeId": str(self.get_task_id(model_type)),
            "inputs": inputs,
        }
        parameters = {
            "version": "v1",
        }
        if model_code(model_type) == "FI-1":
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
                    "fileName": os.path.basename(reference_path),
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

    def cancel_media(self, jobid: str) -> Dict[str, Any]:
        """
        POST /api/v1/jobs/{jobid}/cancel: cancel a job that has not started
        processing (status ``initiated`` or ``queued``). Its credits are refunded.

        Returns:
            The job with ``"status": "cancelled"``.
        """
        url = f"{self.base_url}/api/v1/jobs/{jobid}/cancel"
        resp = requests.post(url, headers=self._headers(), timeout=30)
        if not resp.ok:
            _raise_for_authenta_error(resp)
        return _safe_json(resp)

    def upload_file(self, path: str, model_type: str, **kwargs) -> Dict[str, Any]:
        """
        Upload a file via the two-step Authenta media flow.

        Steps:
            1) POST /api/v1/jobs to create the record and obtain 'jobid' + 'uploadUrl'.
            2) PUT the file bytes to the presigned S3 'uploadUrl'.

        Args:
            path: Local path to the media file.
            model_type: Detection model type to use, e.g. "AC-1" or "DF-1".
            page_count: PDFs only — page count, required by PDF-1. Counted
                        automatically when omitted and ``pypdf`` is installed.

        Returns:
            The JSON response from POST /api/v1/jobs (includes 'jobid', 'status', etc.).
        """
        filename = os.path.basename(path)
        content_type = self._content_type(path)
        size = os.path.getsize(path)
        if content_type == "application/pdf" and kwargs.get("page_count") is None:
            kwargs["page_count"] = count_pdf_pages(path)
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

        if model_code(model_type) == "FI-1" and kwargs.get("reference_path") and len(meta["inputs"]) > 1:
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

        Terminal statuses: COMPLETED, FAILED, CANCELLED.
        Raises TimeoutError if 'timeout' seconds elapse without a terminal state.
        """
        start = time.time()
        while True:
            media = self.get_media(jobid)
            status = (media["status"] or "").upper()
            if is_terminal(status):
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
        page_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        High-level helper:
          1) upload_file(path, model_type) -> get jobid
          2) wait_for_media(jobid)

        ``page_count`` is for PDF-1 only; see :meth:`upload_file`.
        """
        if self._content_type(original_path).startswith("image/") and faceswapCheck:
            raise ValueError("faceswapCheck cannot be True for image media")
        if self._content_type(original_path).startswith("video/") and faceSimilarityCheck:
            raise ValueError("faceSimilarityCheck cannot be True for video media")
        if faceSimilarityCheck and not reference_path:
            raise ValueError("reference_path must be provided if faceSimilarityCheck is True")
        if model_code(model_type) == "FI-1":
            fi_params = {
                "reference_path": reference_path,
                "faceswapCheck": faceswapCheck,
                "livenessCheck": livenessCheck,
                "faceSimilarityCheck": faceSimilarityCheck,
            }
        else:
            fi_params = {}
        if page_count is not None:
            fi_params["page_count"] = page_count
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
        Fetch the detection result JSON from the media's ``result`` artifact.

        The artifact is written by the model. The job-level fields the API adds
        on top — ``tag``, ``tagBoundaries`` and ``modelVersion`` — are merged in
        when the artifact lacks them; values already in the artifact are never
        changed. The same fields are also on ``media["result"]`` directly.

        Args:
            media: A media dict returned by process(), wait_for_media() or
                   get_media() — must contain a ``result`` artifact.

        Returns:
            Parsed detection result dict.

        Raises:
            ValueError: If the media dict has no result artifact.
            RuntimeError: If the result download fails.
        """
        result_url = find_result_url(media)
        resp = requests.get(result_url, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"Failed to fetch resultURL: HTTP {resp.status_code}")
        return merge_job_result(media, _safe_json(resp))


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
