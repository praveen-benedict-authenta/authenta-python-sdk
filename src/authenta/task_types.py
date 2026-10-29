"""
Task types, job statuses and result helpers shared by both clients.

A task type is one detection model. The API identifies it by a numeric
``taskTypeId`` (cloud IDs below) and a stable ``slug``. On-prem deployments can
use different IDs, so pass the numeric ID directly there.
"""

from typing import Any, Dict, Optional

#: SDK model code -> (cloud taskTypeId, slug)
TASK_TYPES: Dict[str, tuple] = {
    "AC-1": ("1", "ai-image-detection"),
    "AF-1": ("2", "ai-audio-detection"),
    "VF-1": ("3", "ai-video-detection"),
    "DF-1": ("4", "faceswap-detection"),  # deprecated server-side, use ED-1
    "FD-1": ("5", "image-forgery-detection"),
    "DI-1": ("6", "document-intelligence"),
    "FL-1": ("7", "face-liveness-detection"),
    "FI-1": ("8", "face-intelligence"),
    "FE-1": ("9", "face-embedding"),
    "AS-1": ("10", "audio-splitter"),
    "ED-1": ("11", "ekyc-deepfake"),
    "PDF-1": ("13", "pdf-tampering-detection"),
}

_IDS_BY_SLUG = {slug: task_id for task_id, slug in TASK_TYPES.values()}
_CODES_BY_ID = {task_id: code for code, (task_id, _) in TASK_TYPES.items()}

#: Statuses after which a job no longer changes. COMPLETED/FAILED/CANCELLED are
#: what the API sends; PROCESSED/ERROR are kept for older deployments.
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "PROCESSED", "ERROR"})

#: The four result tiers, from most to least likely real.
RESULT_TAGS = ("real", "likely_real", "likely_fake", "fake")

#: Fields the API adds to `job.result` that the worker's result artifact lacks.
_JOB_RESULT_FIELDS = ("tag", "tagBoundaries", "modelVersion", "fakeConfidencePercent", "realConfidencePercent")


def resolve_task_type_id(model_type: str) -> str:
    """
    Resolve a model code (``"AC-1"``), slug (``"ai-image-detection"``) or
    numeric ID (``"1"``) to the ``taskTypeId`` the API expects.
    """
    value = str(model_type).strip()
    if value.upper() in TASK_TYPES:
        return TASK_TYPES[value.upper()][0]
    if value.lower() in _IDS_BY_SLUG:
        return _IDS_BY_SLUG[value.lower()]
    if value.isdigit():
        return value
    raise ValueError(
        f"Unknown model_type {model_type!r}. Valid options: {list(TASK_TYPES.keys())}, "
        f"a slug such as 'ai-image-detection', or a numeric task type ID."
    )


def model_code(model_type: str) -> Optional[str]:
    """Return the SDK model code (``"FI-1"``) for any accepted ``model_type``."""
    try:
        return _CODES_BY_ID.get(resolve_task_type_id(model_type))
    except ValueError:
        return None


def is_terminal(status: Optional[str]) -> bool:
    return (status or "").upper() in TERMINAL_STATUSES


def merge_job_result(media: Dict[str, Any], artifact_result: Any) -> Any:
    """
    Add the job-level result fields (``tag``, ``tagBoundaries``,
    ``modelVersion``, ...) to the downloaded result artifact.

    Only fields the artifact lacks are added — values already in the artifact
    are never changed.
    """
    job_result = media.get("result")
    if not isinstance(artifact_result, dict) or not isinstance(job_result, dict):
        return artifact_result
    merged = dict(artifact_result)
    for key in _JOB_RESULT_FIELDS:
        if key in job_result and key not in merged:
            merged[key] = job_result[key]
    return merged


def find_result_url(media: Dict[str, Any]) -> str:
    """Return the ``downloadUrl`` of the job's ``result`` artifact."""
    for artifact in media.get("artifacts") or []:
        if artifact.get("kind") == "result" and artifact.get("downloadUrl"):
            return artifact["downloadUrl"]
    raise ValueError(
        "media dict has no 'result' artifact. Ensure processing is complete "
        "(status=completed) and pass the dict from wait_for_media() or get_media()."
    )


def count_pdf_pages(path: str) -> int:
    """
    Count a PDF's pages for ``pageCount``, which pdf-tampering-detection requires.

    Needs ``pypdf`` (``pip install pypdf``); otherwise pass ``page_count``.
    """
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError:
        raise ValueError(
            "PDF-1 requires the PDF's page count. Pass page_count=..., or "
            "install pypdf to count pages automatically."
        )
    return len(PdfReader(path).pages)
