"""
Face indexing helpers — enrol faces, list subjects, search a face.

Mirrors ``packages/core/src/core/faceauth.ts`` in the React Native SDK. Three
endpoints, all on the same host and API key as the rest of the platform:

    POST /api/v1/facesim/v1/enroll     create a subject + presigned upload URLs
    GET  /api/v1/facesim/v1/subjects   every subject and face
    POST /api/v1/facesim/v1/search     rank enrolled faces against a photo

The functions here do the local work — validating files, deriving names and
content types, uploading to S3. The HTTP calls live on
``AuthentaClient.faceEnroll`` / ``faceSearch`` / ``tenants``.
"""

import mimetypes
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from .authenta_exceptions import ValidationError

# ── Contract limits, matching the React Native SDK ──────────────────────────

BASE = "/api/v1/facesim/v1"

#: Content types the face indexing service can decode.
SUPPORTED_FACE_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")

#: Enrollment accepts between 1 and 10 images per subject.
MIN_ENROLL_IMAGES = 1
MAX_ENROLL_IMAGES = 10

#: Search returns at most 50 ranked faces.
MAX_SEARCH_LIMIT = 50

#: A face has settled once it can no longer change.
TERMINAL_FACE_STATUSES = ("processed", "failed")


def guess_content_type(path: str) -> str:
    """Guess the MIME type of a file from its name."""
    content_type, _ = mimetypes.guess_type(path)
    return (content_type or "application/octet-stream").lower()


def describe_image(image: str) -> Dict[str, str]:
    """
    Build the ``{"name", "contentType"}`` descriptor the enroll endpoint expects.

    The full file name is kept — including the extension — because the server
    uses it to label the face, and the content type is derived from it.

    Raises:
        ValidationError: if the file is missing, unnamed, or an unsupported type.
    """
    if not isinstance(image, str) or not image.strip():
        raise ValidationError("Image path must be a non-empty string.")
    if not os.path.isfile(image):
        raise ValidationError(f"Image file does not exist: {image}")

    name = os.path.basename(image).strip()
    if not name or len(name) > 255:
        raise ValidationError(f'Image name must be 1-255 characters - received: "{name}"')

    content_type = guess_content_type(name)
    if content_type not in SUPPORTED_FACE_IMAGE_TYPES:
        raise ValidationError(
            f'Unsupported image type "{content_type}" for {name}. '
            f"Face indexing accepts {', '.join(SUPPORTED_FACE_IMAGE_TYPES)}."
        )

    return {"name": name, "contentType": content_type}


def image_basename(image: str) -> Dict[str, str]:
    """Backwards-compatible alias of :func:`describe_image`."""
    return describe_image(image)


def clamp_limit(limit: Optional[int]) -> int:
    """Clamp a search limit into the 1-50 range the API accepts."""
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = MAX_SEARCH_LIMIT
    return max(1, min(value, MAX_SEARCH_LIMIT))


class FaceAuth:
    """Local file handling for the face indexing endpoints."""

    def EnrollFace(self, images: List[str]) -> List[Dict[str, str]]:
        """
        Validate the images and build the ``images`` array for ``POST /enroll``.

        Every file is checked before anything is sent, so a bad path fails
        before it can leave a half-built subject on the server.

        Args:
            images: Local paths to 1-10 photos of the same person.

        Returns:
            One ``{"name", "contentType"}`` descriptor per image, in order.
        """
        if not isinstance(images, (list, tuple)):
            raise ValidationError("images must be a list of file paths.")
        if len(images) < MIN_ENROLL_IMAGES or len(images) > MAX_ENROLL_IMAGES:
            raise ValidationError(
                f"Enrollment accepts {MIN_ENROLL_IMAGES}-{MAX_ENROLL_IMAGES} images "
                f"- received {len(images)}."
            )

        return [describe_image(image) for image in images]

    def putSignedUrl(
        self,
        url: str,
        image: str,
        content_type: Optional[str] = None,
    ) -> int:
        """
        PUT one image to its presigned S3 URL.

        The ``Content-Type`` must match the one the enroll response returned, or
        S3 rejects the signature with 403.

        Returns:
            The HTTP status code from S3.
        """
        headers = {"Content-Type": content_type or guess_content_type(image)}
        with open(image, "rb") as handle:
            response = requests.put(url, data=handle, headers=headers, timeout=300)
        return response.status_code

    def prepare_search_image(self, image: str) -> Tuple[str, str]:
        """
        Check a query photo and report how to send it.

        Unsupported formats (HEIC, TIFF, ...) are converted to JPEG when Pillow
        is installed; without Pillow the caller gets a clear error instead of a
        server-side ``invalid_image``.

        Returns:
            ``(path, content_type)`` ready for a multipart upload.
        """
        if not isinstance(image, str) or not image.strip():
            raise ValidationError("A query image is required to search faces.")
        if not os.path.isfile(image):
            raise ValidationError(f"Image file does not exist: {image}")

        content_type = guess_content_type(image)
        if content_type in SUPPORTED_FACE_IMAGE_TYPES:
            return image, content_type

        try:
            from PIL import Image  # optional dependency
        except ImportError:
            raise ValidationError(
                f'Unsupported image type "{content_type}" for {os.path.basename(image)}. '
                f"Face indexing accepts {', '.join(SUPPORTED_FACE_IMAGE_TYPES)} - "
                "install Pillow to convert other formats automatically."
            )

        converted = f"{os.path.splitext(image)[0]}.authenta.jpg"
        with Image.open(image) as handle:
            # Convert to RGB and save: this also bakes any EXIF rotation into
            # the pixels, which is what face detection relies on.
            handle.convert("RGB").save(converted, "JPEG", quality=95)
        return converted, "image/jpeg"

    def search_files(self, image: str) -> Dict[str, Any]:
        """Build the ``files=`` mapping for a multipart search request."""
        path, content_type = self.prepare_search_image(image)
        return {"path": path, "content_type": content_type}
