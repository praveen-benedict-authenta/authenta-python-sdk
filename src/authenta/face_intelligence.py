import mimetypes
import os
from typing import Any, Dict, Optional

import requests

from .authenta_client import (
    _raise_for_authenta_error,
    _safe_json,
)

TASK_TYPE_MAPPING = {
    "FI-1": "8",
}


class FaceIntelligence:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
    ):
        """
        Create Face Intelligence client.

        Args:
            base_url: Authenta API base URL.
            api_key: Optional bearer API key.
        """

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # api_key optional for public task types
        self.auth_enabled = bool(api_key)

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        """
        Return default headers.
        """

        headers = {
            "Content-Type": "application/json",
        }

        if self.auth_enabled:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _content_type(self, path: str) -> str:
        """
        Guess MIME type.
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
    # CREATE JOB
    # ---------------------------------------------------------

    def create_media(
        self,
        name: str,
        content_type: str,
        size: int,
        model_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create media job.
        """

        url = f"{self.base_url}/api/v1/jobs"

        reference_img_path = kwargs.pop(
            "reference_img_path",
            None,
        )

        task_type_id = kwargs.pop(
            "taskTypeId",
            None,
        )

        if not task_type_id:
            task_type_id = self._get_task_type_id(
                model_type
            )

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

        parameters = kwargs.pop("parameters", None) or {}
        for param_key in (
            "isSingleFace",
            "faceSwapCheck",
            "livenessCheck",
            "faceSimilarityCheck",
            "similarityCheck",
            "single_face",
        ):
            if param_key in kwargs:
                parameters[param_key] = kwargs.pop(param_key)

        payload = {
            "taskTypeId": str(task_type_id),
            "inputs": inputs,
        }

        if parameters:
            payload["parameters"] = parameters

        resp = requests.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=30,
        )

        print(
            "create_media raw:",
            resp.status_code,
            repr(resp.text[:200]),
        )

        if not resp.ok:
            _raise_for_authenta_error(resp)

        return _safe_json(resp)

    # ---------------------------------------------------------
    # FILE UPLOAD
    # ---------------------------------------------------------

    def upload_file(
        self,
        path: str,
        model_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Upload media file.
        """

        filename = os.path.basename(path)

        content_type = self._content_type(path)

        size = os.path.getsize(path)

        # Step 1: Create job
        meta = self.create_media(
            name=filename,
            content_type=content_type,
            size=size,
            model_type=model_type,
            **kwargs,
        )

        inputs = meta.get("inputs") or []

        if not inputs:
            raise RuntimeError(
                "No inputs section in create_media response"
            )

        input_map = {
            item.get("slotName") or "original": item
            for item in inputs
        }

        original_input = (
            input_map.get("original")
            or inputs[0]
        )

        upload_url = original_input.get(
            "uploadUrl"
        )

        if not upload_url:
            raise RuntimeError(
                "No uploadUrl in create_media response"
            )

        # Step 2: Upload original file
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

        # Step 3: Upload reference image if exists
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