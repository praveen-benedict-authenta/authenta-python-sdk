# import os
# import time
# import mimetypes
# from typing import Any, Dict, Optional

# import requests
# from .authenta_exceptions import (
#     AuthentaError,
#     AuthenticationError,
#     AuthorizationError,
#     QuotaExceededError,
#     InsufficientCreditsError,
#     ValidationError,
#     ServerError,
# )

# from .authenta_client import _raise_for_authenta_error,_safe_json

# class FaceIntelligence:
#     def __init__(self, base_url: str, client_id: str, client_secret: str):
#         """
#         Create new Authenta client.

#         Args:
#             base_url: Authenta API base URL, e.g. "https://platform.authenta.ai".
#             client_id: Your Authenta client ID.
#             client_secret: Your Authenta client secret.
#         """
#         self.base_url = base_url.rstrip("/")
#         self.client_id = client_id
#         self.client_secret = client_secret

#     def _headers(self) -> Dict[str, str]:
#         """Return default headers for Authenta API calls."""
#         return {
#             "x-client-id": self.client_id,
#             "x-client-secret": self.client_secret,
#             "Content-Type": "application/json",
#         }

#     def _content_type(self, path: str) -> str:
#         """
#         Guess the MIME type for a file path.

#         Falls back to 'application/octet-stream' if unknown.
#         """
#         filetype, _ = mimetypes.guess_type(path)
#         return filetype or "application/octet-stream"
    
#     def create_media(
#         self,
#         name: str,
#         content_type: str,
#         size: int,
#         model_type: str,
#         **kwargs,
#     ) -> Dict[str, Any]:
#         url = f"{self.base_url}/api/media"
        
#         payload = {
#             "name": name,
#             "contentType": content_type,
#             "size": size,
#             "modelType": model_type,
#         }

        
#         fi_params = {
#             "isSingleFace": kwargs.get("isSingleFace", True),
#             "faceSwapCheck": kwargs.get("faceSwapCheck"),
#             "livenessCheck": kwargs.get("livenessCheck"),
#             "faceSimilarityCheck": kwargs.get("faceSimilarityCheck"),
#         }
#         payload.update({
#             "metadata": {
#                 i: (j if j is not None else False) for i, j in fi_params.items()
#             }
#         })
#         resp = requests.post(url, json=payload, headers=self._headers(), timeout=30)
#         print("create_media raw:", resp.status_code, repr(resp.text[:200]))
#         if not resp.ok:
#             _raise_for_authenta_error(resp)
#         return _safe_json(resp)
    
#     def upload_file(self, path: str, model_type: str, **kwargs) -> Dict[str, Any]:
#         """
#         Upload a file via the two-step Authenta media flow.

#         Steps:
#             1) POST /api/media to create the record and obtain 'mid' + 'uploadUrl'.
#             2) PUT the file bytes to the presigned S3 'uploadUrl'.

#         Args:
#             path: Local path to the media file.
#             model_type: Detection model type to use, e.g. "AC-1" or "DF-1".
#             **kwargs: Additional parameters for the media creation request.

#         Returns:
#             The JSON response from POST /api/media (includes 'mid', 'status', etc.).
#         """
#         filename = os.path.basename(path)
#         content_type = self._content_type(path)
#         size = os.path.getsize(path)

#         # Step 1: create media
#         meta = self.create_media(
#             name=filename,
#             content_type=content_type,
#             size=size,
#             model_type=model_type,
#             **kwargs,
#         )
#         upload_url = meta.get("uploadUrl")
#         if not upload_url:
#             raise RuntimeError("No uploadUrl in create_media response")

#         # Step 2: upload to S3
#         with open(path, "rb") as f:
#             put_resp = requests.put(
#                 upload_url,
#                 data=f,
#                 headers={"Content-Type": content_type},
#                 timeout=300,
#             )
#         if model_type.upper() == "FI-1":
#             reference_img_url = meta.get("referenceUploadUrl")
#             if reference_img_url:
#                 with open(kwargs.get("reference_img_path"), "rb") as f:
#                     ref_put
#         put_resp.raise_for_status()
#         return meta