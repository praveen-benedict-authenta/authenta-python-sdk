from __future__ import annotations

from io import BytesIO
from typing import Dict, Any, List, Optional
import os
import mimetypes

import cv2
import requests
from PIL import Image
from pathlib import Path



# -------------------------
# Internal helpers
# -------------------------

def _ensure_dir(path: str) -> None:
    """Create parent directory for a file path if needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# -------------------------
# Heatmap helpers
# -------------------------

# def save_heatmap_image(
#     media: Dict[str, Any],
#     out_path: str,
#     participant_id: int = 0,
# ) -> str:
#     """
#     Download and save the heatmap as an image for one participant.

#     Intended mainly for image model outputs (e.g. AC-1).
#     Returns the output path.
#     """
#     participant = media["participants"][participant_id]
#     heatmap_url = participant.get("heatmap", "")
#     if not heatmap_url:
#         raise RuntimeError("No heatmap URL found for this participant")

#     resp = requests.get(heatmap_url, timeout=30)
#     if resp.status_code == 404:
#         raise RuntimeError(
#             "Heatmap not available (404). The presigned URL may have expired; "
#             "please re-run detection and save the heatmap soon after."
#         )
#     resp.raise_for_status()

#     img = Image.open(BytesIO(resp.content)).convert("RGB")
#     _ensure_dir(out_path)
#     img.save(out_path)
#     return out_path


# def save_heatmap_video(
#     media: Dict[str, Any],
#     out_path: str,
#     participant_id: int = 0,
# ) -> str:
#     """
#     Download and save the heatmap video for one participant.

#     Intended for video model outputs (e.g. DF-1), where `heatmap`
#     is a presigned video URL.
#     Returns the output path.
#     """
#     participant = media["participants"][participant_id]
#     heatmap_url = participant.get("heatmap", "")
#     if not heatmap_url:
#         raise RuntimeError("No heatmap URL found for this participant")

#     resp = requests.get(heatmap_url, stream=True, timeout=60)
#     if resp.status_code == 404:
#         raise RuntimeError(
#             "Heatmap not available (404). The presigned URL may have expired; "
#             "please re-run detection and save the heatmap soon after."
#         )
#     resp.raise_for_status()

#     _ensure_dir(out_path)
#     with open(out_path, "wb") as f:
#         for chunk in resp.iter_content(chunk_size=8192):
#             if chunk:
#                 f.write(chunk)

#     return out_path


# def save_heatmap(
#     media: Dict[str, Any],
#     out_path: str,
#     participant_id: int = 0,
#     model_type: Optional[str] = None,
# ) -> str:
#     """
#     Convenience wrapper that chooses image or video heatmap saver
#     based on model_type (if provided) or URL extension.

#     Returns the output path.
#     """
#     participant = media["participants"][participant_id]
#     heatmap_url = participant.get("heatmap", "")
#     if not heatmap_url:
#         raise RuntimeError("No heatmap URL found for this participant")

#     # Prefer explicit model_type if caller passes it
#     if model_type is not None:
#         mt = model_type.upper()
#         if mt.startswith("AC-"):
#             return save_heatmap_image(media, out_path, participant_id=participant_id)
#         if mt.startswith("DF-"):
#             return save_heatmap_video(media, out_path, participant_id=participant_id)

#     # Fallback: simple heuristic based on URL
#     lower = heatmap_url.lower()
#     if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp")):
#         return save_heatmap_image(media, out_path, participant_id=participant_id)
#     return save_heatmap_video(media, out_path, participant_id=participant_id)

def save_heatmap_image(
    heatmap_url: Dict[str, Any],
    out_path: str,
) -> str:
    """
    Download and save the heatmap as an image.
    Returns the output path.
    """

    resp = requests.get(heatmap_url, timeout=30)
    if resp.status_code == 404:
        raise RuntimeError(
            "Heatmap not available (404)"
        )
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    _ensure_dir(out_path)
    img.save(out_path)
    return out_path

# def save_heatmap_video(
#     media: Dict[str, Any],
#     out_dir: str,
#     base_name: str = "heatmap",
# ) -> List[str]:
#     participants = media.get("participants") or []
#     print(f"found {len(participants)} participants in media")
#     if not participants:
#         raise RuntimeError("No participants found in media for video heatmap")

#     parent = Path(out_dir)
#     parent.mkdir(parents=True, exist_ok=True)

#     outputs: List[str] = []

#     for idx, p in enumerate(participants):
#         heatmap_url = p.get("heatmap")
#         print(f"participant {idx} heatmap URL: {heatmap_url}")
#         if not heatmap_url:
#             print(f"[warn] no heatmap URL for participant {idx}, skipping")
#             continue

#         print(f"downloading participant {idx}…")
#         resp = requests.get(heatmap_url, stream=True, timeout=120)
#         if resp.status_code in (403, 404):
#             print(f"[warn] participant {idx} heatmap returned {resp.status_code}, skipping")
#             continue
#         resp.raise_for_status()

#         content_type = resp.headers.get("Content-Type", "")
#         ext = mimetypes.guess_extension(content_type.split(";")[0]) or ".mp4"

#         dest = parent / f"{base_name}_p{idx}{ext}"
#         with open(dest, "wb") as f:
#             for chunk in resp.iter_content(chunk_size=8192):
#                 if chunk:
#                     f.write(chunk)

#         outputs.append(str(dest))
#         print(f"saved {dest}")

#     return outputs

def save_heatmap(
    media: Dict[str, Any],
    out_path: str,
) -> str:
    """
    wrapper that chooses image or video heatmap saver
    based on model_type (if provided) or media["type"].
    Returns the output path.
    """
    save_path = None
    for artifact in media["artifacts"]:
        save_path = str(out_path+"/"+"heatmap_"+artifact["id"]+".png")
        if artifact["kind"] == "heatmap" and artifact["contentType"] == "image/png":
            save_heatmap_image(artifact["downloadUrl"], save_path)
        # if artifact["kind"] == "heatmap" and artifact["contentType"] in ("video/mp4", "video/webm"):
        #     save_heatmap_video(artifact["downloadUrl"], out_path)
    return save_path



# -------------------------
# BBox
# -------------------------

def draw_bounding_boxes(
    video_path: str,
    sequence_dict: Dict[int, List[Dict[str, Any]]],
    result_video_path: str,
) -> None:
    """
    Original bounding-box renderer.

    sequence_dict:
      frame_index (int) -> list of {
        "data": [x1, y1, x2, y2],
        "class": "real" | "fake",
        "confidence": float in [0,1]
      }
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5)) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _ensure_dir(result_video_path)
    output = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index in sequence_dict:
            for item in sequence_dict[frame_index]:
                bbox = item["data"]
                class_label = item["class"]
                color = (0, 255, 0) if class_label == "real" else (0, 0, 255)
                confidence = round(item["confidence"] * 100, 2)

                # Note: keep original *2 scaling
                xmin, ymin, xmax, ymax = map(int, [b * 2 for b in bbox])

                # Draw the bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                cv2.putText(
                    frame,
                    f"{class_label} {confidence}%",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

        output.write(frame)
        frame_index += 1

    output.release()
    cap.release()


# -------------------------
# Adapter: Authenta -> sequence_dict
# -------------------------

# def authenta_to_sequence_dict(
#     media: Dict[str, Any],
#     participant_id: int = 0,
#     default_class: str = "fake",
#     default_confidence: float = 1.0,
# ) -> Dict[int, List[Dict[str, Any]]]:
#     """
#     Convert Authenta boundingBoxes JSON into sequence_dict format
#     expected by draw_bounding_boxes.
#     """
#     detail_resp = requests.get(media["resultURL"], timeout=30)
#     detail_resp.raise_for_status()
#     detail = detail_resp.json()

#     # detail["boundingBoxes"][str(participant_id)]["boundingBox"]
#     # -> frame_idx(str) -> [x1, y1, x2, y2]
#     bbox_dict = detail["boundingBoxes"][str(participant_id)]["boundingBox"]

#     sequence_dict: Dict[int, List[Dict[str, Any]]] = {}

#     for frame_str, bbox in bbox_dict.items():
#         frame_idx = int(frame_str)
#         item = {
#             "data": bbox,                 # [x1, y1, x2, y2] as-is
#             "class": default_class,       # "fake" or "real"
#             "confidence": default_confidence,
#         }
#         if frame_idx not in sequence_dict:
#             sequence_dict[frame_idx] = []
#         sequence_dict[frame_idx].append(item)

#     return sequence_dict


# def save_bounding_box_video(
#     media: Dict[str, Any],
#     src_video_path: str,
#     out_video_path: str,
#     participant_id: int = 0,
# ) -> str:
#     """
#     Build sequence_dict from Authenta results and call draw_bounding_boxes.

#     Returns the output path.
#     """
#     sequence_dict = authenta_to_sequence_dict(
#         media,
#         participant_id=participant_id,
#         default_class="fake",       # Authenta: participant.fake == True
#         default_confidence=1.0,     # or map from detail JSON if available
#     )
#     draw_bounding_boxes(src_video_path, sequence_dict, out_video_path)
#     return out_video_path

def authenta_to_sequence_dict(
    media: Dict[str, Any],
    default_class: str = "fake",
    default_confidence: float = 1.0,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Convert Authenta boundingBoxes JSON into sequence_dict format
    expected by draw_bounding_boxes.
    
    Supports both FI-1 and DF-1 response structures.
    """
    sequence_dict: Dict[int, List[Dict[str, Any]]] = {}
    
    # Try FI-1 structure first (identityDetails)
    identity_details = media.get("identityDetails")
    if identity_details:
        print(f"[debug] Found identityDetails in media")
        # Use first participant (participant_id = "0")
        participant_id = "0"
        if participant_id in identity_details:
            participant = identity_details[participant_id]
            bbox_dict = participant.get("boundingBox")
            
            if bbox_dict:
                print(f"[debug] Found {len(bbox_dict)} frames with bounding boxes")
                # Get class and confidence from participant data if available
                cls = participant.get("class", default_class)
                conf = participant.get("confidence", default_confidence)
                
                for frame_str, bbox in bbox_dict.items():
                    try:
                        frame_idx = int(frame_str)
                        item = {
                            "data": bbox,
                            "class": cls,
                            "confidence": conf,
                        }
                        sequence_dict.setdefault(frame_idx, []).append(item)
                    except (ValueError, TypeError) as e:
                        print(f"[warn] Failed to parse frame {frame_str}: {e}")
                
                if sequence_dict:
                    return sequence_dict
        else:
            print(f"[debug] Participant {participant_id} not found in identityDetails")
    else:
        print(f"[debug] No identityDetails found, trying DF-1 structure...")
    
    # Try DF-1 structure (fetch from result artifact)
    for artifact in media.get("artifacts", []):
        if artifact.get("kind") == "result":
            result_url = artifact.get("downloadUrl")
            if result_url:
                try:
                    print(f"[debug] Fetching result from artifact...")
                    detail_resp = requests.get(result_url, timeout=30)
                    detail_resp.raise_for_status()
                    detail = detail_resp.json()
                    print(detail)
                    
                    # DF-1: boundingBoxes[participant_id][boundingBox][frame] = bbox
                    bbox_detail = detail.get("identityDetails", {})
                    print(bbox_detail)
                    if bbox_detail:
                        participant_id = "0"  # first participant
                        if participant_id in bbox_detail:
                            bbox_dict = bbox_detail[participant_id].get("boundingBox", {})
                            cls = bbox_detail[participant_id].get("class", default_class)
                            conf = bbox_detail[participant_id].get("confidence", default_confidence)
                            if bbox_dict:
                                print(f"[debug] Found {len(bbox_dict)} frames from DF-1 artifact")
                                for frame_str, bbox in bbox_dict.items():
                                    try:
                                        frame_idx = int(frame_str)
                                        item = {
                                            "data": bbox,
                                            "class": cls,
                                            "confidence": conf,
                                        }
                                        sequence_dict.setdefault(frame_idx, []).append(item)
                                    except (ValueError, TypeError) as e:
                                        print(f"[warn] Failed to parse frame {frame_str}: {e}")
                        
                        if sequence_dict:
                            return sequence_dict
                except Exception as e:
                    print(f"[warn] Failed to fetch result from artifact: {e}")
    
    # If we get here, no bounding box data was found - print debug info
    print(f"[debug] Media top-level keys: {list(media.keys())}")
    if "identityDetails" in media:
        print(f"[debug] identityDetails keys: {list(media.get('identityDetails', {}).keys())}")
    if "artifacts" in media:
        print(f"[debug] Found {len(media['artifacts'])} artifacts")
        for art in media['artifacts']:
            print(f"  - artifact kind: {art.get('kind')}, contentType: {art.get('contentType')}")
    
    raise RuntimeError(
        "No bounding box data found in media. Ensure processing is complete "
        "and the media contains identityDetails (FI-1) or result artifacts (DF-1)."
    )


def save_bounding_box_video(
    media: Dict[str, Any],
    src_video_path: str,
    out_video_path: str,
) -> str:
    """
    Build sequence_dict from Authenta results and call draw_bounding_boxes.
    Returns the output path.
    """
    sequence_dict = authenta_to_sequence_dict(
        media,
        default_class="fake",
        default_confidence=1.0,
    )
    draw_bounding_boxes(src_video_path, sequence_dict, out_video_path)
    return out_video_path

# -------------------------
# Artefact savers
# -------------------------

def save_image_artifacts(
    media: Dict[str, Any],
    out_dir: str,
    base_name: str = "image",
) -> Dict[str, str]:
    """
    Save all visual artefacts for an image (AC-1).
    Args:
        media: AC-1 media JSON returned by AuthentaClient.process(...).
        out_dir: Output directory where artefacts will be saved.
        base_name: Base name for files (default: "image").
    Returns:
        Dict with keys like:
            {
              "heatmap": "<out_dir>/<base_name>_heatmap.jpg"
            }
    """
    out_path = str(Path(out_dir) / f"{base_name}_heatmap.jpg")
    path = save_heatmap_image(media, out_path)
    return {"heatmap": path}


def save_video_artifacts(
    media: Dict[str, Any],
    src_video_path: str,
    out_dir: str,
    base_name: str = "video",
) -> Dict[str, Any]:
    """
    Save all visual artefacts for a deepfake video (DF-1).
    Args:
        media: DF-1 media JSON returned by AuthentaClient.process(...).
        src_video_path: Path to the original input video.
        out_dir: Output directory where artefacts will be saved.
        base_name: Base name for files (default: "video").
    Returns:
        Dict with keys like:
            {
              "heatmaps": [
                  "<out_dir>/<base_name>_heatmap_p0.mp4",
                  "<out_dir>/<base_name>_heatmap_p1.mp4",
                  ...
              ],
              "bbox_video": "<out_dir>/<base_name>_bbox.mp4",
            }
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1) Heatmap videos per participant
    heatmap_base = f"{base_name}_heatmap"
    # heatmap_paths = save_heatmap_video(
    #     media=media,
    #     out_dir=str(out_dir_path),
    #     base_name=heatmap_base,
    # )

    # 2) Bounding-box annotated video
    bbox_path = str(out_dir_path / f"{base_name}_bbox.mp4")
    save_bounding_box_video(
        media=media,
        src_video_path=src_video_path,
        out_video_path=bbox_path,
    )

    return {
        # "heatmap": heatmap_paths,
        "bbox_video": bbox_path,
    }
