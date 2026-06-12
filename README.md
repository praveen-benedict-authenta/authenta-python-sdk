# Authenta Python SDK

Welcome to the official documentation for the **Authenta Python SDK** — your gateway to state-of-the-art deepfake detection, AI-image analysis, and face intelligence.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Models & Capabilities](#2-models--capabilities)
3. [Quick Start](#3-quick-start)
4. [Services](#4-services)
   - [4.1 AC-1 — AI-Generated Image Detection](#41-ac-1--ai-generated-image-detection)
   - [4.2 DF-1 — Deepfake Video Detection](#42-df-1--deepfake-video-detection)
   - [4.3 FI-1 — Face Intelligence](#43-fi-1--face-intelligence)
   - [4.4 FE-1 — Face Embedding](#44-fe-1--face-embedding)
   - [4.5 Media Management](#45-media-management)
5. [Visualization](#5-visualization)
6. [Error Handling](#6-error-handling)
7. [API Reference](#7-api-reference)

---

## 1. Getting Started

### Installation

**Option A: Install from PyPI (Recommended)**

```bash
pip install authentasdk
```

**Option B: Local Development**

```bash
git clone https://github.com/phospheneai/authenta-python-sdk.git
cd authenta-python-sdk
pip install -e .
```

### Authentication & Initialization

**Synchronous Client**

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)
```

**Asynchronous Client**

```python
import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient

async def main():
    async with AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        api_key="api_xxxxxxxx...",
    ) as client:
        # use client here
        pass

asyncio.run(main())
```

> The async client is a context manager (`async with`) that automatically manages the underlying HTTP session. You can also call `await client.aclose()` manually if you prefer.

---

## 1.1 Why Use the Async Client?

The SDK ships two clients that are functionally identical — the difference is in how they handle waiting.

### Synchronous client (`AuthentaClient`)

The sync client blocks the calling thread while it waits for processing to complete. This is the right choice when:

- You are writing a **script**, a CLI tool, or a Jupyter notebook.
- Your workload is **sequential** — one file at a time, results needed before moving on.
- You are not running inside an async framework (FastAPI, aiohttp, etc.).

```python
# Blocks here until the result is ready
media = client.process("photo.jpg", model_type="AC-1")
print(media["fake"])
```

### Async client (`AsyncAuthentaClient`)

The async client never blocks the event loop. While it is waiting for the API to finish processing, your application can continue doing other work. This is the right choice when:

- You are building a **web server** (FastAPI, Starlette, aiohttp) and need to keep handling other requests while waiting for results.
- You want to run **multiple detections in parallel** without spawning threads.
- You are already writing `async/await` code.

```python
# Submits both jobs concurrently — total wait ≈ max(t1, t2), not t1 + t2
results = await asyncio.gather(
    client.process("photo1.jpg", model_type="AC-1"),
    client.process("video1.mp4", model_type="DF-1"),
)
```

### Comparison

| | `AuthentaClient` | `AsyncAuthentaClient` |
| :-- | :-- | :-- |
| Blocks the thread while polling | Yes | No |
| Works without an event loop | Yes | No — needs `asyncio` |
| Concurrent requests | No (sequential) | Yes — via `asyncio.gather` |
| Best for | Scripts, notebooks, CLIs | Web servers, async apps |
| Import | `from authenta import AuthentaClient` | `from authenta.async_authenta_client import AsyncAuthentaClient` |

> **Rule of thumb:** if you're not sure which to use, start with the sync client. Switch to async when you need to serve multiple users simultaneously or run detections in parallel.

---

## 2. Models & Capabilities

| Model | Modality | Capability |
| :-- | :-- | :-- |
| `AC-1` | Image | Detects AI-generated or manipulated images (Midjourney, Stable Diffusion, Photoshop, etc.) |
| `AF-1` | Audio | Detects AI-generated or synthetically cloned audio |
| `VF-1` | Video | Detects AI-generated video content |
| `DF-1` | Video | Detects deepfake videos — face swaps, reenactments, and facial manipulations |
| `FD-1` | Image | Detects forged or manipulated regions in images |
| `FL-1` | Image / Video | Face liveness detection — identifies real faces vs. presentation attacks |
| `FI-1` | Image / Video | Face Intelligence — liveness detection, face swap detection, face similarity comparison |
| `FE-1` | Image | Extracts 512D face embedding for recognition and matching |
| `DI-1` | Image / PDF | Detects forged or tampered content in documents |

---

## 3. Quick Start

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

# Detect AI-generated image (blocks until result is ready)
media = client.process("photo.jpg", model_type="AC-1")
print(f"Status : {media['status']}")
print(f"Is Fake: {media.get('fake')}")
```

---

## 4. Services

### 4.1 AC-1 — AI-Generated Image Detection

Identify whether an image was created by generative AI or manipulated with editing tools.

#### Synchronous

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

# One-call: upload + wait for result
media = client.process("samples/photo.jpg", model_type="AC-1")

print(f"Job ID   : {media['id']}")
print(f"Status   : {media['status']}")
print(f"Is Fake  : {media.get('fake')}")
print(f"Result   : {media.get('resultURL')}")
print(f"Heatmap  : {media.get('heatmapURL')}")
```

**Two-step (upload now, poll later)**

```python
# Step 1 — upload
upload_meta = client.upload_file("samples/photo.jpg", model_type="AC-1")
job_id = upload_meta["job"]["id"]
print(f"Uploaded. Job ID: {job_id}")

# ... do other work ...

# Step 2 — wait for result
media = client.wait_for_job(job_id)
print(f"Status : {media['status']}")
print(f"Is Fake: {media.get('fake')}")
```

#### Asynchronous

```python
import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient

async def detect_image():
    async with AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        api_key="api_xxxxxxxx...",
    ) as client:
        # One-call: upload + wait
        media = await client.process("samples/photo.jpg", model_type="AC-1")
        print(f"Status : {media['status']}")
        print(f"Is Fake: {media.get('fake')}")

asyncio.run(detect_image())
```

**Two-step async (upload now, poll later)**

```python
async def detect_image_async():
    async with AsyncAuthentaClient(...) as client:
        # Step 1 — upload
        upload_meta = await client.upload_file("samples/photo.jpg", model_type="AC-1")
        job_id = upload_meta["job"]["id"]

        # Step 2 — poll when ready
        media = await client.wait_for_job(job_id)
        print(f"Status : {media['status']}")
        print(f"Is Fake: {media.get('fake')}")

asyncio.run(detect_image_async())
```

---

### 4.2 DF-1 — Deepfake Video Detection

Detect face swaps, reenactments, and other facial manipulations in video content.

#### Synchronous

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

# One-call: upload + wait for result
media = client.process("samples/video.mp4", model_type="DF-1")

print(f"Job ID      : {media['id']}")
print(f"Status      : {media['status']}")
print(f"Is Fake     : {media.get('fake')}")
print(f"Participants: {len(media.get('participants', []))}")
```

**Two-step**

```python
# Step 1 — upload
upload_meta = client.upload_file("samples/video.mp4", model_type="DF-1")
job_id = upload_meta["job"]["id"]

# Step 2 — poll with custom interval/timeout
media = client.wait_for_job(job_id, interval=10.0, timeout=900.0)
print(f"Status : {media['status']}")
print(f"Is Fake: {media.get('fake')}")
```

#### Asynchronous

```python
import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient

async def detect_deepfake():
    async with AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        api_key="api_xxxxxxxx...",
    ) as client:
        media = await client.process("samples/video.mp4", model_type="DF-1")
        print(f"Status : {media['status']}")
        print(f"Is Fake: {media.get('fake')}")

asyncio.run(detect_deepfake())
```

**Batch processing multiple videos (async)**

```python
async def process_batch(video_paths: list):
    async with AsyncAuthentaClient(...) as client:
        tasks = [client.process(p, model_type="DF-1") for p in video_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for path, result in zip(video_paths, results):
            if isinstance(result, Exception):
                print(f"[FAILED] {path}: {result}")
            else:
                print(f"[OK] {path}: fake={result.get('fake')}")

asyncio.run(process_batch(["video1.mp4", "video2.mp4", "video3.mp4"]))
```

---

### 4.3 FI-1 — Face Intelligence

Face Intelligence provides four detection capabilities. You can enable any combination of them in a single call.

| Parameter | Type | Modality | Description |
| :-- | :-- | :-- | :-- |
| `livenessCheck` | `bool` | Image / Video | Detect whether the face is real or a presentation attack |
| `faceswapCheck` | `bool` | **Video only** | Detect face-swap manipulation |
| `faceSimilarityCheck` | `bool` | **Image only** | Compare face against a reference image |
| `isSingleFace` | `bool` | Image / Video | Validate that only one face is present |
| `reference_img_path` | `str` | Image | Required when `faceSimilarityCheck=True` |
| `auto_polling` | `bool` | — | `True` (default): block until result ready. `False`: return upload metadata immediately |

#### Liveness Detection

**Synchronous**

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

media = client.face_intelligence(
    path="samples/face_video.mp4",
    model_type="FI-1",
    livenessCheck=True,
)

print(f"Job ID   : {media['id']}")
print(f"Status   : {media['status']}")
print(f"Liveness : {media['result']['isLiveness']}")
```

When `auto_polling=True` (default), `face_intelligence()` automatically fetches the detection output from the API and attaches it to the returned dict under `media['result']`. The result contains:

| Field | Description |
| :-- | :-- |
| `isLiveness` | `True` if live face, `False` if presentation attack |
| `isDeepFake` | `True` if face swap detected |
| `isSimilar` | `True` if faces match |
| `similarityScore` | Similarity percentage (0–100) |

**Asynchronous**

```python
import asyncio
from authenta import AuthentaClient
from authenta.async_authenta_client import AsyncAuthentaClient

sync_client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

async def liveness():
    async with AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        api_key="api_xxxxxxxx...",
    ) as async_client:
        media = await async_client.process_FI(
            path="samples/face_video.mp4",
            model_type="FI-1",
            livenessCheck=True,
        )
        result = sync_client.get_result(media)
        print(f"Status   : {media['status']}")
        print(f"Liveness : {result['isLiveness']}")

asyncio.run(liveness())
```

---

#### Face Swap Detection (Video Only)

**Synchronous**

```python
media = client.face_intelligence(
    path="samples/face_video.mp4",
    model_type="FI-1",
    faceswapCheck=True,
)

print(f"Status    : {media['status']}")
print(f"Face Swap : {media['result']['isDeepFake']}")
```

**Asynchronous**

```python
async def faceswap():
    async with AsyncAuthentaClient(...) as async_client:
        media = await async_client.process_FI(
            path="samples/face_video.mp4",
            model_type="FI-1",
            faceSwapCheck=True,
        )
        result = sync_client.get_result(media)
        print(f"Status    : {media['status']}")
        print(f"Face Swap : {result['isDeepFake']}")

asyncio.run(faceswap())
```

---

#### Face Similarity Check (Image Only)

Compare two faces and determine whether they belong to the same person.

**Synchronous**

```python
media = client.face_intelligence(
    path="samples/person_A.jpg",
    reference_img_path="samples/person_B.jpg",
    model_type="FI-1",
    faceSimilarityCheck=True,
)

print(f"Status           : {media['status']}")
print(f"Same Person      : {media['result']['isSimilar']}")
print(f"Similarity Score : {media['result']['similarityScore']}")
```

**Asynchronous**

```python
async def similarity():
    async with AsyncAuthentaClient(...) as async_client:
        media = await async_client.process_FI(
            path="samples/person_A.jpg",
            model_type="FI-1",
            faceSimilarityCheck=True,
        )
        result = sync_client.get_result(media)
        print(f"Similar : {result['isSimilar']}")
        print(f"Score   : {result['similarityScore']}")

asyncio.run(similarity())
```

---

#### Manual Polling with `auto_polling=False`

By default, `face_intelligence()` and `process_FI()` block until processing is complete (`auto_polling=True`). Set `auto_polling=False` to return immediately after upload and poll manually — useful for web servers, background workers, or batched jobs.

**Synchronous**

```python
# Step 1 — fire upload, return immediately
upload_meta = client.face_intelligence(
    path="samples/face_video.mp4",
    model_type="FI-1",
    livenessCheck=True,
    auto_polling=False,        # do not block
)
job_id = upload_meta["job"]["id"]
print(f"Upload started. Job ID: {job_id}")

# ... do other work ...

# Step 2 — poll when ready
media = client.wait_for_job(job_id, interval=5.0, timeout=600.0)

# Step 3 — fetch result
result = client.get_result(media)
print(f"Status   : {media['status']}")
print(f"Liveness : {result['isLiveness']}")
```

**Asynchronous**

```python
async def manual_poll():
    async with AsyncAuthentaClient(...) as async_client:
        # Step 1 — upload without blocking
        upload_meta = await async_client.process_FI(
            path="samples/face_video.mp4",
            model_type="FI-1",
            livenessCheck=True,
            auto_polling=False,
        )
        job_id = upload_meta["job"]["id"]

        # Step 2 — poll when ready
        media = await async_client.wait_for_job(job_id)

        # Step 3 — fetch result
        result = sync_client.get_result(media)
        print(f"Status   : {media['status']}")
        print(f"Liveness : {result['isLiveness']}")

asyncio.run(manual_poll())
```

---

### 4.4 FE-1 — Face Embedding

Extract a **512-dimensional face embedding** from an input image for face recognition, similarity matching, and identity verification.

#### Synchronous

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

media = client.extract_face_vector(
    img_path="samples/face.jpg",
    auto_polling=True
)

embedding = media["result"]["embedding"]

print(f"Job ID     : {media.get('id')}")
print(f"Status     : {media['status']}")
print(f"Vector Dim : {len(embedding)}")  # 512
```


#### Asynchronous

```python
import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient

async def extract_embedding():
    async with AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        api_key="api_xxxxxxxx...",
    ) as client:

        media = await client.extract_face_vector(
            img_path="samples/face.jpg",
            auto_polling=True,
        )

        embedding = media["result"]["embedding"]

        print(f"Job ID     : {media.get('id')}")
        print(f"Status     : {media['status']}")
        print(f"Vector Dim : {len(embedding)}")  # 512

asyncio.run(extract_embedding())
```


### 4.5 Media Management

#### Get Job

Retrieve the current state of a job by its ID.

**Synchronous**

```python
job = client.get_job("YOUR_JOB_ID")
print(f"Status : {job['status']}")
print(f"Type   : {job.get('type')}")
```

**Asynchronous**

```python
async def get():
    async with AsyncAuthentaClient(...) as client:
        job = await client.get_job("YOUR_JOB_ID")
        print(f"Status : {job['status']}")

asyncio.run(get())
```

---

#### List Media

Retrieve a paginated list of all media records associated with your account.

**Synchronous**

```python
# All media (default page)
all_media = client.list_jobs()
print(f"Total records: {len(all_media.get('items', []))}")

# With pagination
page_2 = client.list_jobs(page=2, pageSize=20)
for item in page_2.get("items", []):
    print(f"  {item['id']} — {item['status']}")
```

**Asynchronous**

```python
async def list_all():
    async with AsyncAuthentaClient(...) as client:
        all_media = await client.list_jobs(page=1, pageSize=50)
        for item in all_media.get("items", []):
            print(f"  {item['id']} — {item['status']}")

asyncio.run(list_all())
```

---

#### Delete Job

Permanently remove a job record and its associated data.

**Synchronous**

```python
client.delete_job("YOUR_JOB_ID")
print("Deleted.")
```

**Asynchronous**

```python
async def delete():
    async with AsyncAuthentaClient(...) as client:
        await client.delete_job("YOUR_JOB_ID")
        print("Deleted.")

asyncio.run(delete())
```

---

#### Wait for Media (Manual Poll)

Poll a known media ID until processing completes. Useful after `upload_file()` or `face_intelligence(auto_polling=False)`.

**Synchronous**

```python
media = client.wait_for_job(
    job_id="YOUR_JOB_ID",
    interval=5.0,    # seconds between polls
    timeout=600.0,   # max wait time in seconds
)
print(f"Final status: {media['status']}")
```

**Asynchronous**

```python
async def poll():
    async with AsyncAuthentaClient(...) as client:
        media = await client.wait_for_job(
            job_id="YOUR_JOB_ID",
            interval=5.0,
            timeout=600.0,
        )
        print(f"Final status: {media['status']}")

asyncio.run(poll())
```

---

## 5. Visualization

The SDK includes a `visualization` module to generate visual overlays for detection results.

### Heatmaps — AC-1 (Images)

```python
from authenta.visualization import save_heatmap

media = client.process("samples/photo.jpg", model_type="AC-1")

save_heatmap(
    media=media,
    out_path="results/heatmap.jpg",
    model_type="AC-1",
)
```

Downloads the `heatmapURL` and saves an RGB overlay image showing manipulated regions.

---

### Heatmaps — DF-1 (Videos)

For DF-1, the API may return multiple participants (faces). One heatmap video is saved per participant.

```python
from authenta.visualization import save_heatmap

media = client.process("samples/video.mp4", model_type="DF-1")

# Pass a folder path; saves heatmap_p0.mp4, heatmap_p1.mp4, ...
save_heatmap(
    media=media,
    out_path="./results",
    model_type="DF-1",
)
```

---

### Bounding Box Video — DF-1

Draw detection boxes around faces in a deepfake video and save an annotated copy.

```python
from authenta.visualization import save_bounding_box_video

media = client.process("samples/video.mp4", model_type="DF-1")

save_bounding_box_video(
    media,
    src_video_path="samples/video.mp4",
    out_video_path="results/annotated_video.mp4",
)
```

Fetches bounding box data from `resultURL` and renders labels and confidence scores onto each frame using OpenCV.

---

## 6. Error Handling

All SDK methods raise typed exceptions defined in `authenta_exceptions.py`.

| Exception | API Code | Cause |
| :-- | :-- | :-- |
| `AuthenticationError` | `IAM001` | Invalid or missing credentials |
| `AuthorizationError` | `IAM002` | Insufficient permissions |
| `QuotaExceededError` | `AA001` | API limit reached for your plan |
| `InsufficientCreditsError` | `U007` | Not enough credits |
| `ValidationError` | — | Bad request / unexpected response |
| `ServerError` | — | Server-side 5xx error |
| `AuthentaError` | — | Base class for all SDK errors |

```python
from authenta import AuthentaClient
from authenta import (
    AuthentaError,
    AuthenticationError,
    AuthorizationError,
    QuotaExceededError,
    InsufficientCreditsError,
    ValidationError,
    ServerError,
)

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

try:
    media = client.process("samples/photo.jpg", model_type="AC-1")
except AuthenticationError:
    print("Check your api_key.")
except QuotaExceededError:
    print("API quota exceeded. Upgrade your plan.")
except InsufficientCreditsError:
    print("Not enough credits.")
except TimeoutError as e:
    print(f"Processing timed out: {e}")
except AuthentaError as e:
    print(f"Authenta error [{e.code}]: {e.message}")
```

The same exception classes are raised by `AsyncAuthentaClient`.

---

## 7. API Reference

### `AuthentaClient` (Synchronous)

#### `__init__(base_url, api_key)`

```python
AuthentaClient(base_url: str, api_key: str)
```

Initializes the synchronous client.

---

#### `process(path, model_type, interval=5.0, timeout=600.0) -> Dict`

High-level wrapper: upload + poll until complete.

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `path` | `str` | required | Local path to the media file |
| `model_type` | `str` | required | `"AC-1"` or `"DF-1"` |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time in seconds |

Returns the final processed media dict. Raises `TimeoutError` if `timeout` elapses.

---

#### `face_intelligence(path, model_type, *, reference_img_path=None, isSingleFace=True, faceswapCheck=False, livenessCheck=False, faceSimilarityCheck=False, auto_polling=True, interval=5.0, timeout=600.0) -> Dict`

High-level wrapper for the FI-1 Face Intelligence model.

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `path` | `str` | required | Local path to image or video |
| `model_type` | `str` | required | Use `"FI-1"` |
| `reference_img_path` | `str` | `None` | Required when `faceSimilarityCheck=True` |
| `isSingleFace` | `bool` | `True` | Validate only one face is present |
| `faceswapCheck` | `bool` | `False` | Face swap detection (video only) |
| `livenessCheck` | `bool` | `False` | Liveness verification |
| `faceSimilarityCheck` | `bool` | `False` | Face comparison (image only) |
| `auto_polling` | `bool` | `True` | `True`: block until done. `False`: return upload metadata immediately |
| `interval` | `float` | `5.0` | Seconds between polls (when `auto_polling=True`) |
| `timeout` | `float` | `600.0` | Max wait time (when `auto_polling=True`) |

Returns final media dict when `auto_polling=True`; initial upload metadata when `auto_polling=False`.
Raises `ValueError` for invalid combinations (e.g. `faceswapCheck=True` on an image).

---


#### `extract_face_vector(img_path, auto_polling=True, interval=5.0, timeout=600.0) -> Dict`

High-level wrapper for the FE-1 Face Embedding model.

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `img_path` | `str` | required | Local path to image |
| `auto_polling` | `bool` | `True` | `True`: block until done. `False`: return upload metadata immediately |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time |

Returns media dict with `result['embedding']` when `auto_polling=True`.



#### `upload_file(path, model_type, **kwargs) -> Dict`

Two-step file upload: POST `/api/v1/jobs` → PUT to S3 presigned URL.

| Parameter | Type | Description |
| :-- | :-- | :-- |
| `path` | `str` | Local path to the file |
| `model_type` | `str` | `"AC-1"`, `"DF-1"`, or `"FI-1"` |

Returns the initial job metadata dict (includes `job.id`, `status`, `inputs`).

---

#### `wait_for_job(job_id, interval=5.0, timeout=600.0) -> Dict`

Poll `GET /api/v1/jobs/{job_id}` until terminal status (`COMPLETED`, `PROCESSED`, `FAILED`, `ERROR`, `CANCELLED`, `CANCELED`).

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `job_id` | `str` | required | Job ID |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time in seconds |

Raises `TimeoutError` if `timeout` elapses.

---

#### `get_result(media) -> Dict`

Fetch the detection output JSON from a processed media dict's `resultURL`.

| Parameter | Type | Description |
| :-- | :-- | :-- |
| `media` | `dict` | A job dict returned by `face_intelligence()`, `wait_for_job()`, or `get_job()` — must have `status=PROCESSED` and contain a `resultURL` key |

Returns the detection result dict.

- For FI-1: contains `isLiveness`, `isDeepFake`, `isSimilar`, `similarityScore`
- For FE-1: contains `embedding`

Raises `ValueError` if `resultURL` is missing. 

> When `auto_polling=True` (default), `face_intelligence()` calls `get_result()` automatically and attaches the result under `media['result']`. Call `get_result()` explicitly only when using `auto_polling=False` or when working with the async client.

---

#### `get_job(job_id) -> Dict`

`GET /api/v1/jobs/{job_id}` — fetch the current state of a media record.

---

#### `list_jobs(**params) -> Dict`

`GET /api/v1/jobs` — list all media records.

| Param | Description |
| :-- | :-- |
| `page` | Page number (1-based) |
| `pageSize` | Number of records per page |

---

#### `delete_job(job_id) -> None`

`DELETE /api/v1/jobs/{job_id}` — permanently delete a media record.

---

### `AsyncAuthentaClient` (Asynchronous)

Mirrors `AuthentaClient` with `async/await`. Use as a context manager (`async with`) or call `await client.aclose()` when done.

#### `__init__(base_url, api_key, *, timeout=30.0, client=None)`

```python
AsyncAuthentaClient(
    base_url: str,
    api_key: str,
    timeout: float = 30.0,          # httpx client timeout
    client: httpx.AsyncClient = None  # optional pre-built client
)
```

---

#### `await process(path, model_type, interval=5.0, timeout=600.0) -> Dict`

Async equivalent of `AuthentaClient.process()`.

---

#### `await process_FI(path, model_type, *, isSingleFace=None, faceSwapCheck=None, livenessCheck=None, faceSimilarityCheck=None, auto_polling=True, interval=5.0, timeout=600.0) -> Dict`

Async equivalent of `AuthentaClient.face_intelligence()`.

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `path` | `str` | required | Local path to image or video |
| `model_type` | `str` | required | Use `"FI-1"` |
| `isSingleFace` | `bool` | `None` | Validate single face |
| `faceSwapCheck` | `bool` | `None` | Face swap detection (video only) |
| `livenessCheck` | `bool` | `None` | Liveness verification |
| `faceSimilarityCheck` | `bool` | `None` | Face comparison (image only) |
| `auto_polling` | `bool` | `True` | `True`: await until done. `False`: return upload metadata immediately |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time in seconds |

---


#### `await extract_face_vector(img_path, auto_polling=True, interval=5.0, timeout=600.0) -> Dict`

Async equivalent of face embedding extraction (FE-1).

Returns media dict with `result['embedding']` when `auto_polling=True`.



#### `await upload_file(path, model_type, **kwargs) -> Dict`

Async two-step upload. Returns initial media metadata.

---

#### `await wait_for_job(job_id, interval=5.0, timeout=600.0) -> Dict`

Async poll until terminal status. Raises `TimeoutError` on timeout.

---

#### `await get_job(job_id) -> Dict`

Async fetch of a single media record.

---

#### `await list_jobs(**params) -> Dict`

Async list of media records.

---

#### `await delete_job(job_id) -> None`

Async delete of a media record.