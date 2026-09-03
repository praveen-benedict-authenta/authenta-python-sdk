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
   - [4.5 DI-1 — Document Forgery Detection](#45-di-1--document-forgery-detection)
   - [4.6 Face Indexing](#46-face-indexing)
   - [4.7 Media Management](#47-media-management)
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
| `DI-1` | Image | Detects forged or tampered content in documents |

---

## 3. Quick Start

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

# Detect AI-generated image (blocks until result is ready)
media = client.process(original_path="photo.jpg", model_type="AC-1")
result = client.get_result(media)
print(f"Status : {media['status']}")
print(f"Result : {result}")
```

> **See also:** [examples/](examples/) directory contains complete Jupyter notebooks for every service:
> - [AuthentaSDK_Examples.ipynb](examples/AuthentaSDK_Examples.ipynb) — Full SDK walkthrough
> - [AC1_Image_Detection.ipynb](examples/AC1_Image_Detection.ipynb) — Image detection
> - [DF1_Deepfake_Video_Detection.ipynb](examples/DF1_Deepfake_Video_Detection.ipynb) — Video deepfake detection
> - [FI1_Face_Intelligence.ipynb](examples/FI1_Face_Intelligence.ipynb) — Face intelligence services
> - [FE1_face_extraction.ipynb](examples/FE1_face_extraction.ipynb) — Face embedding extraction
> - [DI1_Client_demo.ipynb](examples/DI1_Client_demo.ipynb) — Document intelligence

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
media = client.process(original_path="samples/photo.jpg", model_type="AC-1")
result = client.get_result(media)

print(f"Status : {media['status']}")
print(f"Result : {result}")
```

**Two-step (upload now, poll later)**

```python
# Step 1 — upload
upload_meta = client.upload_file("samples/photo.jpg", model_type="AC-1")
jobid = upload_meta["job"]["id"]
print(f"Uploaded. Job ID: {jobid}")

# ... do other work ...

# Step 2 — wait for result
media = client.wait_for_media(jobid)
print(f"Status : {media['status']}")

# Step 3 — get detection result
result = client.get_result(media)
print(f"Result: {result}")
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
        media = await client.process(original_path="samples/photo.jpg", model_type="AC-1")
        print(f"Status : {media['status']}")
        
        # Fetch result
        result = client.get_result(media)
        print(f"Result: {result}")

asyncio.run(detect_image())
```

**Two-step async (upload now, poll later)**

```python
async def detect_image_async():
    async with AsyncAuthentaClient(...) as client:
        # Step 1 — upload
        upload_meta = await client.upload_file("samples/photo.jpg", model_type="AC-1")
        jobid = upload_meta["job"]["id"]

        # Step 2 — poll when ready
        media = await client.wait_for_media(jobid)
        print(f"Status : {media['status']}")
        
        # Step 3 — fetch result
        result = client.get_result(media)
        print(f"Result: {result}")

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
media = client.process(original_path="samples/video.mp4", model_type="DF-1")
result = client.get_result(media)

print(f"Status      : {media['status']}")
print(f"Result      : {result}")
```

**Two-step**

```python
# Step 1 — upload
upload_meta = client.upload_file("samples/video.mp4", model_type="DF-1")
jobid = upload_meta["job"]["id"]

# Step 2 — poll with custom interval/timeout
media = client.wait_for_media(jobid, interval=10.0, timeout=900.0)
print(f"Status : {media['status']}")

# Step 3 — get result
result = client.get_result(media)
print(f"Result: {result}")
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
        media = await client.process(original_path="samples/video.mp4", model_type="DF-1")
        print(f"Status : {media['status']}")
        
        result = client.get_result(media)
        print(f"Result: {result}")

asyncio.run(detect_deepfake())
```

**Batch processing multiple videos (async)**

```python
import asyncio

async def process_batch(video_paths: list):
    async with AsyncAuthentaClient(...) as client:
        # Submit all tasks in parallel
        tasks = [client.process(original_path=p, model_type="DF-1") for p in video_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for path, result in zip(video_paths, results):
            if isinstance(result, Exception):
                print(f"[FAILED] {path}: {result}")
            else:
                result_data = client.get_result(result)
                print(f"[OK] {path}: status={result.get('status')}")

asyncio.run(process_batch(["video1.mp4", "video2.mp4", "video3.mp4"]))
```

---

### 4.3 FI-1 — Face Intelligence

Face Intelligence (FI-1) provides multiple detection capabilities. You can enable any combination in a single call using the `process()` method with `model_type="FI-1"` and optional parameters.

| Parameter | Type | Modality | Description |
| :-- | :-- | :-- | :-- |
| `livenessCheck` | `bool` | Image / Video | Detect whether the face is real or a presentation attack |
| `faceswapCheck` | `bool` | **Video only** | Detect face-swap manipulation |
| `faceSimilarityCheck` | `bool` | **Image only** | Compare face against a reference image |
| `reference_path` | `str` | Image | Required when `faceSimilarityCheck=True` |
| `auto_polling` | `bool` | — | `True` (default): block until result ready. `False`: return upload metadata immediately |

#### Liveness Detection

**Synchronous**

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

media = client.process(
    original_path="samples/face_video.mp4",
    model_type="FI-1",
    livenessCheck=True,
)

print(f"Job ID   : {media['id']}")
print(f"Status   : {media['status']}")

# Fetch result when auto_polling=False or after processing
result = client.get_result(media)
print(f"Liveness : {result['isLiveness']}")
```

When `auto_polling=True` (default for `process()`), the method automatically waits for completion. The result contains:

| Field | Description |
| :-- | :-- |
| `isLiveness` | `True` if live face, `False` if presentation attack |
| `isDeepFake` | `True` if face swap detected |
| `isSimilar` | `True` if faces match |
| `similarityScore` | Similarity percentage (0–100) |

**Asynchronous**

```python
import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient

async def liveness():
    async with AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        api_key="api_xxxxxxxx...",
    ) as client:
        media = await client.process(
            original_path="samples/face_video.mp4",
            model_type="FI-1",
            livenessCheck=True,
        )
        result = client.get_result(media)
        print(f"Status   : {media['status']}")
        print(f"Liveness : {result['isLiveness']}")

asyncio.run(liveness())
```

---

#### Face Swap Detection (Video Only)

**Synchronous**

```python
media = client.process(
    original_path="samples/face_video.mp4",
    model_type="FI-1",
    faceswapCheck=True,
)

result = client.get_result(media)
print(f"Status    : {media['status']}")
print(f"Face Swap : {result['isDeepFake']}")
```

**Asynchronous**

```python
async def faceswap():
    async with AsyncAuthentaClient(...) as client:
        media = await client.process(
            original_path="samples/face_video.mp4",
            model_type="FI-1",
            faceswapCheck=True,
        )
        result = client.get_result(media)
        print(f"Status    : {media['status']}")
        print(f"Face Swap : {result['isDeepFake']}")

asyncio.run(faceswap())
```

---

#### Face Similarity Check (Image Only)

Compare two faces and determine whether they belong to the same person.

**Synchronous**

```python
media = client.process(
    original_path="samples/person_A.jpg",
    model_type="FI-1",
    faceSimilarityCheck=True,
    reference_path="samples/person_B.jpg",
)

result = client.get_result(media)
print(f"Status           : {media['status']}")
print(f"Same Person      : {result['isSimilar']}")
print(f"Similarity Score : {result['similarityScore']}")
```

**Asynchronous**

```python
async def similarity():
    async with AsyncAuthentaClient(...) as client:
        media = await client.process(
            original_path="samples/person_A.jpg",
            model_type="FI-1",
            faceSimilarityCheck=True,
            reference_path="samples/person_B.jpg",
        )
        result = client.get_result(media)
        print(f"Similar : {result['isSimilar']}")
        print(f"Score   : {result['similarityScore']}")

asyncio.run(similarity())
```

---

#### Manual Polling with `auto_polling=False`

By default, `process()` blocks until processing is complete (`auto_polling=True`). Set `auto_polling=False` to return immediately after upload and poll manually — useful for web servers, background workers, or batched jobs.

**Synchronous**

```python
# Step 1 — fire upload, return immediately
upload_meta = client.process(
    original_path="samples/face_video.mp4",
    model_type="FI-1",
    livenessCheck=True,
    auto_polling=False,        # do not block
)
jobid = upload_meta["job"]["id"]
print(f"Upload started. Job ID: {jobid}")

# ... do other work ...

# Step 2 — poll when ready
media = client.wait_for_media(jobid, interval=5.0, timeout=600.0)

# Step 3 — fetch result
result = client.get_result(media)
print(f"Status   : {media['status']}")
print(f"Liveness : {result['isLiveness']}")
```

**Asynchronous**

```python
async def manual_poll():
    async with AsyncAuthentaClient(...) as client:
        # Step 1 — upload without blocking
        upload_meta = await client.process(
            original_path="samples/face_video.mp4",
            model_type="FI-1",
            livenessCheck=True,
            auto_polling=False,
        )
        jobid = upload_meta["job"]["id"]

        # Step 2 — poll when ready
        media = await client.wait_for_media(jobid)

        # Step 3 — fetch result
        result = client.get_result(media)
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

result = client.get_result(media)
embedding = result.get("embedding", [])

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

        result = client.get_result(media)
        embedding = result.get("embedding", [])

        print(f"Job ID     : {media.get('id')}")
        print(f"Status     : {media['status']}")
        print(f"Vector Dim : {len(embedding)}")  # 512

asyncio.run(extract_embedding())
```


### 4.5 DI-1 — Document Forgery Detection

Detect whether a document image has been forged or contains tampered content (altered text, stamps, signatures, etc.).

#### Synchronous

```python
from authenta import AuthentaClient

client = AuthentaClient(
    base_url="https://platform.authenta.ai",
    api_key="api_xxxxxxxx...",
)

media = client.process(original_path="samples/bank_statement.png", model_type="DI-1")
result = client.get_result(media)

print(f"Job ID     : {media['id']}")
print(f"Status     : {media['status']}")
```

#### Asynchronous

```python
import asyncio
from authenta.async_authenta_client import AsyncAuthentaClient

async def detect_document():
    async with AsyncAuthentaClient(
        base_url="https://platform.authenta.ai",
        api_key="api_xxxxxxxx...",
    ) as client:
        media = await client.process(original_path="samples/bank_statement.png", model_type="DI-1")
        result = client.get_result(media)
        print(f"Status: {media['status']}")

asyncio.run(detect_document())
```

---

### 4.6 Face Indexing

Index a person's photos, then find that face again later. This is separate from
the detection models — it runs no FI-1/DF-1/AC-1 job, and uses three endpoints
on the same host and API key:

| Method | Endpoint |
|---|---|
| `faceEnroll(images)` | `POST /api/v1/facesim/v1/enroll` |
| `tenants()` | `GET /api/v1/facesim/v1/subjects` |
| `faceSearch(image, limit)` | `POST /api/v1/facesim/v1/search` |

#### Enrol a face

```python
enrolled = client.faceEnroll([
    "data_samples/face_similiar/person_1/A.jpeg",
    "data_samples/face_similiar/person_1/B.jpeg",
])

print(enrolled["subject_id"])
for face in enrolled["faces"]:
    print(face["face_id"], face["status"])   # "uploaded" or "failed"
```

Accepts **1–10 images** of the same person, and only **JPEG, PNG, or WebP**.
Every file is validated before anything is sent, so a bad path never leaves a
half-built subject on the server.

`faceEnroll` returns as soon as the photos reach S3. The embeddings are
generated in the background, so each face comes back as `uploaded`, not
`processed` — poll `tenants()` to see them settle. One failed upload marks only
its own face rather than sinking the batch.

#### Check indexing progress

```python
for subject in client.tenants()["subjects"]:
    for face in subject["faces"]:
        print(subject["subject_id"], face["name"], face["status"], face["error"])
```

| Status | Meaning |
|---|---|
| `pending` | Upload URL created; the object has not been acknowledged yet |
| `uploaded` | Stored; waiting for a worker |
| `processing` | A worker is generating the embedding |
| `processed` | Embedding stored — the face is searchable |
| `failed` | Upload validation or embedding failed — see `error` |

Only `processed` faces are searched, so searching too early returns `count: 0`.

#### Search a face

```python
matches = client.faceSearch("data_samples/face_similiar/person_1/A.jpeg", limit=10)

print(matches["count"])
for match in matches["results"]:
    print(match["rank"], match["subject_id"], match["similarity_score"])
```

The photo is posted as `multipart/form-data` rather than Base64 in a JSON body,
which keeps it clear of the server's 100 KiB JSON limit. `limit` is clamped to
1–50 and defaults to 50.

Search is independent of enrolment: it matches everything already indexed on the
account, including from earlier sessions. The same subject can appear more than
once because every enrolled face has its own embedding, and results run from
highest similarity down.

Unsupported formats are converted to JPEG when **Pillow** is installed
(`pip install Pillow`); without it you get a clear `ValidationError` instead of
a server-side `invalid_image`. The conversion also bakes any EXIF rotation into
the pixels — an image left with a stale orientation tag is read sideways by face
detection and comes back as `no_face_detected`.

A full runnable script lives in
[`examples/test_face_indexing.py`](examples/test_face_indexing.py).

---

### 4.7 Media Management

#### Get Media

Retrieve the current state of a media record by its ID.

**Synchronous**

```python
media = client.get_media("YOUR_JOB_ID")
print(f"Status : {media['status']}")
```

**Asynchronous**

```python
async def get():
    async with AsyncAuthentaClient(...) as client:
        media = await client.get_media("YOUR_JOB_ID")
        print(f"Status : {media['status']}")

asyncio.run(get())
```

---

#### List Media

Retrieve a list of all media records associated with your account.

**Synchronous**

```python
# All media (first page)
all_media = client.list_media()
items = all_media.get("data", [])
print(f"Total records: {len(items)}")
for item in items[:10]:
    print(f"  {item['id']} — {item['status']}")
```

**Asynchronous**

```python
async def list_all():
    async with AsyncAuthentaClient(...) as client:
        all_media = await client.list_media(page=1, pageSize=10)
        for item in all_media.get("data", []):
            print(f"  {item['id']} — {item['status']}")

asyncio.run(list_all())
```

---

#### Delete Media

Permanently remove a media record and its associated data.

**Synchronous**

```python
client.delete_media("YOUR_JOB_ID")
print("Deleted.")
```

**Asynchronous**

```python
async def delete():
    async with AsyncAuthentaClient(...) as client:
        await client.delete_media("YOUR_JOB_ID")
        print("Deleted.")

asyncio.run(delete())
```

---

#### Wait for Media (Manual Poll)

Poll a known media ID until processing completes. Useful after `upload_file()` or `process(auto_polling=False)`.

**Synchronous**

```python
media = client.wait_for_media(
    jobid="YOUR_JOB_ID",
    interval=5.0,    # seconds between polls
    timeout=600.0,   # max wait time in seconds
)
print(f"Final status: {media['status']}")
```

**Asynchronous**

```python
async def poll():
    async with AsyncAuthentaClient(...) as client:
        media = await client.wait_for_media(
            jobid="YOUR_JOB_ID",
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

media = client.process(original_path="samples/photo.jpg", model_type="AC-1")

os.makedirs("results", exist_ok=True)
save_heatmap(
    media=media,
    out_path="results/",  # output directory
)
```

Downloads the heatmap artifact and saves an RGB overlay image showing manipulated regions.

---

### Heatmaps — DF-1 (Videos)

For video models, heatmap artifacts are extracted from the media response.

```python
from authenta.visualization import save_heatmap

media = client.process(original_path="samples/video.mp4", model_type="DF-1")

os.makedirs("results", exist_ok=True)
save_heatmap(
    media=media,
    out_path="results/",  # output directory
)
```

---

### Bounding Box Video — DF-1

Draw detection boxes around faces in a deepfake video and save an annotated copy.

```python
from authenta.visualization import save_bounding_box_video

media = client.process(original_path="samples/video.mp4", model_type="DF-1")

save_bounding_box_video(
    media=media,
    src_video_path="samples/video.mp4",
    out_video_path="results/annotated_video.mp4",
)
```

Fetches bounding box data from the result artifact and renders boxes, labels, and confidence scores onto each frame using OpenCV.

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

#### `process(original_path, model_type, reference_path=None, faceswapCheck=False, livenessCheck=False, faceSimilarityCheck=False, auto_polling=True, interval=5.0, timeout=600.0) -> Dict`

High-level wrapper: upload + poll until complete.

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `original_path` | `str` | required | Local path to the media file |
| `model_type` | `str` | required | Model type (e.g., `"AC-1"`, `"DF-1"`, `"FI-1"`, `"FE-1"`, `"DI-1"`) |
| `reference_path` | `str` | `None` | Path to reference image for FI-1 face similarity check |
| `faceswapCheck` | `bool` | `False` | Enable face swap detection (FI-1 video only) |
| `livenessCheck` | `bool` | `False` | Enable liveness detection (FI-1) |
| `faceSimilarityCheck` | `bool` | `False` | Enable face similarity check (FI-1 image only) |
| `auto_polling` | `bool` | `True` | `True`: block until done. `False`: return upload metadata immediately |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time in seconds |

Returns the final processed media dict. Raises `TimeoutError` if `timeout` elapses.

---

#### `extract_face_vector(img_path, auto_polling=True, interval=5.0, timeout=600.0) -> Dict`

High-level wrapper for the FE-1 Face Embedding model.

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `img_path` | `str` | required | Local path to image |
| `auto_polling` | `bool` | `True` | `True`: block until done. `False`: return upload metadata immediately |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time |

Returns result dict with `embedding` key when `auto_polling=True`.

---

#### `upload_file(path, model_type, **kwargs) -> Dict`

Two-step file upload: POST `/api/v1/jobs` → PUT to S3 presigned URL.

| Parameter | Type | Description |
| :-- | :-- | :-- |
| `path` | `str` | Local path to the file |
| `model_type` | `str` | Model type (e.g., `"AC-1"`, `"DF-1"`, `"FI-1"`) |

Returns the initial media metadata dict (includes `job.id`, `status`, `inputs`).

---

#### `wait_for_media(jobid, interval=5.0, timeout=600.0) -> Dict`

Poll `GET /api/v1/jobs/{jobid}` until terminal status (`COMPLETED`, `PROCESSED`, `FAILED`, `ERROR`).

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `jobid` | `str` | required | Job ID |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time in seconds |

Raises `TimeoutError` if `timeout` elapses.

---

#### `get_result(media) -> Dict`

Fetch the detection result JSON from a processed media dict's artifact.

| Parameter | Type | Description |
| :-- | :-- | :-- |
| `media` | `dict` | A media dict returned by `process()`, `wait_for_media()`, or `get_media()` — must have `status=PROCESSED` and contain result artifacts |

Returns the detection result dict.

Raises `ValueError` if no result artifact is found.

---

#### `faceEnroll(images) -> Dict`

Enrol 1–10 photos of one person: creates the subject, then uploads each image
to its presigned S3 URL.

| Parameter | Type | Description |
|---|---|---|
| `images` | `list[str]` | Local paths. JPEG, PNG, or WebP only |

Returns the enroll response — `subject_id`, `status`, and a `faces` array whose
`status` is set to `uploaded` or `failed` per image. Embeddings follow out of
band; poll `tenants()` for `processed`.

Raises `ValidationError` for an empty list, more than 10 images, a missing file,
or an unsupported image type — all checked before any request is sent.

#### `tenants() -> Dict`

Every subject and face on the account: `{"tenant_id", "subjects"}`, where each
subject holds `face_id`, `name`, `status`, `embedding`, `image_url`, and `error`.

#### `faceSearch(image, limit=50, timeout=120.0) -> Dict`

Rank enrolled faces against a query photo, posted as `multipart/form-data`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str` | — | Local path to the query photo |
| `limit` | `int` | `50` | Matches to return, clamped to 1–50 |
| `timeout` | `float` | `120.0` | Seconds — embedding takes longer than a plain request |

Returns `{"tenant_id", "count", "results"}` with results ordered from highest
similarity down. Only faces with `status: "processed"` are searched.

#### `get_media(jobid) -> Dict`

`GET /api/v1/jobs/{jobid}` — fetch the current state of a media record.

---

#### `list_media(**params) -> Dict`

`GET /api/v1/jobs` — list all media records.

---

#### `delete_media(jobid) -> None`

`DELETE /api/v1/jobs/{jobid}` — permanently delete a media record.

---

#### `finalize_media(jobid) -> bool`

`POST /api/v1/jobs/{jobid}/finalize` — finalize a media upload.

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

#### `await process(original_path, model_type, reference_path=None, faceswapCheck=False, livenessCheck=False, faceSimilarityCheck=False, auto_polling=True, interval=5.0, timeout=600.0) -> Dict`

Async equivalent of `AuthentaClient.process()`.

---

#### `await extract_face_vector(img_path, auto_polling=True, interval=5.0, timeout=600.0) -> Dict`

Async equivalent of face embedding extraction (FE-1).

Returns result dict with `embedding` key when `auto_polling=True`.

---

#### `await upload_file(path, model_type, **kwargs) -> Dict`

Async two-step upload. Returns initial media metadata.

---

#### `await wait_for_media(jobid, interval=5.0, timeout=600.0) -> Dict`

Async poll until terminal status. Raises `TimeoutError` on timeout.

---

#### `await get_media(jobid) -> Dict`

Async fetch of a single media record.

---

#### `await list_media(**params) -> Dict`

Async list of media records.

---

#### `await delete_media(jobid) -> None`

Async delete of a media record.

---

#### `await finalize_media(jobid) -> bool`

Async finalize of a media upload.