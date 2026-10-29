# Authenta Python SDK

Welcome to the official documentation for the **Authenta Python SDK** — your gateway to state-of-the-art deepfake detection, AI-image analysis, and face intelligence.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Models & Capabilities](#2-models--capabilities)
3. [Quick Start](#3-quick-start)
   - [3.1 Reading Results](#31-reading-results)
4. [Services](#4-services)
   - [4.1 AC-1 — AI-Generated Image Detection](#41-ac-1--ai-generated-image-detection)
   - [4.2 DF-1 — Deepfake Video Detection](#42-df-1--deepfake-video-detection)
   - [4.3 FI-1 — Face Intelligence](#43-fi-1--face-intelligence)
   - [4.4 FE-1 — Face Embedding](#44-fe-1--face-embedding)
   - [4.5 DI-1 — Document Forgery Detection](#45-di-1--document-forgery-detection)
   - [4.6 Face Indexing](#46-face-indexing)
   - [4.7 Media Management](#47-media-management)
   - [4.8 PDF-1 — PDF Tampering Detection](#48-pdf-1--pdf-tampering-detection)
5. [Visualization](#5-visualization)
6. [Error Handling](#6-error-handling)
7. [API Reference](#7-api-reference)
8. [Changelog](#8-changelog)

---

## 1. Getting Started

### Installation

**Option A: Install from PyPI (Recommended)**

```bash
pip install authentasdk
```

Optional extras:

```bash
pip install "authentasdk[viz]"   # visualization helpers (OpenCV, Pillow)
pip install "authentasdk[pdf]"   # count PDF pages automatically for PDF-1 (pypdf)
pip install "authentasdk[all]"
```

**Option B: Local Development**

```bash
git clone https://github.com/praveen-benedict-authenta/authenta-python-sdk.git
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
print(media["result"]["isFake"])
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

Pass the **model code**, the **slug**, or the numeric **task type ID** as `model_type` — all three are accepted. The IDs below are the cloud IDs; on-prem deployments can use different IDs, so prefer the code or slug, or pass your environment's numeric ID directly.

| Model | Slug | ID | Modality | Capability |
| :-- | :-- | :-- | :-- | :-- |
| `AC-1` | `ai-image-detection` | 1 | Image | Detects AI-generated images (Midjourney, Stable Diffusion, etc.) |
| `AF-1` | `ai-audio-detection` | 2 | Audio | Detects AI-generated or cloned audio |
| `VF-1` | `ai-video-detection` | 3 | Video | Detects AI-generated video |
| `DF-1` | `faceswap-detection` | 4 | Video | Face-swap detection — **deprecated**, use `ED-1` |
| `FD-1` | `image-forgery-detection` | 5 | Image | Finds edited or tampered regions in a real image |
| `DI-1` | `document-intelligence` | 6 | Image | Detects AI generation or tampering in a document image |
| `FL-1` | `face-liveness-detection` | 7 | Image | Checks a selfie is a live person, not a photo, screen or mask |
| `FI-1` | `face-intelligence` | 8 | Image / Video | Face-swap and liveness checks in one job |
| `FE-1` | `face-embedding` | 9 | Image | Extracts a face embedding (not a detector) |
| `AS-1` | `audio-splitter` | 10 | Audio | Separates voice from background audio (not a detector) |
| `ED-1` | `ekyc-deepfake` | 11 | Video | eKYC deepfake detection — latest face-swap detector |
| `PDF-1` | `pdf-tampering-detection` | 13 | PDF | Detects edits to a PDF (needs the page count — see [4.8](#48-pdf-1--pdf-tampering-detection)) |

Full per-model inputs, limits and result fields: [docs.authenta.ai → Task Types](https://docs.authenta.ai/api/task-types).

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
print(f"Tag    : {result.get('tag')}")      # real / likely_real / likely_fake / fake
print(f"Result : {result}")
```

### 3.1 Reading Results

Every scored detection result carries one score, `fakeConfidencePercent`, and a `tag` derived from it. Use `tag` or `fakeConfidencePercent` for decisions; never threshold on `confidencePercent`.

| Field | Type | Meaning |
| :-- | :-- | :-- |
| `fakeConfidencePercent` | number, 0–100 | Estimate that the media is AI-generated or manipulated. Higher = more likely fake |
| `realConfidencePercent` | number, 0–100 | `100 - fakeConfidencePercent` |
| `confidencePercent` | number, 0–100 | Confidence in the returned verdict. Kept for compatibility |
| `tag` | string | `real`, `likely_real`, `likely_fake` or `fake` |
| `tagBoundaries` | object | Boundaries that produced `tag`, e.g. `{"real": 30, "likelyReal": 50, "likelyFake": 75}` |
| `isFake` | bool | `True` when `tag` is `likely_fake` or `fake` |
| `modelVersion` | string | Version of the model that produced the result |

The same fields are on `media["result"]` (from `process()`, `wait_for_media()` or `get_media()`) and in the dict `get_result()` returns.

```python
result = media["result"]
tag = result.get("tag")

if tag in ("real", "likely_real"):
    accept(media)
elif tag == "likely_fake":
    review(media)
elif tag == "fake":
    reject(media)
else:  # job completed before tags existed, or a task type with no single score
    review(media) if result.get("isFake") else accept(media)
```

- **Treat `tag`, `tagBoundaries` and `modelVersion` as optional.** Older jobs don't have them, and task types without a single score (e.g. `FI-1`, `FE-1`, `AS-1`) have no `tag`.
- **Read boundaries from `tagBoundaries`**, don't hard-code them — they differ per task type and are stored with each job.
- **Store `modelVersion`** with your decision; re-check your own cutoffs when it changes.
- **Ignore fields you don't recognise** — new ones can be added at any time.

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
| `tenants()` | `GET /api/v1/facesim/v1/tenant` |
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
# First page (default 20, newest first)
all_media = client.list_media()
items = all_media.get("data", [])
print(f"Total records: {all_media.get('total')}")

# Filters: limit (1-100), offset, status, taskTypeSlug, createdAfter, createdBefore
done = client.list_media(limit=10, status="completed", taskTypeSlug="ai-image-detection")
for item in items[:10]:
    print(f"  {item['id']} — {item['status']}")
```

**Asynchronous**

```python
async def list_all():
    async with AsyncAuthentaClient(...) as client:
        all_media = await client.list_media(limit=10, offset=0)
        for item in all_media.get("data", []):
            print(f"  {item['id']} — {item['status']}")

asyncio.run(list_all())
```

---

#### Cancel Media

Cancel a job that hasn't started processing (`initiated` or `queued`); its credits are refunded. Requires an API key with the **Delete** permission.

```python
job = client.cancel_media("YOUR_JOB_ID")
print(job["status"])  # "cancelled"
```

Async: `await client.cancel_media("YOUR_JOB_ID")`.

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

Poll a known media ID until it reaches `completed`, `failed` or `cancelled`. Useful after `upload_file()` or `process(auto_polling=False)`.

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

### 4.8 PDF-1 — PDF Tampering Detection

Detect edits to a PDF. The API needs the PDF's page count: pass `page_count`, or install `pypdf` (`pip install "authentasdk[pdf]"`) and the SDK counts it for you.

```python
media = client.process("contract.pdf", model_type="PDF-1", page_count=3)
result = client.get_result(media)
print(result["isFake"], result.get("modelVersion"))
```

Without `page_count` and without `pypdf`, the SDK raises `ValueError` before creating the job, so no credits are spent.

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

All SDK methods raise typed exceptions defined in `authenta_exceptions.py`. Every API error carries:

- `e.code` — the API's error name, e.g. `INVALID_API_KEY` (branch on this, not on the message)
- `e.status_code` — HTTP status
- `e.request_id` — the API's request ID; include it when contacting support
- `e.details` — the full error body (for `VALIDATION_ERROR`, `e.details["fields"]` says which fields are invalid)
- `e.retry_after` — seconds to wait, when the API sends `Retry-After`

| Exception | HTTP | API `name` | Cause |
| :-- | :-- | :-- | :-- |
| `AuthenticationError` | 401 | `UNAUTHENTICATED`, `INVALID_API_KEY` | Missing, revoked, expired or unknown API key |
| `AuthorizationError` | 403 | `FORBIDDEN` | The key lacks the permission the endpoint needs |
| `InsufficientCreditsError` | 402 | `INSUFFICIENT_BALANCE`, `PAYG_LIMIT_REACHED` | Not enough credits, or monthly pay-as-you-go limit reached |
| `QuotaExceededError` | 429 | `RATE_LIMITED` | Too many requests — see `retry_after` |
| `ValidationError` | 4xx | `VALIDATION_ERROR`, `TASK_TYPE_NOT_FOUND`, `JOB_NOT_FOUND`, ... | Bad request |
| `ServerError` | 5xx | `INTERNAL_ERROR` | Server-side error; safe to retry with backoff |
| `AuthentaError` | — | — | Base class for all SDK errors |

> **Compatibility:** the four specific 4xx errors above are also instances of `ValidationError`, so existing `except ValidationError` handlers keep catching them as they did before 1.1.0. Put specific `except` clauses before `except ValidationError`.

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
    print(f"Authenta error [{e.code}]: {e.message} (request {e.request_id})")
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

#### `process(original_path, model_type, reference_path=None, faceswapCheck=False, livenessCheck=False, faceSimilarityCheck=False, auto_polling=True, interval=5.0, timeout=600.0, page_count=None) -> Dict`

High-level wrapper: upload + poll until complete.

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `original_path` | `str` | required | Local path to the media file |
| `model_type` | `str` | required | Model code (`"AC-1"`), slug (`"ai-image-detection"`) or numeric task type ID — see [Models](#2-models--capabilities) |
| `reference_path` | `str` | `None` | Path to reference image for FI-1 face similarity check |
| `faceswapCheck` | `bool` | `False` | Enable face swap detection (FI-1 video only) |
| `livenessCheck` | `bool` | `False` | Enable liveness detection (FI-1) |
| `faceSimilarityCheck` | `bool` | `False` | Enable face similarity check (FI-1 image only) |
| `auto_polling` | `bool` | `True` | `True`: block until done. `False`: return upload metadata immediately |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time in seconds |
| `page_count` | `int` | `None` | PDF-1 only: the PDF's page count. Counted automatically if omitted and `pypdf` is installed |

Returns the final media dict (`status`, `result`, `artifacts`). Raises `TimeoutError` if `timeout` elapses.

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

Poll `GET /api/v1/jobs/{jobid}` until a terminal status: `completed`, `failed` or `cancelled`.

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `jobid` | `str` | required | Job ID |
| `interval` | `float` | `5.0` | Seconds between polls |
| `timeout` | `float` | `600.0` | Max wait time in seconds |

Raises `TimeoutError` if `timeout` elapses.

---

#### `get_result(media) -> Dict`

Download the result JSON from a processed media dict's `result` artifact. The job-level fields `tag`, `tagBoundaries` and `modelVersion` are merged in when the artifact lacks them; values already in the artifact are never changed.

| Parameter | Type | Description |
| :-- | :-- | :-- |
| `media` | `dict` | A media dict returned by `process()`, `wait_for_media()`, or `get_media()` — must have `status=completed` and contain a result artifact |

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

`GET /api/v1/jobs` — list media records, newest first. Query params: `limit` (1–100, default 20), `offset`, `status`, `taskTypeSlug`, `createdAfter`, `createdBefore`. Returns `{"data", "total", "limit", "offset"}`.

---

#### `cancel_media(jobid) -> Dict`

`POST /api/v1/jobs/{jobid}/cancel` — cancel a job in `initiated` or `queued` status; credits are refunded. Needs the **Delete** permission.

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

#### `await cancel_media(jobid) -> Dict`

Async cancel of a job that hasn't started processing.

---

#### `await delete_media(jobid) -> None`

Async delete of a media record.

---

#### `await finalize_media(jobid) -> bool`

Async finalize of a media upload.

---

## 8. Changelog

### 1.1.0

Backwards compatible — existing code keeps working without changes.

- **Result tags:** `get_result()` now includes `tag`, `tagBoundaries` and `modelVersion` (also on `media["result"]`). See [Reading Results](#31-reading-results).
- **Errors:** API errors now map to `AuthenticationError`, `AuthorizationError`, `InsufficientCreditsError` and `QuotaExceededError` (previously all surfaced as `ValidationError`, which they still subclass). New `request_id` and `retry_after` attributes; `code` now holds the API error name.
- **PDF-1** now sends `pageCount`, which the API requires (`page_count=` or automatic with `pypdf`).
- `model_type` accepts slugs and numeric task type IDs as well as model codes.
- New `cancel_media()` on both clients.
- `wait_for_media()` returns on `cancelled` instead of waiting until timeout.
- Uploaded file names keep their extension (`photo.jpg`, previously `photo`).
- `httpx` is now installed with the SDK; visualization and PDF page counting are optional extras.
