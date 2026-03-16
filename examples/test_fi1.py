import json
import os
from pathlib import Path
from dotenv import load_dotenv
from authenta.authenta_client import AuthentaClient

load_dotenv(Path(__file__).parent.parent / ".env")

CLIENT_ID     = os.environ["AUTHENTA_CLIENT_ID"]
CLIENT_SECRET = os.environ["AUTHENTA_CLIENT_SECRET"]
BASE_URL      = os.environ.get("AUTHENTA_BASE_URL", "https://platform.authenta.ai")

client = AuthentaClient(
    base_url=BASE_URL,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
)

DIVIDER = "-" * 55

# ── 1. Liveness — Real video ────────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 1 — Liveness: REAL video")
print(DIVIDER)
media = client.face_intelligence(
    path="data_samples/face_live_video/real/1.mp4",
    model_type="FI-1",
    livenessCheck=True,
    auto_polling=True,
)
print(f"  mid    : {media['mid']}")
print(f"  status : {media['status']}")
print(f"  result : {json.dumps(media['result'], indent=4)}")

# ── 2. Liveness — Fake video ────────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 2 — Liveness: FAKE video")
print(DIVIDER)
media = client.face_intelligence(
    path="data_samples/face_live_video/fake/1.mp4",
    model_type="FI-1",
    livenessCheck=True,
    auto_polling=True,
)
print(f"  mid    : {media['mid']}")
print(f"  status : {media['status']}")
print(f"  result : {json.dumps(media['result'], indent=4)}")

# ── 3. Face Swap — Fake video ───────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 3 — Face Swap: FAKE (swapped) video")
print(DIVIDER)
media = client.face_intelligence(
    path="data_samples/faceswap/fake/1.mp4",
    model_type="FI-1",
    faceswapCheck=True,
    auto_polling=True,
)
print(f"  mid    : {media['mid']}")
print(f"  status : {media['status']}")
print(f"  result : {json.dumps(media['result'], indent=4)}")

# ── 4. Face Swap — Real video ───────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 4 — Face Swap: REAL (no swap) video")
print(DIVIDER)
media = client.face_intelligence(
    path="data_samples/faceswap/real/1.mp4",
    model_type="FI-1",
    faceswapCheck=True,
    auto_polling=True,
)
print(f"  mid    : {media['mid']}")
print(f"  status : {media['status']}")
print(f"  result : {json.dumps(media['result'], indent=4)}")

# ── 5. Face Similarity — Same person ───────────────────────
print(f"\n{DIVIDER}")
print("TEST 5 — Face Similarity: SAME person (person_1 A vs B)")
print(DIVIDER)
media = client.face_intelligence(
    path="data_samples/face_similiar/person_1/A.jpeg",
    reference_img_path="data_samples/face_similiar/person_1/B.jpeg",
    model_type="FI-1",
    faceSimilarityCheck=True,
    auto_polling=True,
)
print(f"  mid    : {media['mid']}")
print(f"  status : {media['status']}")
print(f"  result : {json.dumps(media['result'], indent=4)}")

# ── 6. Face Similarity — Different persons ──────────────────
print(f"\n{DIVIDER}")
print("TEST 6 — Face Similarity: DIFFERENT persons (person_1 vs person_2)")
print(DIVIDER)
media = client.face_intelligence(
    path="data_samples/face_similiar/person_1/A.jpeg",
    reference_img_path="data_samples/face_similiar/person_2/A.jpeg",
    model_type="FI-1",
    faceSimilarityCheck=True,
    auto_polling=True,
)
print(f"  mid    : {media['mid']}")
print(f"  status : {media['status']}")
print(f"  result : {json.dumps(media['result'], indent=4)}")

print(f"\n{DIVIDER}")
print("All 6 tests complete.")
print(DIVIDER)
