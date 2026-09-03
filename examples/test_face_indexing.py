"""
Face indexing — enrol a person's photos, then search for that face.

    python examples/test_face_indexing.py

Needs AUTHENTA_API_KEY in .env. Enrolment returns as soon as the photos reach
S3; the embeddings are generated in the background, so this script polls
tenants() until every face settles before searching.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from authenta.authenta_client import AuthentaClient
from authenta.face_auth import TERMINAL_FACE_STATUSES

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.environ["AUTHENTA_API_KEY"]
BASE_URL = os.environ.get("AUTHENTA_BASE_URL", "https://platform.authenta.ai")

client = AuthentaClient(base_url=BASE_URL, api_key=API_KEY)

DIVIDER = "-" * 55

# Photos of one person to index, and a different photo of them to search with.
ENROLL_IMAGES = [
    "data_samples/face_similiar/person_1/A.jpeg",
    "data_samples/face_similiar/person_1/B.jpeg",
]
SEARCH_IMAGE = "data_samples/face_similiar/person_1/A.jpeg"


# ── 1. Enrol ────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 1 — Enrol a face")
print(DIVIDER)

enrolled = client.faceEnroll(ENROLL_IMAGES)
subject_id = enrolled["subject_id"]

print(f"  subject_id : {subject_id}")
print(f"  photos     : {len(enrolled['faces'])}")
for face in enrolled["faces"]:
    print(f"    {face['face_id'][:8]}…  {face['status']}")


# ── 2. Wait for the embeddings ──────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 2 — Wait for embeddings")
print(DIVIDER)

deadline = time.time() + 120
while True:
    subjects = client.tenants()["subjects"]
    faces = [f for s in subjects if s["subject_id"] == subject_id for f in s["faces"]]

    if faces and all(f["status"] in TERMINAL_FACE_STATUSES for f in faces):
        break
    if time.time() > deadline:
        print("  timed out waiting — searching anyway")
        break

    print(f"  still processing: {[f['status'] for f in faces] or 'subject not visible yet'}")
    time.sleep(3)

for face in faces:
    note = f" — {face['error']}" if face.get("error") else ""
    print(f"  {face['name']:<24} {face['status']}{note}")

searchable = sum(1 for f in faces if f["status"] == "processed")
print(f"  searchable : {searchable} of {len(faces)}")


# ── 3. Search ───────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 3 — Search a face")
print(DIVIDER)

matches = client.faceSearch(SEARCH_IMAGE, limit=10)
print(f"  matches : {matches['count']}")

for match in matches["results"]:
    score = match["similarity_score"] * 100
    mine = "  <- just enrolled" if match["subject_id"] == subject_id else ""
    print(f"    #{match['rank']:<2} {score:5.1f}%  subject {match['subject_id'][:8]}…  {match['name']}{mine}")

if matches["count"] == 0:
    print("  Nothing matched. Either no face is `processed` yet, or the photo")
    print("  has no detectable face — try a clear, front-facing image.")

print(f"\n{DIVIDER}\nDone.\n")
