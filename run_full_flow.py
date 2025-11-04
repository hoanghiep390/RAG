"""Run full front->back flow by simulating an uploaded file.

This script creates a small test document in-memory, wraps it in a
FakeUploadedFile that exposes `.name` and `.getbuffer()`, then calls
DocumentPipeline.process_uploaded_file to execute: chunking -> extraction ->
graph building -> embedding (if available).

Run from repository root (Windows PowerShell):
    python .\run_full_flow.py
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from backend.core.pipeline import DocumentPipeline


class FakeUploadedFile:
    def __init__(self, name: str, content_bytes: bytes):
        self.name = name
        self._buf = content_bytes

    def getbuffer(self):
        return self._buf


def ensure_test_doc():
    path = Path("backend/data/test/test_doc.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# LightRAG Integration Test\n\n"
        "Apple Inc. is a technology company founded by Steve Jobs.\n"
        "Microsoft Corporation was founded by Bill Gates.\n"
        "Both companies are based in the United States.\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def main():
    test_path = ensure_test_doc()

    # Read content and create fake upload
    content_bytes = test_path.read_bytes()
    uploaded = FakeUploadedFile(test_path.name, content_bytes)

    pipeline = DocumentPipeline(user_id="integration_test_user", enable_advanced=True)

    print("\n=== Running full pipeline (simulated upload) ===\n")
    result = pipeline.process_uploaded_file(uploaded, enable_gleaning=False)

    print("\n=== Pipeline result summary ===")
    for k, v in result.items():
        if k == 'success' or isinstance(v, (str, int)):
            print(f"  {k}: {v}")
        else:
            # For larger objects, print simplified info
            try:
                print(f"  {k}: {type(v).__name__}")
            except Exception:
                print(f"  {k}: <complex>")

    print("\nOutputs written under: backend/data/integration_test_user/")


if __name__ == "__main__":
    main()
