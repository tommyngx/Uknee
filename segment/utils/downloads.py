"""Download U-Bench datasets or weights from Hugging Face."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_REPO_ID = "FengheTan9/U-Bench"
DOWNLOAD_SPECS = {
    "dataset": {"repo_type": "dataset", "directory": "data"},
    "weights": {"repo_type": "model", "directory": "weights"},
}


def check_internet(endpoint: str = DEFAULT_ENDPOINT, timeout: float = 5.0) -> bool:
    """Return whether the configured Hugging Face endpoint is reachable."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or endpoint
    port = parsed.port or (443 if parsed.scheme != "http" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _load_huggingface_api():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for downloads. Install it with "
            "`pip install huggingface-hub`."
        ) from exc
    return HfApi, snapshot_download


def download(
    kind: str,
    output_dir: str | Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 5.0,
) -> Path:
    """Download a dataset or model snapshot and return its local directory."""
    if kind not in DOWNLOAD_SPECS:
        raise ValueError(f"Unknown download kind: {kind}")

    spec = DOWNLOAD_SPECS[kind]
    target_dir = Path(output_dir or REPO_ROOT / spec["directory"]).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not check_internet(endpoint, timeout=timeout):
        raise ConnectionError(f"Cannot connect to Hugging Face endpoint: {endpoint}")

    HfApi, snapshot_download = _load_huggingface_api()
    api = HfApi(endpoint=endpoint)
    if spec["repo_type"] == "dataset":
        repo_info = api.dataset_info(repo_id=repo_id)
    else:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")

    file_count = len(getattr(repo_info, "siblings", []))
    print(f"Repository: {repo_info.id} ({file_count} files)")
    print(f"Downloading {kind} to: {target_dir}")

    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=spec["repo_type"],
        local_dir=target_dir,
        endpoint=endpoint,
    )
    return Path(downloaded_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(DOWNLOAD_SPECS), help="Artifact type to download")
    parser.add_argument("--output-dir", help="Destination; defaults to <repo>/data or <repo>/weights")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repository ID")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Hugging Face API endpoint")
    parser.add_argument("--timeout", type=float, default=5.0, help="Connectivity timeout in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        downloaded_dir = download(
            kind=args.kind,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            endpoint=args.endpoint,
            timeout=args.timeout,
        )
    except (ConnectionError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Download failed: {exc}") from exc
    print(f"Download complete: {downloaded_dir}")


if __name__ == "__main__":
    main()
