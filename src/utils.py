import hashlib
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _filename_from_url(url: str) -> str:
    base, ext = os.path.splitext(url)
    if not ext or len(ext) > 5:
        ext = ".jpg"
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"{digest}{ext.lower()}"


@dataclass
class DownloadConfig:
    output_dir: str
    timeout: float = 15.0
    retries: int = 3
    backoff_factor: float = 1.5
    concurrent: int = 1  # kept 1 to avoid throttling by many hosts


def _requests_session(cfg: DownloadConfig) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=cfg.retries,
        backoff_factor=cfg.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def download_image(url: str, cfg: DownloadConfig) -> Optional[str]:
    """Downloads a single image and returns its local path or None.

    The filename is a hash of the URL to ensure stable deduplication.
    """
    ensure_dir(cfg.output_dir)
    filename = _filename_from_url(url)
    out_path = os.path.join(cfg.output_dir, filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    session = _requests_session(cfg)
    try:
        with session.get(url, stream=True, timeout=cfg.timeout) as resp:
            if resp.status_code != 200:
                return None
            tmp_path = out_path + ".part"
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            os.replace(tmp_path, out_path)
            return out_path
    except Exception:
        # Swallow and return None. Callers can decide how to handle missing images
        return None


def download_images(urls: Iterable[str], output_dir: str, show_progress: bool = True) -> List[Optional[str]]:
    """Download many images with automatic retries.

    Returns list of local paths (or None if failed) in the same order as input.
    """
    cfg = DownloadConfig(output_dir=output_dir)
    results: List[Optional[str]] = []

    iterator = tqdm(list(urls), desc="Downloading images", disable=not show_progress)
    for url in iterator:
        if not isinstance(url, str) or not url:
            results.append(None)
            continue
        out = download_image(url, cfg)
        results.append(out)
        # Be a good citizen with a small sleep to avoid throttling
        time.sleep(0.05)
    return results


def url_to_hashed_filename(url: str) -> str:
    return _filename_from_url(url)
