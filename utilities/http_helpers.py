"""HTTP request helpers with error handling."""

import logging
from typing import Any, Dict, Optional

import requests

def safe_get(
    url: str,
    params: Dict[str, Any] = None,
    headers: Dict[str, str] = None,
    timeout: int = 20,
) -> Optional[requests.Response]:
    """Safe GET request with error handling and logging."""
    headers = headers or {}
    headers.setdefault("User-Agent", "auroratrip-ai/1.0")
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp
        logging.warning("GET %s failed (%s): %s", url, resp.status_code, resp.text[:200])
    except Exception as e:
        logging.warning("GET %s raised %s", url, e)
    return None


def safe_post(
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str] = None,
    timeout: int = 20,
) -> Optional[requests.Response]:
    """Safe POST request with error handling and logging."""
    headers = headers or {}
    headers.setdefault("User-Agent", "auroratrip-ai/1.0")
    try:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout)
        if resp.status_code in (200, 201):
            return resp
        logging.warning("POST %s failed (%s): %s", url, resp.status_code, resp.text[:200])
    except Exception as e:
        logging.warning("POST %s raised %s", url, e)
    return None
