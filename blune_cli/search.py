"""Discover candidate models on the Hugging Face Hub."""
from typing import Optional

import requests


def search_models(
    query_filter: str = "mixture-of-experts,mlx",
    pipeline_tag: str = "text-generation",
    limit: int = 20,
    sort: str = "downloads",
) -> list[str]:
    """Query the HF Hub API for candidate repo IDs. Default filter targets
    the architecture class we've found gives the best speed/RAM tradeoff on
    unified-memory hardware: MoE models with existing MLX conversions."""
    url = (
        "https://huggingface.co/api/models"
        f"?filter={query_filter}&pipeline_tag={pipeline_tag}"
        f"&sort={sort}&direction=-1&limit={limit}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return [m["id"] for m in resp.json()]


def list_org_models(
    author: str = "mlx-community",
    pipeline_tag: str = "text-generation",
    max_results: Optional[int] = None,
) -> list[str]:
    """Page through every model an org has published (HF caps each page at
    1000 and hands back a `Link: rel="next"` header for the rest). Used to
    bulk-populate the curated config cache from one trusted org in one pass,
    instead of everyone re-discovering the same popular repos one search at
    a time."""
    results: list[str] = []
    url = (
        "https://huggingface.co/api/models"
        f"?author={author}&pipeline_tag={pipeline_tag}&limit=1000"
    )
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        results.extend(m["id"] for m in page)
        if max_results and len(results) >= max_results:
            return results[:max_results]

        url = None
        link = resp.headers.get("link", "")
        if 'rel="next"' in link:
            url = link.split(";")[0].strip("<> ")

    return results
