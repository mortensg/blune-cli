"""Discover candidate models on the Hugging Face Hub."""
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
