"""
Exa Answer API wrapper for GPS coordinate lookup.

Uses the /answer endpoint with output_schema to get structured
{latitude, longitude, confidence} directly from Exa's AI answer.
"""
from __future__ import annotations
import json
import logging
import requests
from typing import Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

EXA_ANSWER_URL = "https://api.exa.ai/answer"


def ask_coords(
    project_name: str,
    company: str,
    region: str,
    country: str,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Ask Exa Answer API for GPS coordinates of a mining project.

    Returns (latitude, longitude, source_url) or (None, None, None) on failure.
    """
    api_key = settings.exa_api_key
    if not api_key:
        logger.error("[ExaAnswer] EXA_API_KEY not set")
        return None, None, None

    location_parts = [p for p in [region, country] if p and p.strip()]
    location_str = ", ".join(location_parts) if location_parts else "unknown location"

    query = (
        f"What are the GPS coordinates of the {project_name} mining project "
        f"by {company} located in {location_str}? "
        f"Provide the exact mine site latitude and longitude."
    )

    payload = {
        "query": query,
        "system_prompt": (
            "You are a mining data specialist. Provide precise GPS coordinates for the "
            "exact mine site location, not the nearest city. Return decimal degrees."
        ),
        "output_schema": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Decimal degrees, e.g. -27.25",
                },
                "longitude": {
                    "type": "number",
                    "description": "Decimal degrees, e.g. -70.08",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "high=exact mine site, medium=project area, low=regional estimate",
                },
                "source_url": {
                    "type": "string",
                    "description": "URL of source used",
                },
            },
            "required": ["latitude", "longitude", "confidence"],
        },
        "text": False,
    }

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(EXA_ANSWER_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        logger.warning(f"[ExaAnswer] Timeout for '{project_name}'")
        return None, None, None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[ExaAnswer] Request error for '{project_name}': {e}")
        return None, None, None

    if resp.status_code != 200:
        logger.warning(
            f"[ExaAnswer] HTTP {resp.status_code} for '{project_name}': {resp.text[:200]}"
        )
        return None, None, None

    data = resp.json()
    raw_answer = data.get("answer")
    if raw_answer is None:
        logger.warning(f"[ExaAnswer] No 'answer' field for '{project_name}'")
        return None, None, None

    if isinstance(raw_answer, str):
        try:
            answer = json.loads(raw_answer)
        except json.JSONDecodeError:
            logger.warning(f"[ExaAnswer] Could not parse answer JSON for '{project_name}': {raw_answer[:200]}")
            return None, None, None
    else:
        answer = raw_answer

    lat = answer.get("latitude")
    lng = answer.get("longitude")
    source_url = answer.get("source_url")

    if lat is None or lng is None:
        logger.warning(f"[ExaAnswer] Missing lat/lng for '{project_name}': {answer}")
        return None, None, None

    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        logger.warning(f"[ExaAnswer] Out-of-range coords for '{project_name}': lat={lat}, lng={lng}")
        return None, None, None

    confidence = answer.get("confidence", "unknown")
    logger.info(
        f"[ExaAnswer] '{project_name}': lat={lat}, lng={lng}, "
        f"confidence={confidence}, source={source_url}"
    )
    return float(lat), float(lng), source_url


def ask_company_name(
    project_name: str,
    material: str,
    country: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ask Exa Answer API for the company that owns a mining project.
    Returns (company_name, confidence) or (None, None) on failure.
    """
    api_key = settings.exa_api_key
    if not api_key:
        logger.error("[ExaAnswer] EXA_API_KEY not set")
        return None, None

    location_parts = [p for p in [region, country] if p and p.strip()]
    location_str = ", ".join(location_parts) if location_parts else "unknown location"

    query = (
        f"What company owns or operates the {project_name} {material} mining project "
        f"in {location_str}? Provide the full official company name."
    )

    payload = {
        "query": query,
        "system_prompt": (
            "You are a mining industry expert. Return the company that owns or operates "
            "the mining project. Use the official company name as used in press releases or filings."
        ),
        "output_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Full company name, e.g. 'NexGen Energy Ltd'",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "high=confirmed from official source, medium=likely, low=uncertain",
                },
            },
            "required": ["company_name", "confidence"],
        },
        "text": False,
    }

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(EXA_ANSWER_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        logger.warning(f"[ExaAnswer] Timeout looking up company for '{project_name}'")
        return None, None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[ExaAnswer] Request error for '{project_name}': {e}")
        return None, None

    if resp.status_code != 200:
        logger.warning(f"[ExaAnswer] HTTP {resp.status_code} for '{project_name}': {resp.text[:200]}")
        return None, None

    data = resp.json()
    raw_answer = data.get("answer")
    if raw_answer is None:
        logger.warning(f"[ExaAnswer] No 'answer' field for '{project_name}'")
        return None, None

    if isinstance(raw_answer, str):
        try:
            answer = json.loads(raw_answer)
        except json.JSONDecodeError:
            logger.warning(f"[ExaAnswer] Could not parse answer JSON for '{project_name}': {raw_answer[:200]}")
            return None, None
    else:
        answer = raw_answer

    company_name = answer.get("company_name")
    confidence = answer.get("confidence", "low")

    if not company_name:
        logger.warning(f"[ExaAnswer] No company_name in answer for '{project_name}': {answer}")
        return None, None

    logger.info(f"[ExaAnswer] '{project_name}' company='{company_name}' confidence={confidence}")
    return company_name, confidence
