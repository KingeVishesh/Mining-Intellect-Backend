"""NI 43-101 technical report drilling extractor.

The press-release-driven `drilling_extractor` returns whatever Exa Answer
finds first — usually a single high-grade intercept and a single drilling
program's headline numbers. That is not cumulative project history and
shouldn't drive a tonnage prediction.

This module is the better path:

    1.  Use Exa `/search` (NOT `/answer`) with `category="company"` and
        a domain filter biased toward regulatory filings (sedarplus,
        sec.gov, sedar.com) and the major mining-document aggregators
        (minedocs, technicalreport, q4cdn IR sites). Query phrased to
        target NI 43-101 / JORC technical reports specifically.

    2.  Use Exa `/contents` with `text=True` on the top hits to get
        parsed full text — for PDF URLs Exa runs OCR/text extraction
        server-side, returning clean text we can feed to an LLM.

    3.  Hand the text to Grok-3 with a structured output_schema asking
        for cumulative drilling totals as of the report cutoff date,
        deposit-average grade (not best-intercept grade), and the
        drilled footprint.

The output is the same `drilling_evidence` JSONB shape as the press-
release extractor so callers (model_runner, build_model_1) consume it
through the existing interface.

Returns None on any failure — the caller falls back to the press-release
extractor or runs the model without a drilling signal.
"""
from __future__ import annotations
import json
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

EXA_SEARCH_URL   = "https://api.exa.ai/search"
EXA_CONTENTS_URL = "https://api.exa.ai/contents"
EXA_ANSWER_URL   = "https://api.exa.ai/answer"
GROK_URL         = "https://api.x.ai/v1/chat/completions"

# Domains where NI 43-101 / JORC technical reports actually live. Biasing
# Exa search to these gives the LLM real cumulative-drilling text rather
# than press-release headlines.
_TECHNICAL_REPORT_DOMAINS = [
    "sedarplus.ca", "sedar.com", "sec.gov",
    "minedocs.com", "technicalreports.miningdataonline.com",
    "q4cdn.com",  # major operator IR PDF host
]


def _find_company_website(
    project_name: str, material: str,
    country: Optional[str] = None, region: Optional[str] = None,
) -> Optional[str]:
    """Ask Exa Answer for the operating company's website domain. Used as
    a fallback search target when regulatory aggregators don't have the
    project's technical report — juniors like Cartier Resources host
    their NI 43-101 PDFs on their own IR site (cartierresources.com),
    not on sedarplus or SEC.

    Returns a bare domain string (e.g. "cartierresources.com") or None.
    """
    api_key = settings.exa_api_key
    if not api_key:
        return None
    loc = ", ".join(p for p in (region, country) if p) or "unknown location"
    query = (
        f"What is the official website domain of the company that owns or "
        f"operates the {project_name} {material} mining project in {loc}? "
        f"Return only the bare domain (e.g. 'cartierresources.com'), no "
        f"protocol, no path."
    )
    payload = {
        "query": query,
        "system_prompt": (
            "You are a mining-industry researcher. Identify the operating "
            "company's primary website (not LinkedIn, not Wikipedia, not "
            "news aggregators). Return the bare hostname only."
        ),
        "output_schema": {
            "type": "object",
            "properties": {
                "company_website": {
                    "type": ["string", "null"],
                    "description": "Bare domain like 'cartierresources.com'",
                },
                "company_name": {"type": ["string", "null"]},
                "confidence": {
                    "type": "string", "enum": ["high", "medium", "low"],
                },
            },
            "required": ["company_website", "confidence"],
        },
        "text": False,
    }
    try:
        resp = requests.post(
            EXA_ANSWER_URL,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload, timeout=30,
        )
    except requests.exceptions.RequestException as e:
        logger.warning(f"[NI43-101] company-website lookup error: {e}")
        return None
    if resp.status_code != 200:
        return None
    try:
        raw = resp.json().get("answer")
        ans = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (ValueError, json.JSONDecodeError):
        return None
    domain = ans.get("company_website")
    if not domain or not isinstance(domain, str):
        return None
    # Strip protocol and path if present
    domain = domain.lower().strip()
    for prefix in ("https://", "http://"):
        if domain.startswith(prefix):
            domain = domain[len(prefix):]
    domain = domain.split("/")[0].strip()
    # Require at least one dot — guards against stray strings like "company"
    if "." not in domain or len(domain) < 5:
        return None
    logger.info(
        f"[NI43-101] Company website for '{project_name}': {domain} "
        f"(confidence={ans.get('confidence')})"
    )
    return domain


def _exa_search(
    query: str, num_results: int = 6, include_domains: Optional[List[str]] = None,
) -> List[Dict]:
    """Return a list of search hits (each with url + title + score)."""
    api_key = settings.exa_api_key
    if not api_key:
        logger.warning("[NI43-101] EXA_API_KEY not set; skipping search")
        return []
    payload: Dict = {
        "query": query, "num_results": num_results,
        "type": "neural", "use_autoprompt": True,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    try:
        resp = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload, timeout=30,
        )
    except requests.exceptions.RequestException as e:
        logger.warning(f"[NI43-101] search error: {e}")
        return []
    if resp.status_code != 200:
        logger.warning(f"[NI43-101] search HTTP {resp.status_code}: {resp.text[:200]}")
        return []
    return (resp.json() or {}).get("results", []) or []


def _exa_contents(urls: List[str], max_chars_per_doc: int = 60_000) -> Dict[str, str]:
    """Fetch parsed text for each URL via Exa contents. Returns {url: text}."""
    api_key = settings.exa_api_key
    if not api_key or not urls:
        return {}
    payload = {"urls": urls, "text": {"max_characters": max_chars_per_doc}}
    try:
        resp = requests.post(
            EXA_CONTENTS_URL,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload, timeout=120,
        )
    except requests.exceptions.RequestException as e:
        logger.warning(f"[NI43-101] contents error: {e}")
        return {}
    if resp.status_code != 200:
        logger.warning(f"[NI43-101] contents HTTP {resp.status_code}: {resp.text[:200]}")
        return {}
    out: Dict[str, str] = {}
    for r in (resp.json() or {}).get("results", []) or []:
        text = r.get("text") or ""
        if text:
            out[r.get("url", "")] = text
    return out


def _grok_extract(
    project_name: str, material: str, text_blob: str, source_urls: List[str],
) -> Optional[Dict]:
    """Send the assembled technical-report text to Grok-3 with a strict
    structured-output schema. Returns the parsed drilling-evidence dict
    or None on failure.
    """
    api_key = settings.grok_api_key or settings.xai_api_key
    if not api_key:
        logger.warning("[NI43-101] GROK_API_KEY / XAI_API_KEY not set")
        return None

    # Truncate to keep total prompt under ~120k chars — well within Grok's
    # context window but bounded so a single huge PDF doesn't blow up the
    # call.
    if len(text_blob) > 110_000:
        text_blob = text_blob[:110_000] + "\n…[truncated]…"

    sys_prompt = (
        "You are a mining-industry analyst reading NI 43-101 / JORC "
        "technical reports and extracting cumulative project-history "
        "drilling statistics as of the report cutoff date. Return numeric "
        "values exactly as cited in the report — do not estimate. If a "
        "metric is not stated, return null for that field. "
        "IMPORTANT: total_meters_drilled and total_holes must be "
        "CUMULATIVE for the project (all programs to date), not the "
        "delta from one campaign. weighted_grade_g_t must be the "
        "REPORTED RESOURCE GRADE (or deposit-average grade) — never a "
        "single high-grade intercept."
    )
    user_prompt = (
        f"PROJECT: {project_name}\n"
        f"MATERIAL: {material}\n"
        f"SOURCES: {', '.join(source_urls)}\n\n"
        f"TECHNICAL-REPORT TEXT:\n{text_blob}\n\n"
        "Return a JSON object with fields: "
        "total_holes (int or null), total_meters_drilled (number or null), "
        "drilled_area_km2 (number or null), best_intercepts (array of up "
        "to 5 objects with hole_id, from_m, to_m, interval_m, grade_g_t, "
        "source_url), weighted_grade_g_t (number or null — must be the "
        "deposit-average grade, NOT a single high-grade intercept), "
        "qa_qc_present (bool), confidence ('high'/'medium'/'low'), "
        "report_cutoff_date (ISO or null), notes (string)."
    )

    try:
        resp = requests.post(
            GROK_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": "grok-3",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            },
            timeout=180,
        )
    except requests.exceptions.RequestException as e:
        logger.warning(f"[NI43-101] Grok call error: {e}")
        return None
    if resp.status_code != 200:
        logger.warning(f"[NI43-101] Grok HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    try:
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[NI43-101] Grok response parse error: {e}")
        return None

    # Normalise the output to our drilling_evidence shape.
    out = {
        "total_holes":           parsed.get("total_holes"),
        "total_meters_drilled":  parsed.get("total_meters_drilled"),
        "drilled_area_km2":      parsed.get("drilled_area_km2"),
        "best_intercepts":       parsed.get("best_intercepts") or [],
        "weighted_grade_g_t":    parsed.get("weighted_grade_g_t"),
        "qa_qc_present":         parsed.get("qa_qc_present"),
        "confidence":            parsed.get("confidence", "low"),
        "report_cutoff_date":    parsed.get("report_cutoff_date"),
        "notes":                 parsed.get("notes"),
        "source":                "ni_43_101",
        "source_url":            source_urls[0] if source_urls else None,
        "all_source_urls":       source_urls,
        "extracted_at":          datetime.now(timezone.utc).isoformat(),
    }
    return out


def extract_from_ni_43_101(
    project_name: str,
    material: str,
    country: Optional[str] = None,
    region: Optional[str] = None,
    deposit_type: Optional[str] = None,
    pre_mre: bool = True,
    max_docs: int = 3,
) -> Optional[Dict]:
    """End-to-end: search for technical reports → fetch text → Grok extraction.

    `pre_mre=True` adds wording to the search query asking for "prior to"
    or "as of [date]" reports — for a backtest you want cumulative
    drilling at the date of the resource update being matched.
    `max_docs` caps how many top hits we pass to Grok (each is a PDF or
    long HTML page, so 1–3 is realistic).
    """
    loc = ", ".join(p for p in (region, country) if p) or ""
    deposit_clause = f" ({deposit_type})" if deposit_type else ""
    pre_mre_clause = (
        " technical report cumulative drilling completed before the "
        "most recent Mineral Resource Estimate"
        if pre_mre else
        " NI 43-101 technical report cumulative drilling"
    )
    query = (
        f"{project_name}{deposit_clause} {material} {loc}{pre_mre_clause}"
    )

    # Step 1: regulatory aggregator search
    hits = _exa_search(
        query, num_results=max_docs * 3,
        include_domains=_TECHNICAL_REPORT_DOMAINS,
    )
    # Step 2: open search if regulators don't have it
    if not hits:
        hits = _exa_search(query, num_results=max_docs * 3)
    # Step 3: company-website fallback. Juniors like Cartier Resources host
    # their technical reports on their own IR site rather than on
    # sedarplus or SEC. Look up the operator's domain via Exa Answer,
    # then re-search filtered to that domain. We append rather than
    # replace so company-IR hits enrich whatever the open search found.
    company_domain = _find_company_website(project_name, material, country, region)
    if company_domain:
        company_hits = _exa_search(
            query, num_results=max_docs * 2,
            include_domains=[company_domain],
        )
        # Dedupe by URL while preserving search ordering — company-IR hits
        # are usually more relevant for juniors, so put them first.
        seen = set()
        combined: List[Dict] = []
        for h in company_hits + hits:
            u = h.get("url")
            if u and u not in seen:
                seen.add(u)
                combined.append(h)
        hits = combined
    if not hits:
        logger.info(f"[NI43-101] No search hits for '{project_name}'")
        return None

    top_urls = [h["url"] for h in hits[:max_docs] if h.get("url")]
    contents = _exa_contents(top_urls)
    if not contents:
        logger.info(f"[NI43-101] No content retrievable for '{project_name}'")
        return None

    # Concatenate all retrieved texts with clear source separators so
    # Grok can attribute facts back to a URL.
    blob_parts = []
    for url, text in contents.items():
        blob_parts.append(f"=== SOURCE: {url} ===\n{text}")
    blob = "\n\n".join(blob_parts)

    evidence = _grok_extract(project_name, material, blob, list(contents.keys()))
    if evidence:
        logger.info(
            f"[NI43-101] '{project_name}': "
            f"holes={evidence.get('total_holes')}, "
            f"m={evidence.get('total_meters_drilled')}, "
            f"km2={evidence.get('drilled_area_km2')}, "
            f"grade={evidence.get('weighted_grade_g_t')}, "
            f"conf={evidence.get('confidence')}, "
            f"cutoff={evidence.get('report_cutoff_date')}"
        )
    return evidence
