"""
Geocoder — resolve lat/lng from a location name string.
Uses Nominatim (OSM, free, no key required).
"""
from __future__ import annotations
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def geocode(location_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (latitude, longitude) for a location name, or (None, None) on failure.
    Adds a 1-second delay to respect Nominatim's rate limit (1 req/s).
    """
    if not location_name:
        return None, None
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError

        geolocator = Nominatim(user_agent="mining-intellect-v2")
        time.sleep(1)  # Nominatim rate limit
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            logger.info(f"[Geocode] '{location_name}' → ({location.latitude:.4f}, {location.longitude:.4f})")
            return location.latitude, location.longitude
        logger.warning(f"[Geocode] No result for '{location_name}'")
        return None, None
    except Exception as e:
        logger.error(f"[Geocode] Error for '{location_name}': {e}")
        return None, None
