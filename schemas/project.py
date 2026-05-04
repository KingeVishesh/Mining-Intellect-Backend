"""
Mining Intellect — Project Schema (Pydantic)
Maps 1:1 to the `projects` table in Supabase.
"""
from __future__ import annotations
from typing import Any, List, Optional
from pydantic import BaseModel


class ProjectData(BaseModel):
    id: str
    name: str
    material: str

    # Location
    country: Optional[str] = None
    region: Optional[str] = None
    district: Optional[str] = None
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Company
    company_name: Optional[str] = None
    company_id: Optional[str] = None

    # Resource
    deposit_type: Optional[str] = None
    commodity: Optional[str] = None
    project_stage: Optional[str] = None
    tonnage_mt: Optional[float] = None
    grade_value: Optional[float] = None
    grade_unit: Optional[str] = None
    resource_category: Optional[str] = None
    resource_size_value: Optional[float] = None
    resource_size_unit: Optional[str] = None
    by_product_commodities: Optional[List[str]] = None

    # Geology
    host_rock: Optional[str] = None
    mineralization_style: Optional[str] = None
    depth_meters: Optional[float] = None
    width_meters: Optional[float] = None
    strike_length_meters: Optional[float] = None

    # Mining / Processing
    mining_method: Optional[str] = None
    processing_method: Optional[str] = None
    recovery_rate: Optional[float] = None
    mine_life_years: Optional[float] = None
    production_rate_per_year: Optional[float] = None
    production_rate_unit: Optional[str] = None
    final_product: Optional[str] = None
    energy_source: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None

    # Economics
    npv_usd_millions: Optional[float] = None
    capex_usd_millions: Optional[float] = None
    irr_percent: Optional[float] = None
    opex_per_unit: Optional[float] = None
    payback_years: Optional[float] = None

    # Company / Ownership
    ownership_type: Optional[str] = None

    # Location (extended)
    elevation_meters: Optional[float] = None
    climate_terrain: Optional[str] = None

    # Permitting / ESG
    permitting_status: Optional[List[str]] = None

    # Status / Meta
    status: Optional[str] = None
    enrichment_status: Optional[str] = None
    coverage_percent: Optional[float] = None
    fields_missing: Optional[List[str]] = None
    field_statuses: Optional[Any] = None
    data_sources: Optional[Any] = None
    has_model_1: Optional[bool] = None
    has_model_2: Optional[bool] = None
