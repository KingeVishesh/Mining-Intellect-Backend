"""
Mining Intellect — Analog Schema (Pydantic)
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class AnalogProject(BaseModel):
    project_id: Optional[str] = None   # Supabase project id (if in DB)
    name: str
    material: str
    deposit_type: Optional[str] = None
    tonnage_mt: Optional[float] = None
    grade_value: Optional[float] = None
    grade_unit: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    project_stage: Optional[str] = None
    mining_method: Optional[str] = None

    similarity_score: float = 0.0      # 0-100 relevance score
    similarity_reasons: List[str] = []  # why it was selected
    source: str = "db"                  # "db" | "exa"
    source_url: Optional[str] = None    # if from Exa
    approved: bool = False              # set to True after human review
