"""
Canonical positive/negative analog fixtures.

Each fixture is a fully-populated geological profile for a real mining project.
They are intentionally redundant on free-text fields so the analog finder's
heuristic detectors land on the right slug even if the structured fields
weren't set — that's the production reality (Exa narratives are messy).

The golden test suite (`tests/test_analog_rules.py`) loads each commodity's
rule, constructs the matching target profile, and asserts that:
  * every `must_pick` candidate passes the cascade
  * every `must_drop` candidate is dropped at the expected level

When a rule's exclusions change, the test surface either confirms the new
behaviour (good) or fails (caught regression). The fixtures themselves should
rarely change — they represent ground truth.
"""
from __future__ import annotations

# ── COPPER ───────────────────────────────────────────────────────────────────

# BC alkalic Cu-Au porphyries (the right Hat Copper analogs)
MT_MILLIGAN = {
    "name": "Mt. Milligan", "material": "copper",
    "deposit_type": "alkalic porphyry copper-gold",
    "mineralization_style": "stockwork sulphide chalcopyrite",
    "deposit_subtype": "alkalic_porphyry", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "bc_quesnel_stikine", "recovery_method": "flotation",
    "metal_suite": "cu_au",
    "country": "Canada", "region": "British Columbia", "district": "Quesnel terrane",
    "processing_method": "flotation",
    "tonnage_mt": 720.0, "grade_value": 0.21, "grade_unit": "% Cu",
}
MT_POLLEY = {
    "name": "Mt. Polley", "material": "copper",
    "deposit_type": "alkalic porphyry Cu-Au",
    "mineralization_style": "sulphide stockwork",
    "deposit_subtype": "alkalic_porphyry", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "bc_quesnel_stikine", "recovery_method": "flotation",
    "metal_suite": "cu_au",
    "country": "Canada", "region": "British Columbia", "district": "Quesnel",
    "processing_method": "flotation",
    "tonnage_mt": 400.0, "grade_value": 0.29, "grade_unit": "% Cu",
}
COPPER_MOUNTAIN = {
    "name": "Copper Mountain", "material": "copper",
    "deposit_type": "alkalic porphyry copper-gold",
    "mineralization_style": "sulphide stockwork", "deposit_subtype": "alkalic_porphyry",
    "mineralization_mode": "primary_sulfide", "tectonic_belt": "bc_quesnel_stikine",
    "recovery_method": "flotation", "metal_suite": "cu_au",
    "country": "Canada", "region": "British Columbia", "district": "Princeton",
    "processing_method": "flotation",
    "tonnage_mt": 250.0, "grade_value": 0.33, "grade_unit": "% Cu",
}
RED_CHRIS = {
    "name": "Red Chris", "material": "copper",
    "deposit_type": "alkalic porphyry copper-gold",
    "mineralization_style": "stockwork sulphide", "deposit_subtype": "alkalic_porphyry",
    "mineralization_mode": "primary_sulfide", "tectonic_belt": "bc_quesnel_stikine",
    "recovery_method": "flotation", "metal_suite": "cu_au",
    "country": "Canada", "region": "British Columbia", "district": "Stikine Golden Triangle",
    "processing_method": "flotation",
    "tonnage_mt": 1000.0, "grade_value": 0.40, "grade_unit": "% Cu",
}
CADIA = {
    "name": "Cadia", "material": "copper",
    "deposit_type": "alkalic porphyry Cu-Au",
    "mineralization_style": "stockwork sulphide", "deposit_subtype": "alkalic_porphyry",
    "mineralization_mode": "primary_sulfide", "tectonic_belt": "lachlan",
    "recovery_method": "flotation", "metal_suite": "cu_au",
    "country": "Australia", "region": "New South Wales", "district": "Lachlan",
    "processing_method": "flotation",
    "tonnage_mt": 2200.0, "grade_value": 0.32, "grade_unit": "% Cu",
}

# Wrong analogs the Hat Copper run picked
MARIMACA = {
    "name": "Marimaca", "material": "copper",
    "deposit_type": "IOCG oxide",
    "mineralization_style": "supergene oxide chrysocolla brochantite",
    "deposit_subtype": "iocg_oxide", "mineralization_mode": "supergene_oxide",
    "tectonic_belt": "andean", "recovery_method": "heap_leach", "metal_suite": "cu_au",
    "country": "Chile", "region": "Antofagasta", "district": "Atacama",
    "processing_method": "heap leach",
    "tonnage_mt": 200.0, "grade_value": 0.45, "grade_unit": "% Cu",
}
FLORENCE = {
    "name": "Florence", "material": "copper",
    "deposit_type": "porphyry oxide ISCR",
    "mineralization_style": "supergene oxide blanket chrysocolla",
    "deposit_subtype": "oxide_iscr_supergene_blanket",
    "mineralization_mode": "supergene_oxide",
    "tectonic_belt": "laramide_southwest", "recovery_method": "iscr",
    "metal_suite": "cu_au",
    "country": "USA", "region": "Arizona", "district": "Pinal",
    "processing_method": "in-situ copper recovery SX-EW",
    "tonnage_mt": 363.0, "grade_value": 0.35, "grade_unit": "% Cu",
}
VAN_DYKE = {
    "name": "Van Dyke", "material": "copper",
    "deposit_type": "porphyry oxide ISCR",
    "mineralization_style": "oxide supergene",
    "deposit_subtype": "oxide_iscr_supergene_blanket",
    "mineralization_mode": "supergene_oxide",
    "tectonic_belt": "laramide_southwest", "recovery_method": "iscr",
    "metal_suite": "cu_au",
    "country": "USA", "region": "Arizona", "district": "Pinal",
    "processing_method": "ISCR SX-EW",
    "tonnage_mt": 98.0, "grade_value": 0.33, "grade_unit": "% Cu",
}
MOONLIGHT_SUPERIOR = {
    "name": "Moonlight-Superior", "material": "copper",
    "deposit_type": "IOCG-hybrid",
    "mineralization_style": "sulphide bornite chalcopyrite",
    "deposit_subtype": "iocg_hybrid", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": None, "recovery_method": "flotation", "metal_suite": "cu_au",
    "country": "USA", "region": "California", "district": "Lights Creek",
    "processing_method": "flotation",
    "tonnage_mt": 403.0, "grade_value": 0.31, "grade_unit": "% Cu",
}
KAMOA_KAKULA = {
    "name": "Kamoa-Kakula", "material": "copper",
    "deposit_type": "stratabound high-grade copper",
    "mineralization_style": "stratabound sediment-hosted",
    "deposit_subtype": "kupferschiefer_style", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "central_african_copperbelt", "recovery_method": "flotation",
    "metal_suite": "cu_au",
    "country": "DR Congo", "region": "Lualaba", "district": "Katanga Copperbelt",
    "processing_method": "flotation",
    "tonnage_mt": 1272.0, "grade_value": 2.65, "grade_unit": "% Cu",
}
LA_GRANJA = {
    "name": "La Granja", "material": "copper",
    "deposit_type": "calc-alkaline porphyry copper",
    "mineralization_style": "porphyry stockwork sulphide",
    "deposit_subtype": "calc_alkalic_porphyry", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "andean", "recovery_method": "flotation",
    "metal_suite": "cu_mo",
    "country": "Peru", "region": "Cajamarca", "district": "Northern Andes",
    "processing_method": "flotation",
    "tonnage_mt": 1427.0, "grade_value": 0.56, "grade_unit": "% Cu",
}

# ── GOLD ─────────────────────────────────────────────────────────────────────

DETOUR_LAKE = {
    "name": "Detour Lake", "material": "gold",
    "deposit_type": "orogenic shear-hosted gold",
    "mineralization_style": "shear-hosted sulphide vein",
    "deposit_subtype": "orogenic_general", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "abitibi", "recovery_method": "cil_cip", "metal_suite": "au_only",
    "country": "Canada", "region": "Ontario", "district": "Abitibi greenstone",
    "processing_method": "CIL",
    "tonnage_mt": 350.0, "grade_value": 1.0, "grade_unit": "g/t Au",
}
SUNRISE_DAM = {
    "name": "Sunrise Dam", "material": "gold",
    "deposit_type": "greenstone orogenic gold",
    "mineralization_style": "greenstone-hosted sulphide",
    "deposit_subtype": "greenstone_orogenic", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "yilgarn", "recovery_method": "cil_cip", "metal_suite": "au_only",
    "country": "Australia", "region": "Western Australia", "district": "Yilgarn craton",
    "processing_method": "CIL",
    "tonnage_mt": 70.0, "grade_value": 2.4, "grade_unit": "g/t Au",
}
KIBALI = {
    "name": "Kibali", "material": "gold",
    "deposit_type": "turbidite-hosted orogenic gold",
    "mineralization_style": "turbidite-hosted sulphide vein",
    "deposit_subtype": "turbidite_orogenic", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "west_african_birimian", "recovery_method": "cil_cip",
    "metal_suite": "au_only",
    "country": "DRC", "region": "Haut-Uele", "district": "Birimian",
    "processing_method": "CIL",
    "tonnage_mt": 100.0, "grade_value": 3.5, "grade_unit": "g/t Au",
}
YANACOCHA = {
    "name": "Yanacocha", "material": "gold",
    "deposit_type": "high-sulfidation epithermal gold",
    "mineralization_style": "advanced argillic alteration",
    "deposit_subtype": "high_sulfidation_epithermal",
    "mineralization_mode": "supergene_oxide",
    "tectonic_belt": "andean", "recovery_method": "heap_leach", "metal_suite": "au_only",
    "country": "Peru", "region": "Cajamarca", "district": "Andean",
    "processing_method": "heap leach",
    "tonnage_mt": 1500.0, "grade_value": 0.7, "grade_unit": "g/t Au",
}
GOLDSTRIKE = {
    "name": "Goldstrike", "material": "gold",
    "deposit_type": "carlin disseminated refractory",
    "mineralization_style": "carlin-type decalcified",
    "deposit_subtype": "carlin_general", "mineralization_mode": "refractory_sulfide",
    "tectonic_belt": "great_basin_carlin", "recovery_method": "cn_leach",
    "metal_suite": "au_only",
    "country": "USA", "region": "Nevada", "district": "Carlin Trend",
    "processing_method": "autoclave CIL",
    "tonnage_mt": 400.0, "grade_value": 2.5, "grade_unit": "g/t Au",
}
BINGHAM_AU = {
    "name": "Bingham", "material": "gold",
    "deposit_type": "Laramide porphyry copper-gold",
    "mineralization_style": "porphyry stockwork sulphide",
    "deposit_subtype": "laramide_porphyry", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "laramide_southwest", "recovery_method": "flotation",
    "metal_suite": "cu_au",
    "country": "USA", "region": "Utah", "district": "Bingham Canyon",
    "processing_method": "flotation",
    "tonnage_mt": 2000.0, "grade_value": 0.5, "grade_unit": "g/t Au",
}

# ── NICKEL ───────────────────────────────────────────────────────────────────

SUDBURY = {
    "name": "Sudbury", "material": "nickel",
    "deposit_type": "magmatic sulphide Ni-Cu-PGE",
    "mineralization_style": "conduit-hosted Ni-Cu-PGE sulphide",
    "deposit_subtype": "conduit_hosted", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "abitibi", "recovery_method": "flotation", "metal_suite": "ni_cu_pge",
    "country": "Canada", "region": "Ontario", "district": "Sudbury basin",
    "processing_method": "flotation",
    "tonnage_mt": 500.0, "grade_value": 1.5, "grade_unit": "% Ni",
}
KAMBALDA = {
    "name": "Kambalda", "material": "nickel",
    "deposit_type": "komatiite-hosted nickel sulphide",
    "mineralization_style": "komatiite massive sulphide",
    "deposit_subtype": "komatiite_hosted", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "yilgarn", "recovery_method": "flotation", "metal_suite": "ni_cu_pge",
    "country": "Australia", "region": "Western Australia", "district": "Yilgarn",
    "processing_method": "flotation",
    "tonnage_mt": 60.0, "grade_value": 2.4, "grade_unit": "% Ni",
}
RAMU_LATERITE = {
    "name": "Ramu", "material": "nickel",
    "deposit_type": "nickel laterite limonite",
    "mineralization_style": "limonite laterite blanket",
    "deposit_subtype": "limonite_laterite", "mineralization_mode": "supergene_oxide",
    "tectonic_belt": "indonesia_philippines_arc", "recovery_method": "hpal",
    "metal_suite": "ni_co",
    "country": "Papua New Guinea", "region": "Madang", "district": "Bismarck Arc",
    "processing_method": "HPAL",
    "tonnage_mt": 143.0, "grade_value": 0.91, "grade_unit": "% Ni",
}
GORO_LATERITE = {
    "name": "Goro", "material": "nickel",
    "deposit_type": "nickel laterite saprolite",
    "mineralization_style": "saprolite laterite",
    "deposit_subtype": "saprolite_laterite", "mineralization_mode": "supergene_oxide",
    "tectonic_belt": "new_caledonia_laterite", "recovery_method": "hpal",
    "metal_suite": "ni_co",
    "country": "New Caledonia", "region": "South Province", "district": "New Caledonia laterite",
    "processing_method": "HPAL",
    "tonnage_mt": 75.0, "grade_value": 1.6, "grade_unit": "% Ni",
}


# ── Golden test cases ────────────────────────────────────────────────────────
# Each entry pins a rule, a synthetic target profile that should route to it,
# the analogs that MUST pass, and analogs that MUST drop.

HAT_TARGET = {
    "material": "copper", "deposit_type": "alkalic porphyry copper-gold",
    "deposit_subtype": "alkalic_porphyry", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "bc_quesnel_stikine", "recovery_method": "flotation",
    "metal_suite": "cu_au_co_sc",
    "country": "Canada", "region": "British Columbia", "district": "Stikine",
    "name": "Hat Copper",
}

ABITIBI_GOLD_TARGET = {
    "material": "gold", "deposit_type": "orogenic shear-hosted gold",
    "deposit_subtype": "greenstone_orogenic", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "abitibi", "recovery_method": "cil_cip", "metal_suite": "au_only",
    "country": "Canada", "region": "Ontario", "district": "Abitibi",
    "processing_method": "CIL",
    "name": "Abitibi Test Project",
}

NI_SULPHIDE_TARGET = {
    "material": "nickel", "deposit_type": "magmatic Ni-Cu sulphide",
    "deposit_subtype": "magmatic_sulphide_general", "mineralization_mode": "primary_sulfide",
    "tectonic_belt": "newfoundland_appalachian", "recovery_method": "flotation",
    "metal_suite": "ni_cu_pge",
    "country": "Canada", "region": "Newfoundland", "district": "Voisey's Bay",
    "name": "Test Magmatic Ni Project",
}


GOLDEN_CASES: list[dict] = [
    {
        "name": "alkalic Cu-Au porphyry → BC analogs only",
        "rule_id": "analog_sel_copper_porphyry_alkalic",
        "target": HAT_TARGET,
        "must_pick": [MT_MILLIGAN, MT_POLLEY, COPPER_MOUNTAIN, RED_CHRIS, CADIA],
        "must_drop": [
            (MARIMACA,         {"rule_subtype", "rule_required_subtype"}),
            (FLORENCE,         {"rule_subtype", "rule_required_subtype"}),
            (VAN_DYKE,         {"rule_subtype", "rule_required_subtype"}),
            (MOONLIGHT_SUPERIOR, {"rule_subtype", "rule_required_subtype"}),
            (KAMOA_KAKULA,     {"rule_subtype", "rule_required_subtype"}),
            (LA_GRANJA,        {"rule_subtype", "rule_required_subtype"}),
            # Different commodity — caught by either L1 or rule_subtype since
            # Goldstrike's carlin_general / Ramu's limonite_laterite are in the
            # alkalic rule's excluded_subtypes (defence in depth before L1).
            (GOLDSTRIKE,       {"L1", "rule_subtype", "rule_required_subtype"}),
            (RAMU_LATERITE,    {"L1", "rule_subtype", "rule_required_subtype"}),
        ],
    },
    {
        "name": "orogenic gold → Abitibi/Yilgarn/Birimian, drop epi/Carlin/porphyry",
        "rule_id": "analog_sel_gold_orogenic",
        "target": ABITIBI_GOLD_TARGET,
        "must_pick": [DETOUR_LAKE, SUNRISE_DAM, KIBALI],
        "must_drop": [
            (YANACOCHA,    {"rule_subtype", "rule_required_subtype", "L2"}),
            (GOLDSTRIKE,   {"rule_subtype", "rule_required_subtype", "L2"}),
            (BINGHAM_AU,   {"rule_subtype", "rule_required_subtype", "L2"}),
            # Mt. Milligan is copper not gold — caught at L1 or by the
            # orogenic rule's excluded_subtypes (alkalic_porphyry).
            (MT_MILLIGAN,  {"L1", "rule_subtype", "rule_required_subtype"}),
        ],
    },
    {
        "name": "magmatic Ni sulphide → Sudbury/Kambalda, drop laterite",
        "rule_id": "analog_sel_nickel_magmatic_sulphide",
        "target": NI_SULPHIDE_TARGET,
        "must_pick": [SUDBURY, KAMBALDA],
        "must_drop": [
            (RAMU_LATERITE, {"rule_subtype", "rule_required_subtype", "rule_mode",
                              "rule_recovery", "L2"}),
            (GORO_LATERITE, {"rule_subtype", "rule_required_subtype", "rule_mode",
                              "rule_recovery", "L2"}),
            # Marimaca is copper not nickel — caught at L1 or rule_subtype
            # (its iocg_oxide is in the magmatic-sulphide rule's exclusions).
            (MARIMACA,      {"L1", "rule_subtype", "rule_required_subtype"}),
        ],
    },
]
