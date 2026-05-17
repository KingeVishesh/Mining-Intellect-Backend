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


# ── More analogs for Liberty Gold Black Pine + 1911 Gold True North ─────────

# Black Pine target (super-large bulk Carlin)
BLACK_PINE_TARGET = {
    "name": "Black Pine", "material": "gold",
    "deposit_type": "Carlin-style sediment-hosted disseminated gold",
    "deposit_subtype": "carlin_general",
    "mineralization_mode": "supergene_oxide",
    "mineralization_pattern": "disseminated_bulk",
    "host_rock_class": "carbonate_sediment",
    "tectonic_belt": "great_basin_carlin",
    "recovery_method": "cn_leach",
    "metal_suite": "au_only",
    "country": "USA", "region": "Idaho", "district": "Black Pine",
    "mineralization_style": "bulk low-grade disseminated gold in arsenian pyrite",
    "host_rock": "Pennsylvanian limestone Oquirrh Formation",
    "processing_method": "heap leach",
    "tonnage_mt": 500.0, "grade_value": 0.3, "grade_unit": "g/t Au",
}

# Carlin super-large analogs (correct picks)
MARIGOLD = {
    "name": "Marigold", "material": "gold",
    "deposit_type": "Carlin-style sediment-hosted disseminated",
    "deposit_subtype": "carlin_general",
    "mineralization_mode": "supergene_oxide",
    "mineralization_pattern": "disseminated_bulk",
    "host_rock_class": "clastic_sediment",
    "tectonic_belt": "great_basin_carlin",
    "recovery_method": "cn_leach",
    "metal_suite": "au_only",
    "country": "USA", "region": "Nevada", "district": "Battle Mountain",
    "mineralization_style": "disseminated gold in oxidized arsenian pyrite",
    "host_rock": "siltstone limestone",
    "processing_method": "heap leach",
    "tonnage_mt": 740.0, "grade_value": 0.42, "grade_unit": "g/t Au",
}
ROUND_MOUNTAIN = {
    "name": "Round Mountain", "material": "gold",
    "deposit_type": "Carlin-style bulk-disseminated",
    "deposit_subtype": "carlin_general",
    "mineralization_mode": "supergene_oxide",
    "mineralization_pattern": "disseminated_bulk",
    "host_rock_class": "carbonate_sediment",
    "tectonic_belt": "great_basin_carlin",
    "recovery_method": "cn_leach",
    "metal_suite": "au_only",
    "country": "USA", "region": "Nevada", "district": "Toiyabe",
    "mineralization_style": "bulk low-grade disseminated",
    "host_rock": "ash-flow tuff (volcanic-sedimentary)",
    "processing_method": "heap leach",
    "tonnage_mt": 800.0, "grade_value": 0.5, "grade_unit": "g/t Au",
}
# Pan Mine — geologically correct Carlin but TOO SMALL for super-large rule.
PAN_MINE = {
    "name": "Pan Mine", "material": "gold",
    "deposit_type": "Carlin-style sediment-hosted disseminated",
    "deposit_subtype": "carlin_general",
    "mineralization_mode": "supergene_oxide",
    "mineralization_pattern": "disseminated_bulk",
    "host_rock_class": "carbonate_sediment",
    "tectonic_belt": "great_basin_carlin",
    "recovery_method": "cn_leach",
    "metal_suite": "au_only",
    "country": "USA", "region": "Nevada", "district": "White Pine",
    "mineralization_style": "disseminated invisible gold",
    "host_rock": "limestone",
    "processing_method": "heap leach",
    "tonnage_mt": 52.0, "grade_value": 0.45, "grade_unit": "g/t Au",
}
# Wrong analogs for Black Pine
TRIXIE = {
    "name": "Trixie", "material": "gold",
    "deposit_type": "polymetallic vein/replacement",
    "deposit_subtype": "crd",
    "mineralization_mode": "primary_sulfide",
    "mineralization_pattern": "replacement",
    "host_rock_class": "carbonate_sediment",
    "tectonic_belt": "great_basin_carlin",
    "recovery_method": "flotation",
    "metal_suite": "ag_pb_zn",
    "country": "USA", "region": "Utah", "district": "Tintic",
    "mineralization_style": "high-grade quartz vein, base-metal sulphides",
    "host_rock": "limestone carbonate",
    "processing_method": "flotation",
    "tonnage_mt": 5.0, "grade_value": 8.0, "grade_unit": "g/t Au",
}
FLORIDA_CANYON = {
    "name": "Florida Canyon", "material": "gold",
    "deposit_type": "low-sulfidation epithermal gold",
    "deposit_subtype": "low_sulfidation_epithermal",
    "mineralization_mode": "supergene_oxide",
    "mineralization_pattern": "vein_hosted",
    "host_rock_class": "clastic_sediment",
    "tectonic_belt": "great_basin_carlin",
    "recovery_method": "heap_leach",
    "metal_suite": "au_ag",
    "country": "USA", "region": "Nevada", "district": "Humboldt",
    "mineralization_style": "stockwork quartz veins/microveinlets, epithermal",
    "host_rock": "Triassic siltstone mudstone",
    "processing_method": "heap leach",
    "tonnage_mt": 150.0, "grade_value": 0.6, "grade_unit": "g/t Au",
}

# True North target (vein-orogenic)
TRUE_NORTH_TARGET = {
    "name": "True North", "material": "gold",
    "deposit_type": "Archean orogenic gold",
    "deposit_subtype": "orogenic_general",
    "mineralization_mode": "primary_sulfide",
    "mineralization_pattern": "vein_hosted",
    "host_rock_class": "intrusive_mafic",
    "tectonic_belt": "abitibi",
    "recovery_method": "cil_cip",
    "metal_suite": "au_only",
    "country": "Canada", "region": "Manitoba", "district": "Rice Lake greenstone",
    "mineralization_style": "quartz-carbonate veins shear-hosted free gold",
    "host_rock": "San Antonio gabbro sill, greenstone volcanics",
    "processing_method": "CIL",
    "tonnage_mt": 5.0, "grade_value": 6.0, "grade_unit": "g/t Au",
}
# Brucejack — BC Cordilleran arc orogenic vein gold. Geologically vein-
# hosted underground gold like True North, but in a DIFFERENT belt-
# compatibility group (phanerozoic_arc, not archean_greenstone). With the
# 2026-05 belt hard filter, Brucejack is correctly NOT a must_pick for a
# True North-style Manitoba/Abitibi target — it would mislead modelling.
# Kept here as a must_drop reference (drops at L2.5).
BRUCEJACK = {
    "name": "Brucejack", "material": "gold",
    "deposit_type": "orogenic gold high-grade veins",
    "deposit_subtype": "orogenic_general",
    "mineralization_mode": "primary_sulfide",
    "mineralization_pattern": "vein_hosted",
    "host_rock_class": "volcanic_mafic",
    "tectonic_belt": "bc_quesnel_stikine",
    "recovery_method": "cil_cip",
    "metal_suite": "au_only",
    "country": "Canada", "region": "British Columbia", "district": "Sulphurets",
    "mineralization_style": "quartz-carbonate veins shear-hosted free gold",
    "host_rock": "andesite volcanic",
    "processing_method": "CIL",
    "tonnage_mt": 16.0, "grade_value": 8.4, "grade_unit": "g/t Au",
}
# Macassa — Kirkland Lake Ontario Abitibi, high-grade vein orogenic gold.
# Same belt group as True North; sibling subtype. Scale (14 Mt) fits within
# the orogenic_vein rule's 5× tonnage tolerance vs a 5 Mt True North target
# (ratio 2.8×).
MACASSA_FIXTURE = {
    "name": "Macassa Mine", "material": "gold",
    "deposit_type": "high-grade orogenic quartz vein gold",
    "deposit_subtype": "greenstone_orogenic",
    "mineralization_mode": "primary_sulfide",
    "mineralization_pattern": "vein_hosted",
    "host_rock_class": "intrusive_felsic",
    "tectonic_belt": "abitibi",
    "recovery_method": "cil_cip",
    "metal_suite": "au_only",
    "country": "Canada", "region": "Ontario", "district": "Kirkland Lake",
    "mineralization_style": "high-grade quartz-carbonate veins in syenite",
    "host_rock": "syenite intrusion cutting mafic volcanic",
    "processing_method": "CIL",
    "tonnage_mt": 14.0, "grade_value": 21.0, "grade_unit": "g/t Au",
}
RED_LAKE = {
    "name": "Red Lake", "material": "gold",
    "deposit_type": "Archean orogenic gold veins",
    "deposit_subtype": "greenstone_orogenic",
    "mineralization_mode": "primary_sulfide",
    "mineralization_pattern": "vein_hosted",
    "host_rock_class": "metamorphic_low_grade",
    "tectonic_belt": "abitibi",
    "recovery_method": "cil_cip",
    "metal_suite": "au_only",
    "country": "Canada", "region": "Ontario", "district": "Red Lake greenstone",
    "mineralization_style": "quartz-carbonate veins shear-hosted",
    "host_rock": "Archean greenstone schist",
    "processing_method": "CIL",
    "tonnage_mt": 8.0, "grade_value": 13.0, "grade_unit": "g/t Au",
}
# Wrong analogs for True North
SPRINGPOLE = {
    "name": "Springpole", "material": "gold",
    "deposit_type": "alkaline porphyry-epithermal hybrid",
    "deposit_subtype": "orogenic_general",
    "mineralization_mode": "primary_sulfide",
    "mineralization_pattern": "disseminated_bulk",
    "host_rock_class": "intrusive_felsic",
    "tectonic_belt": "abitibi",
    "recovery_method": "flotation",
    "metal_suite": "au_ag",
    "country": "Canada", "region": "Ontario", "district": "Birch-Uchi greenstone",
    "mineralization_style": "bulk-tonnage low-grade disseminated, potassic biotite",
    "host_rock": "trachyte porphyry, diatreme breccia",
    "processing_method": "flotation",
    "tonnage_mt": 191.0, "grade_value": 0.78, "grade_unit": "g/t Au",
}
WHITE_GOLD = {
    "name": "White Gold (Golden Saddle)", "material": "gold",
    "deposit_type": "orogenic gold gneiss-hosted",
    "deposit_subtype": "orogenic_general",
    "mineralization_mode": "primary_sulfide",
    "mineralization_pattern": "breccia_hosted",
    "host_rock_class": "metamorphic_high_grade",
    "tectonic_belt": "yukon_tintina",
    "recovery_method": "cil_cip",
    "metal_suite": "au_only",
    "country": "Canada", "region": "Yukon", "district": "White Gold",
    "mineralization_style": "high-grade breccia + disseminated in altered gneiss",
    "host_rock": "felsic orthogneiss amphibolite",
    "processing_method": "CIL",
    "tonnage_mt": 30.0, "grade_value": 1.5, "grade_unit": "g/t Au",
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
        "rule_id": "analog_sel_gold_orogenic_vein",
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
    {
        # Liberty Gold Black Pine — super-large Carlin. Picks Marigold &
        # Round Mountain. Drops Pan at scale_mismatch (L5.5), Trixie at
        # rule_pattern (replacement), Florida Canyon at rule_subtype
        # (epithermal not Carlin).
        "name": "Black Pine — super-large Carlin",
        "rule_id": "analog_sel_gold_carlin_super_large",
        "target": BLACK_PINE_TARGET,
        "must_pick": [MARIGOLD, ROUND_MOUNTAIN],
        "must_drop": [
            (PAN_MINE,        {"L5.5"}),       # geologically perfect but too small
            # Trixie hits rule_recovery first (flotation excluded for Carlin super-large)
            # — its replacement pattern would catch it too at rule_pattern.
            (TRIXIE,          {"rule_recovery", "rule_pattern", "rule_subtype",
                                 "rule_required_pattern", "rule_required_subtype"}),
            (FLORIDA_CANYON,  {"rule_pattern", "rule_subtype",
                                "rule_required_pattern", "rule_required_subtype"}),
        ],
    },
    {
        # 1911 Gold True North — vein-orogenic Manitoba Rice Lake (Archean
        # greenstone). Picks Red Lake (Abitibi, same group) and Casa Berardi
        # (Abitibi, same group). Drops:
        #   - Brucejack at L2.5 (BC Cordilleran arc, different belt group)
        #   - White Gold at L2.5 (Yukon Tintina, Cordilleran arc)
        #   - Springpole at rule_required_pattern (disseminated bulk)
        # The belt hard filter is the key change from the May-2026 audit:
        # cross-belt orogenic vein gold analogs were producing wrong
        # matches for Cartier-Cadillac (Brucejack, Aurora/Toroparu/Rosebel).
        "name": "True North — vein-orogenic",
        "rule_id": "analog_sel_gold_orogenic_vein",
        "target": TRUE_NORTH_TARGET,
        "must_pick": [RED_LAKE, MACASSA_FIXTURE],
        "must_drop": [
            (BRUCEJACK,       {"L2.5"}),
            (SPRINGPOLE,      {"rule_pattern", "rule_required_pattern", "L4.5"}),
            (WHITE_GOLD,      {"rule_pattern", "rule_required_pattern", "L4.5", "L2.5"}),
            # Marigold is gold but bulk Carlin not orogenic vein — drops at
            # subtype level.
            (MARIGOLD,        {"rule_subtype", "rule_required_subtype",
                                "rule_pattern", "rule_required_pattern", "L2", "L3"}),
        ],
    },
]
