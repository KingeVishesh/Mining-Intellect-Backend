"""
Geological taxonomy — controlled vocabularies and detection heuristics.

Used by:
  * field_extractor.py — populates the new schema columns at research time
  * analog_finder.py   — runtime fallback when columns are null on existing rows
  * backfill_geological_profiles.py — populates new columns for existing rows

Detection functions are intentionally heuristic substring matchers over freeform
text. They are not perfect; the LLM is the primary extractor. These exist to
fill the gap on rows that have only old-style freeform `deposit_type`,
`mineralization_style`, `processing_method`, etc.
"""
from __future__ import annotations
from typing import Optional, Dict, List, FrozenSet


# ── Vocabularies ─────────────────────────────────────────────────────────────
#
# SINGLE SOURCE OF TRUTH for every geological vocabulary used by the analog
# finder. Every detect_*() heuristic, every rule's required/excluded slugs,
# every Grok prompt enumeration, every Pydantic schema validator MUST derive
# from these constants. Drift is prevented because there is no second copy.

# Subtype → family map. The canonical list of every deposit sub-type slug used
# at runtime, paired with its broad deposit family for L2 family-gate logic.
# When `analog_finder._cascading_match()` checks "same deposit family" it now
# looks up by subtype slug here rather than re-deriving from freeform text.
SUBTYPE_TO_FAMILY: Dict[str, str] = {
    # Porphyry family
    "alkalic_porphyry":                    "porphyry",
    "calc_alkalic_porphyry":               "porphyry",
    "laramide_porphyry":                   "porphyry",
    "high_sulfidation_lithocap_porphyry":  "porphyry",
    # Oxide ISCR — semantically distinct family (different metallurgy)
    "oxide_iscr_supergene_blanket":        "oxide_iscr",
    # IOCG family
    "iocg_oxide":                          "iocg",
    "iocg_sulfide":                        "iocg",
    "iocg_hybrid":                         "iocg",
    # Epithermal family
    "low_sulfidation_epithermal":          "epithermal",
    "high_sulfidation_epithermal":         "epithermal",
    "intermediate_sulfidation_epithermal": "epithermal",
    # Orogenic family
    "greenstone_orogenic":                 "orogenic",
    "turbidite_orogenic":                  "orogenic",
    "bif_hosted_orogenic":                 "orogenic",
    "orogenic_general":                    "orogenic",
    # Sediment-hosted family
    "sedex":                               "sediment_hosted",
    "kupferschiefer_style":                "sediment_hosted",
    "manto_cu":                            "sediment_hosted",
    "crd":                                 "sediment_hosted",
    "mvt":                                 "sediment_hosted",
    "redbed_cu":                           "sediment_hosted",
    "sediment_hosted_general":             "sediment_hosted",
    # VMS family
    "vms_general":                         "vms",
    # Carlin family
    "carlin_general":                      "carlin",
    # Skarn family
    "cu_au_skarn":                         "skarn",
    "fe_skarn":                            "skarn",
    "zn_pb_skarn":                         "skarn",
    "w_mo_skarn":                          "skarn",
    "skarn_general":                       "skarn",
    # PGM reef family
    "merensky_reef":                       "pgm_reef",
    "ug2_reef":                            "pgm_reef",
    "platreef":                            "pgm_reef",
    # Laterite family
    "limonite_laterite":                   "laterite",
    "saprolite_laterite":                  "laterite",
    "laterite_general":                    "laterite",
    # Magmatic sulphide family
    "komatiite_hosted":                    "magmatic_sulphide",
    "conduit_hosted":                      "magmatic_sulphide",
    "magmatic_sulphide_general":           "magmatic_sulphide",
    # BIF family
    "bif_general":                         "bif",
}

# Flat set of every valid subtype slug — derived from SUBTYPE_TO_FAMILY so the
# two can never drift. Used by Pydantic validators, field_extractor validation,
# and Grok prompt enum generation.
ALL_SUBTYPE_SLUGS: FrozenSet[str] = frozenset(SUBTYPE_TO_FAMILY.keys())

# Set of distinct family names (porphyry, iocg, epithermal, …). Derived.
ALL_FAMILY_SLUGS: FrozenSet[str] = frozenset(SUBTYPE_TO_FAMILY.values())

# Kept for backward-compatibility / documentation. NOT the runtime gate.
# (The previous nested structure used parts of slugs like "alkalic" rather than
# compound slugs like "alkalic_porphyry", which made it dead code.)
DEPOSIT_SUBTYPES: Dict[str, List[str]] = {
    family: sorted(s for s, f in SUBTYPE_TO_FAMILY.items() if f == family)
    for family in sorted(ALL_FAMILY_SLUGS)
}


# Mineralization modes — primary form of the ore
MINERALIZATION_MODES: List[str] = [
    "primary_sulfide",
    "supergene_oxide",
    "mixed_oxide_sulfide",
    "refractory_sulfide",
    "free_milling_oxide",
    "placer",
]
ALL_MODE_SLUGS: FrozenSet[str] = frozenset(MINERALIZATION_MODES)


# Mineralization patterns — the GEOMETRY of the orebody, orthogonal to
# subtype/mode. Carlin can be bulk-disseminated (Marigold) OR replacement
# (Trixie). Orogenic gold can be vein-hosted (Brucejack) OR bulk-disseminated
# (Springpole). The cascade L4.5 hard-filters on rule.required_patterns so
# orogenic-vein projects never get orogenic-bulk analogs and vice versa.
MINERALIZATION_PATTERNS: List[str] = [
    "disseminated_bulk",   # Marigold, Black Pine, Springpole, Douay
    "vein_hosted",         # Brucejack, Red Lake, Fosterville
    "stockwork",           # Most porphyry cores; Tower Gold mixed
    "breccia_hosted",      # Some IOCG, White Gold
    "replacement",         # CRD, manto, Trixie
    "massive_sulphide",    # VMS lenses
    "reef",                # Merensky, UG2, Platreef
    "placer",              # alluvial gold
    "blanket",             # supergene oxide / laterite
]
ALL_PATTERN_SLUGS: FrozenSet[str] = frozenset(MINERALIZATION_PATTERNS)


# Host rock classes — coarse classification of the dominant host lithology,
# orthogonal to deposit type. Orogenic gold in gabbro shear zones (True North,
# Brucejack) mines differently from orogenic gold in gneiss breccia (White
# Gold). Carlin gold in carbonate sediments (Black Pine) is non-comparable to
# Carlin-style gold in clastic siltstones (Pipeline) when modelling continuity.
HOST_ROCK_CLASSES: List[str] = [
    "carbonate_sediment",     # limestone, dolomite — Carlin host
    "clastic_sediment",       # sandstone, siltstone, shale — sediment-hosted Cu, roll-front U
    "volcanic_mafic",         # basalt, gabbro flows, komatiite
    "volcanic_felsic",        # rhyolite, dacite tuff — HS epithermal cap rock
    "intrusive_mafic",        # gabbro, diorite — magmatic Ni-Cu, alkalic porphyry hosts
    "intrusive_felsic",       # granite, granodiorite, syenite — porphyry, intrusive U
    "metamorphic_high_grade", # gneiss, amphibolite — Yukon White Gold
    "metamorphic_low_grade",  # greenstone, schist — Abitibi, Yilgarn orogenic gold
    "bif",                    # banded iron formation — iron ore, BIF-hosted Au
    "carbonatite",            # rare earths
    "ultramafic",             # peridotite, dunite — laterite Ni, magmatic Ni
]
ALL_HOST_CLASS_SLUGS: FrozenSet[str] = frozenset(HOST_ROCK_CLASSES)


# Project stage classes — drives the L4.6 hard filter. Comparing analogs at
# different stages of development to the target inflates or deflates resource
# estimates (production reconciliation vs greenfield drill grid).
PROJECT_STAGES: List[str] = [
    "exploration",            # grass-roots to advanced exploration, no resource
    "resource_inferred",      # initial inferred resource only
    "resource_m_and_i",       # measured + indicated resource defined
    "pea",                    # preliminary economic assessment / scoping
    "pfs",                    # pre-feasibility
    "feasibility",            # bankable feasibility / DFS
    "construction",
    "production",
    "care_maintenance",       # closed but resource still on the books
    "closed",                 # rehabilitated, no future production
]
ALL_STAGE_SLUGS: FrozenSet[str] = frozenset(PROJECT_STAGES)


# Mining method classes — drives the L4.8 hard filter. An open-pit bulk
# project should not be analogged by an underground vein mine and vice versa
# (cut-off grade, dilution, recovery, capex all differ fundamentally).
MINING_METHOD_CLASSES: List[str] = [
    "open_pit_bulk",          # bulk-tonnage open pit (Marigold, Bingham)
    "open_pit_selective",     # selective open pit with high stripping (Carlin trend)
    "underground_vein",       # narrow-vein UG (Brucejack, Red Lake)
    "underground_bulk",       # bulk UG (sublevel caving, panel caving)
    "block_cave",             # very-bulk UG (Cadia East, El Teniente)
    "iscr_in_situ",           # in-situ leach (uranium, oxide Cu)
    "heap_leach_pad",         # surface heap-leach (oxide gold, oxide copper)
    "dredging",               # placer / alluvial
    "highwall",               # specific bulk variant
    "solution_mining",        # potash, salt
]
ALL_MINING_METHOD_SLUGS: FrozenSet[str] = frozenset(MINING_METHOD_CLASSES)


# Resource category classes — drives the L4.9 hard filter. Inferred-only
# analogs are weaker than M&I analogs; modelling against an inferred-only
# reference inflates uncertainty in the wrong direction.
RESOURCE_CATEGORY_CLASSES: List[str] = [
    "measured",
    "indicated",
    "m_and_i",                # measured + indicated combined
    "inferred",
    "m_and_i_plus_inferred",  # M+I+I total resource
    "reserve_proven",
    "reserve_probable",
    "reserve_p_and_p",
    "exploration_target",     # NOT a resource, lowest reliability
    "historical",             # non-compliant historical resource
]
ALL_RESOURCE_CATEGORY_SLUGS: FrozenSet[str] = frozenset(RESOURCE_CATEGORY_CLASSES)


# Resource compliance standards — drives the L4.95 hard filter. Modern
# compliant resources (NI 43-101 post-2010, JORC 2012, SK-1300) are
# substantially different documents from a 1985 press-release "tonnage".
RESOURCE_COMPLIANCE_STANDARDS: List[str] = [
    "ni_43_101",
    "jorc",
    "sk_1300",
    "samrec",
    "pera",                   # Peruvian / Russian / other regional codes
    "historical",             # explicitly non-compliant historical estimate
    "press_release",          # company announcement, no qualified-person sign-off
    "internal",               # company-internal, never publicly compliant
]
ALL_COMPLIANCE_SLUGS: FrozenSet[str] = frozenset(RESOURCE_COMPLIANCE_STANDARDS)


# Stage compatibility — analogs MUST share a stage class with the target
# OR be one of these compatible substitutes. Production-stage projects can
# inform M&I-stage targets (lots of reconciliation data); the reverse is
# weaker but acceptable.
STAGE_COMPATIBILITY: Dict[str, FrozenSet[str]] = {
    "exploration":         frozenset({"exploration", "resource_inferred"}),
    "resource_inferred":   frozenset({"exploration", "resource_inferred", "resource_m_and_i"}),
    "resource_m_and_i":    frozenset({"resource_inferred", "resource_m_and_i", "pea", "pfs"}),
    "pea":                 frozenset({"resource_m_and_i", "pea", "pfs", "feasibility"}),
    "pfs":                 frozenset({"pea", "pfs", "feasibility", "construction"}),
    "feasibility":         frozenset({"pfs", "feasibility", "construction", "production"}),
    "construction":        frozenset({"feasibility", "construction", "production"}),
    "production":          frozenset({"production", "feasibility", "construction", "care_maintenance"}),
    "care_maintenance":    frozenset({"production", "care_maintenance"}),
    "closed":              frozenset({"closed", "care_maintenance"}),
}


# Mining-method incompatibility — pairs that absolutely cannot substitute.
# Underground vein mining has no shared assumptions with surface ISCR.
MINING_METHOD_INCOMPATIBILITY: Dict[str, FrozenSet[str]] = {
    "open_pit_bulk":      frozenset({"underground_vein", "iscr_in_situ"}),
    "open_pit_selective": frozenset({"underground_vein", "iscr_in_situ"}),
    "underground_vein":   frozenset({"open_pit_bulk", "open_pit_selective",
                                       "iscr_in_situ", "heap_leach_pad", "block_cave"}),
    "underground_bulk":   frozenset({"underground_vein", "iscr_in_situ",
                                       "heap_leach_pad", "dredging"}),
    "block_cave":         frozenset({"underground_vein", "iscr_in_situ",
                                       "heap_leach_pad", "dredging"}),
    "iscr_in_situ":       frozenset({"open_pit_bulk", "open_pit_selective",
                                       "underground_vein", "underground_bulk",
                                       "block_cave", "heap_leach_pad", "dredging"}),
    "heap_leach_pad":     frozenset({"underground_vein", "iscr_in_situ", "dredging"}),
    "dredging":           frozenset({"underground_vein", "underground_bulk",
                                       "iscr_in_situ", "heap_leach_pad", "block_cave"}),
}


# Resource-category compatibility — for L4.9 hard filter when a rule pins
# min_resource_category. Inferred-only analogs are rejected when the rule
# demands at least M&I quality.
RESOURCE_CATEGORY_RANK: Dict[str, int] = {
    "reserve_p_and_p":         100,
    "reserve_proven":          95,
    "reserve_probable":        90,
    "m_and_i":                 80,
    "measured":                85,
    "indicated":               75,
    "m_and_i_plus_inferred":   70,
    "inferred":                40,
    "historical":              20,
    "exploration_target":      10,
}


# Tectonic belts — mineralization provinces with shared genesis. Lookup is by
# (country, region|district). Order matters: more specific belts checked first.
TECTONIC_BELTS: Dict[str, Dict[str, List[str]]] = {
    "bc_quesnel_stikine": {
        "countries": ["canada"],
        "regions": ["british columbia", "bc", "quesnel", "stikine", "golden triangle",
                    "iskut", "babine", "toodoggone"],
    },
    "yukon_tintina": {
        "countries": ["canada", "usa"],
        "regions": ["yukon", "alaska", "tintina"],
    },
    "abitibi": {
        "countries": ["canada"],
        "regions": ["ontario", "quebec", "abitibi", "timmins", "kirkland lake", "val d'or", "rouyn"],
    },
    "newfoundland_appalachian": {
        "countries": ["canada"],
        "regions": ["newfoundland", "labrador", "nova scotia", "new brunswick", "appalachian"],
    },
    "laramide_southwest": {
        "countries": ["usa", "mexico"],
        "regions": ["arizona", "new mexico", "utah", "sonora", "chihuahua", "laramide"],
    },
    "great_basin_carlin": {
        "countries": ["usa"],
        "regions": [
            # Nevada — core Carlin Trend and outliers
            "nevada", "carlin", "battle mountain", "eureka",
            "cortez", "getchell", "humboldt", "white pine",
            # Idaho/Utah — Oquirrh Formation Carlin systems (Black Pine,
            # Long Canyon, parts of Carlin Trend extension). The geology
            # is continuous across state lines.
            "idaho", "utah", "oquirrh", "oneida", "elko",
        ],
    },
    "andean": {
        "countries": ["chile", "peru", "argentina", "ecuador", "bolivia", "colombia"],
        "regions": ["andes", "atacama", "antofagasta", "tarapacá"],
    },
    "brazilian_shield": {
        "countries": ["brazil"],
        "regions": ["carajás", "carajas", "minas gerais", "bahia", "goiás"],
    },
    "central_african_copperbelt": {
        "countries": [
            "zambia", "drc", "dr congo", "congo",
            "democratic republic of the congo", "democratic republic of congo",
        ],
        "regions": ["copperbelt", "katanga", "lualaba"],
    },
    "bushveld": {
        "countries": ["south africa"],
        "regions": ["bushveld", "rustenburg", "merensky", "ug2", "platreef"],
    },
    "west_african_birimian": {
        "countries": ["ghana", "mali", "burkina faso", "senegal", "ivory coast", "guinea"],
        "regions": ["birimian", "ashanti"],
    },
    "tanzania_archean": {
        "countries": ["tanzania", "kenya"],
        "regions": ["lake victoria", "geita", "musoma"],
    },
    "lachlan": {
        "countries": ["australia"],
        "regions": ["new south wales", "nsw", "lachlan", "cadia", "macquarie arc"],
    },
    "yilgarn": {
        "countries": ["australia"],
        "regions": ["western australia", "wa", "yilgarn", "kalgoorlie", "perth"],
    },
    "fennoscandian": {
        "countries": ["finland", "norway", "sweden"],
        "regions": [
            "fennoscandian", "baltic shield",
            # Finnish gold/Cu/Ni belts
            "lapland", "central lapland", "kuusamo", "kittilä", "kittila",
            "outokumpu", "ostrobothnia", "kuhmo",
            # Swedish belts
            "skellefte", "skellefteå", "skelleftea", "norrbotten",
            "kiruna", "bergslagen",
            # Norwegian
            "finnmark",
        ],
    },
    "central_asian_orogenic": {
        "countries": ["kazakhstan", "uzbekistan", "kyrgyzstan", "mongolia", "russia"],
        "regions": ["tien shan", "altaid", "gobi"],
    },
    "indonesia_philippines_arc": {
        "countries": ["indonesia", "philippines", "papua new guinea"],
        "regions": ["sumatra", "sulawesi", "halmahera", "luzon", "mindanao"],
    },
    "new_caledonia_laterite": {
        "countries": ["new caledonia"],
        "regions": ["new caledonia"],
    },
    "iberian_pyrite": {
        "countries": ["portugal", "spain"],
        "regions": ["iberian pyrite belt", "ipb", "alentejo", "huelva"],
    },
}
ALL_BELT_SLUGS: FrozenSet[str] = frozenset(TECTONIC_BELTS.keys())


# Metal suites — characteristic byproduct/co-product patterns
METAL_SUITES: List[str] = [
    "cu_au",          # Cu-Au porphyry, IOCG
    "cu_mo",          # Cu-Mo porphyry
    "cu_au_co_sc",    # Doubleview Hat-style alkalic with critical-metal byproducts
    "cu_ag",          # sediment-hosted Cu-Ag
    "cu_zn_pb",       # VMS polymetallic
    "au_only",        # orogenic gold, Carlin
    "au_ag",          # epithermal precious
    "ag_pb_zn",       # sediment-hosted Ag-Pb-Zn
    "ni_cu_pge",      # magmatic sulphide
    "ni_co",          # laterite nickel
    "pt_pd_rh",       # PGM reef
    "u_only",         # uranium-dominant
    "fe_only",        # iron-dominant (BIF, magnetite)
    "li_only",        # lithium brines/pegmatites
    "ree_only",       # rare-earth dominant
]
ALL_SUITE_SLUGS: FrozenSet[str] = frozenset(METAL_SUITES)


# Alteration signatures — characteristic assemblages
ALTERATION_SIGNATURES: List[str] = [
    "potassic_phyllic",          # alkalic + calc-alkalic porphyry core
    "potassic_propylitic",       # outer porphyry halo
    "sodic_calcic",              # IOCG, alkalic-affinity intrusion-related
    "hematite_specularite",      # IOCG oxidized
    "argillic_advanced_argillic", # high-sulfidation epithermal
    "sericitic_quartz",          # mesothermal / orogenic gold
    "skarn_calc_silicate",       # garnet-pyroxene-epidote skarn
    "silicification_decalcified", # Carlin-type
    "chlorite_carbonate",        # VMS / orogenic
    "supergene_oxidation",       # weathering blanket
    "lateritic_weathering",      # laterite Ni
]
ALL_ALTERATION_SLUGS: FrozenSet[str] = frozenset(ALTERATION_SIGNATURES)


# Recovery methods
RECOVERY_METHODS: List[str] = [
    "flotation",
    "heap_leach",
    "iscr",          # in-situ copper recovery
    "sx_ew",         # solvent extraction / electrowinning
    "cn_leach",      # cyanide leach (Au/Ag)
    "cil_cip",       # carbon-in-leach / carbon-in-pulp
    "gravity",       # placer gold, coarse-particle PGM
    "smelting",      # direct-shipping iron ore, some Cu concentrate
    "atmospheric_leach", # laterite Ni atmospheric leach
    "hpal",          # high-pressure acid leach (laterite Ni)
]
ALL_RECOVERY_SLUGS: FrozenSet[str] = frozenset(RECOVERY_METHODS)


# Recovery incompatibility — pairs that cannot substitute for each other
# (different metallurgical regime, different reagent stack, different capex profile)
RECOVERY_INCOMPATIBILITY: Dict[str, FrozenSet[str]] = {
    "flotation":         frozenset({"heap_leach", "iscr", "sx_ew", "hpal", "atmospheric_leach"}),
    "heap_leach":        frozenset({"flotation", "iscr"}),
    "iscr":              frozenset({"flotation", "heap_leach", "smelting"}),
    "sx_ew":             frozenset({"flotation", "smelting"}),
    "cn_leach":          frozenset({"flotation", "iscr", "hpal"}),
    "cil_cip":           frozenset({"flotation", "iscr", "hpal"}),
    "hpal":              frozenset({"flotation", "heap_leach", "iscr", "cn_leach"}),
    "atmospheric_leach": frozenset({"flotation", "iscr"}),
    "smelting":          frozenset({"iscr", "sx_ew", "cn_leach", "cil_cip"}),
    "gravity":           frozenset(),  # gravity is compatible with most circuits as a pre-step
}


# ── Detection helpers ────────────────────────────────────────────────────────

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def detect_subtype(
    deposit_type: Optional[str],
    mineralization_style: Optional[str] = None,
    alteration: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[str]:
    """
    Detect a finer-grained sub-type from freeform text. Returns slug like 'alkalic_porphyry'.
    Returns None when no confident match.

    Strategy: look for the most specific signal across all input strings.
    """
    blob = " ".join(filter(None, [
        _norm(deposit_type), _norm(mineralization_style),
        _norm(alteration), _norm(description),
    ]))
    if not blob:
        return None

    # Oxide ISCR — check BEFORE porphyry. Florence/Van Dyke have porphyry roots
    # but are modeled as oxide ISCR blankets. ISCR / SX-EW / supergene oxide blanket
    # signals override the porphyry classification because the metallurgy and
    # resource modeling are fundamentally different.
    if ("iscr" in blob or "in-situ copper recovery" in blob or "in situ recovery" in blob
            or ("oxide" in blob and ("blanket" in blob or "supergene" in blob)
                and ("sx-ew" in blob or "sx/ew" in blob or "sx_ew" in blob))):
        return "oxide_iscr_supergene_blanket"

    # Porphyry sub-types (order matters: most specific first)
    if "porphyry" in blob:
        if "alkalic" in blob or "alkaline" in blob:
            return "alkalic_porphyry"
        if "laramide" in blob or ("arizona" in blob and "oxide" in blob):
            return "laramide_porphyry"
        if "high sulphidation" in blob or "high-sulfidation" in blob or "lithocap" in blob:
            return "high_sulfidation_lithocap_porphyry"
        return "calc_alkalic_porphyry"

    # IOCG — separated by oxide vs sulfide vs hybrid
    if "iocg" in blob or "iron oxide copper" in blob or "iron-oxide-copper-gold" in blob:
        if "oxide" in blob and "sulfide" not in blob and "sulphide" not in blob:
            return "iocg_oxide"
        if ("sulfide" in blob or "sulphide" in blob) and "oxide" not in blob:
            return "iocg_sulfide"
        return "iocg_hybrid"

    # Epithermal sub-types
    if "epithermal" in blob:
        if "high sulphidation" in blob or "high-sulfidation" in blob or "hs " in blob:
            return "high_sulfidation_epithermal"
        if "low sulphidation" in blob or "low-sulfidation" in blob or "ls " in blob:
            return "low_sulfidation_epithermal"
        if "intermediate" in blob:
            return "intermediate_sulfidation_epithermal"

    # Orogenic gold sub-types
    if "orogenic" in blob or "mesothermal" in blob or "lode gold" in blob:
        if "greenstone" in blob:
            return "greenstone_orogenic"
        if "turbidite" in blob:
            return "turbidite_orogenic"
        if "bif" in blob and "host" in blob:
            return "bif_hosted_orogenic"
        return "orogenic_general"

    # Carlin — check BEFORE the sediment-hosted Cu block because Carlin gold
    # is routinely described as "sediment-hosted disseminated gold" in NI 43-101
    # reports, and the generic sediment-hosted detector would otherwise win.
    if "carlin" in blob:
        return "carlin_general"

    # Sediment-hosted Cu — includes stratabound/stratiform language and the
    # Central African Copperbelt (Kamoa-Kakula, Tenke-Fungurume) style which
    # is rarely tagged with the "sediment-hosted" keyword in technical reports.
    # We've already returned for Carlin above, so plain "sediment-hosted" here
    # is taken to mean sediment-hosted COPPER.
    if any(k in blob for k in ("sedex", "kupferschiefer", "manto", "crd", "mvt",
                                "sediment hosted", "sediment-hosted", "carbonate replacement",
                                "redbed copper", "stratabound", "stratiform",
                                "central african copperbelt", "copperbelt", "katanga",
                                "lualaba")):
        if "sedex" in blob:
            return "sedex"
        if "kupferschiefer" in blob:
            return "kupferschiefer_style"
        if "manto" in blob:
            return "manto_cu"
        if "crd" in blob or "carbonate replacement" in blob:
            return "crd"
        if "mvt" in blob:
            return "mvt"
        if "redbed" in blob:
            return "redbed_cu"
        if any(k in blob for k in ("copperbelt", "katanga", "lualaba")):
            return "kupferschiefer_style"  # CACB is the closest analog family
        return "sediment_hosted_general"

    # VMS (after porphyry/iocg/sediment because those are more specific)
    if any(k in blob for k in ("vms", "vhms", "volcanic hosted massive sulphide",
                                "volcanic-hosted massive sulphide",
                                "volcanogenic massive sulphide", "volcanogenic massive sulfide")):
        return "vms_general"

    # Skarn
    if "skarn" in blob:
        if "cu" in blob and "au" in blob:
            return "cu_au_skarn"
        if "fe" in blob or "iron" in blob:
            return "fe_skarn"
        if "zn" in blob or "pb" in blob:
            return "zn_pb_skarn"
        if "w " in blob or "mo " in blob or "tungsten" in blob or "molybdenum" in blob:
            return "w_mo_skarn"
        return "skarn_general"

    # PGM reef
    if "merensky" in blob:
        return "merensky_reef"
    if "ug2" in blob or "ug-2" in blob:
        return "ug2_reef"
    if "platreef" in blob:
        return "platreef"

    # Laterite Ni
    if "laterite" in blob and ("ni" in blob or "nickel" in blob):
        if "limonite" in blob:
            return "limonite_laterite"
        if "saprolite" in blob:
            return "saprolite_laterite"
        return "laterite_general"

    # Magmatic sulphide Ni-Cu-PGE
    has_magmatic_keyword = (
        "magmatic sulphide" in blob or "magmatic sulfide" in blob
        or "komatiite" in blob
        or ("conduit" in blob and ("ni" in blob or "nickel" in blob))
        or (("ni-cu" in blob or "ni cu" in blob) and ("sulphide" in blob or "sulfide" in blob))
    )
    if has_magmatic_keyword:
        if "komatiite" in blob:
            return "komatiite_hosted"
        if "conduit" in blob:
            return "conduit_hosted"
        return "magmatic_sulphide_general"

    # BIF
    if "bif" in blob or "banded iron" in blob or "magnetite taconite" in blob:
        return "bif_general"

    return None


def detect_mode(
    processing_method: Optional[str] = None,
    mineralization_style: Optional[str] = None,
    description: Optional[str] = None,
    deposit_type: Optional[str] = None,
) -> Optional[str]:
    """Detect primary mineralization mode. Returns slug or None."""
    blob = " ".join(filter(None, [
        _norm(processing_method), _norm(mineralization_style),
        _norm(description), _norm(deposit_type),
    ]))
    if not blob:
        return None

    has_oxide = any(k in blob for k in ("oxide", "supergene", "chrysocolla", "malachite",
                                          "azurite", "atacamite", "brochantite", "limonite",
                                          "leached", "heap leach", "heap-leach", "sx-ew", "sx_ew",
                                          "iscr", "in-situ copper recovery"))
    has_sulfide = any(k in blob for k in ("sulfide", "sulphide", "chalcopyrite", "bornite",
                                            "pyrite", "pentlandite", "pyrrhotite", "galena",
                                            "sphalerite", "flotation"))
    has_refractory = "refractory" in blob

    if has_refractory:
        return "refractory_sulfide"
    if has_oxide and has_sulfide:
        return "mixed_oxide_sulfide"
    if has_oxide:
        return "supergene_oxide"
    if has_sulfide:
        return "primary_sulfide"
    if "placer" in blob:
        return "placer"
    if "free milling" in blob or "free-milling" in blob:
        return "free_milling_oxide"
    return None


def detect_belt(
    country: Optional[str] = None,
    region: Optional[str] = None,
    district: Optional[str] = None,
) -> Optional[str]:
    """
    Map country + region/district to a tectonic belt slug.

    Strategy: score every belt by the number of its region keywords that hit
    region+district text, with a +1 for matching country. Return the belt
    with the highest score. This handles overlapping coverage (e.g. Utah is
    listed in both laramide_southwest as a region AND great_basin_carlin via
    'oquirrh' — when the project mentions Oquirrh Formation explicitly,
    great_basin wins).
    """
    c = _norm(country)
    region_blob = " ".join(filter(None, [_norm(region), _norm(district)]))

    if not c and not region_blob:
        return None

    best_belt: Optional[str] = None
    best_score: int = 0
    for belt, criteria in TECTONIC_BELTS.items():
        score = 0
        # Count region keyword matches in the region_blob
        for r in criteria.get("regions", []):
            if r and r in region_blob:
                score += 1
        # +1 if country matches (or no country constraint on the belt)
        belt_countries = criteria.get("countries", [])
        if score > 0:
            if not belt_countries or c in belt_countries:
                score += 1
            else:
                # Region hit but country doesn't match — disqualify this belt
                score = 0
        # Strictly higher score wins; on ties the first-iterated belt keeps it
        if score > best_score:
            best_score = score
            best_belt = belt

    if best_belt is not None:
        return best_belt

    # No region hit — fall back to country-only when the country uniquely
    # maps to a single belt (Finland → fennoscandian; New Caledonia → its
    # laterite belt). Multi-belt countries (Canada, USA) skip this fallback.
    if c:
        country_belts = [
            belt for belt, criteria in TECTONIC_BELTS.items()
            if c in criteria.get("countries", [])
        ]
        if len(country_belts) == 1:
            return country_belts[0]
    return None


def detect_metal_suite(
    material: Optional[str] = None,
    byproducts: Optional[str] = None,
    description: Optional[str] = None,
    deposit_type: Optional[str] = None,
) -> Optional[str]:
    """Detect characteristic metal suite. Returns slug or None."""
    m = _norm(material)
    blob = " ".join(filter(None, [
        _norm(byproducts), _norm(description), _norm(deposit_type),
    ]))

    # Cu-bearing
    if m == "copper" or "copper" in blob:
        has_au = "au" in blob or "gold" in blob
        has_mo = "mo" in blob or "molybdenum" in blob
        has_co = "co " in blob or "cobalt" in blob
        has_sc = "sc " in blob or "scandium" in blob
        has_ag = "silver" in blob or " ag " in blob
        has_zn = "zinc" in blob or " zn " in blob
        if has_au and (has_co or has_sc):
            return "cu_au_co_sc"
        if has_au:
            return "cu_au"
        if has_mo:
            return "cu_mo"
        if has_ag:
            return "cu_ag"
        if has_zn:
            return "cu_zn_pb"
        return "cu_au"  # most common for porphyry/IOCG

    if m == "gold":
        if "silver" in blob or "ag" in blob:
            return "au_ag"
        return "au_only"

    if m == "silver":
        if any(k in blob for k in ("pb ", "lead", "zn ", "zinc")):
            return "ag_pb_zn"
        return "au_ag"

    if m == "nickel":
        if "cobalt" in blob and "laterite" not in blob:
            return "ni_co"
        if "pge" in blob or "pgm" in blob or "platinum" in blob:
            return "ni_cu_pge"
        if "laterite" in blob:
            return "ni_co"
        return "ni_cu_pge"

    if m in ("pgm", "platinum", "palladium"):
        return "pt_pd_rh"
    if m == "uranium":
        return "u_only"
    if m == "iron":
        return "fe_only"
    if m == "lithium":
        return "li_only"
    if m in ("rare earth", "ree"):
        return "ree_only"

    return None


def detect_alteration_signature(
    alteration: Optional[str] = None,
    description: Optional[str] = None,
    deposit_type: Optional[str] = None,
) -> Optional[str]:
    """Detect alteration assemblage signature. Returns slug or None."""
    blob = " ".join(filter(None, [
        _norm(alteration), _norm(description), _norm(deposit_type),
    ]))
    if not blob:
        return None

    if "potassic" in blob and ("phyllic" in blob or "sericite" in blob):
        return "potassic_phyllic"
    if "potassic" in blob and ("propylitic" in blob or "chlorite-epidote" in blob):
        return "potassic_propylitic"
    if "sodic" in blob and "calcic" in blob:
        return "sodic_calcic"
    if "hematite" in blob or "specularite" in blob:
        return "hematite_specularite"
    if "advanced argillic" in blob or ("argillic" in blob and "alunite" in blob):
        return "argillic_advanced_argillic"
    if "sericite" in blob or ("quartz" in blob and "sericite" in blob):
        return "sericitic_quartz"
    if "skarn" in blob and ("garnet" in blob or "pyroxene" in blob or "calc-silicate" in blob):
        return "skarn_calc_silicate"
    if "decalcified" in blob or "silicification" in blob:
        return "silicification_decalcified"
    if "chlorite" in blob and "carbonate" in blob:
        return "chlorite_carbonate"
    if "supergene" in blob and ("oxidation" in blob or "weathering" in blob):
        return "supergene_oxidation"
    if "laterite" in blob:
        return "lateritic_weathering"
    return None


def detect_recovery_method(
    processing_method: Optional[str] = None,
    description: Optional[str] = None,
    deposit_type: Optional[str] = None,
) -> Optional[str]:
    """Detect primary metallurgical recovery method. Returns slug or None."""
    blob = " ".join(filter(None, [
        _norm(processing_method), _norm(description), _norm(deposit_type),
    ]))
    if not blob:
        return None

    if "iscr" in blob or "in-situ copper recovery" in blob or "in situ recovery" in blob:
        return "iscr"
    if "hpal" in blob or "high pressure acid leach" in blob:
        return "hpal"
    if "atmospheric leach" in blob:
        return "atmospheric_leach"
    if "heap leach" in blob or "heap-leach" in blob:
        return "heap_leach"
    if "sx-ew" in blob or "sx/ew" in blob or "sx_ew" in blob or "solvent extraction" in blob:
        return "sx_ew"
    if "cil" in blob or "cip" in blob or "carbon-in-leach" in blob or "carbon-in-pulp" in blob:
        return "cil_cip"
    if "cyanide" in blob and "leach" in blob:
        return "cn_leach"
    if "smelt" in blob:
        return "smelting"
    if "gravity" in blob:
        return "gravity"
    if "flotation" in blob or "flot " in blob:
        return "flotation"
    return None


def detect_host_class(
    host_rock: Optional[str] = None,
    deposit_type: Optional[str] = None,
    mineralization_style: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[str]:
    """
    Classify the dominant host lithology into one coarse class. Returns one
    of HOST_ROCK_CLASSES or None.
    """
    # Prefer the host_rock field strongly — it's specifically about lithology.
    # mineralization_style and description often mention vein mineralogy
    # ("quartz-carbonate veins") that should NOT be read as host lithology.
    host_blob = _norm(host_rock)
    aux_blob = " ".join(filter(None, [
        _norm(deposit_type), _norm(mineralization_style), _norm(description),
    ]))
    if not host_blob and not aux_blob:
        return None

    def _hit(needles, *blobs):
        for needle in needles:
            for b in blobs:
                if needle in b:
                    return True
        return False

    # Carbonatite (very specific — REE host)
    if _hit(("carbonatite",), host_blob, aux_blob):
        return "carbonatite"

    # BIF (sedimentary, but distinct iron-ore host)
    if _hit(("bif", "banded iron formation", "iron formation",
              "magnetite taconite", "itabira"), host_blob, aux_blob):
        return "bif"

    # Carbonate sediment — only count when it's a clearly sedimentary carbonate.
    # Exclude "quartz-carbonate" (vein mineralogy) and "carbonate replacement"
    # (depositional style, not necessarily carbonate host).
    has_carbonate = _hit(
        ("limestone", "dolostone", "calcarenite", "calcareous", "silty limestone"),
        host_blob, aux_blob,
    )
    if not has_carbonate and "dolomite" in host_blob and "quartz-carbonate" not in host_blob:
        has_carbonate = True
    # bare "carbonate" mention only counts if it's in host_rock alone (not vein gangue context)
    if not has_carbonate and "carbonate" in host_blob and "quartz-carbonate" not in host_blob:
        has_carbonate = True
    if has_carbonate:
        return "carbonate_sediment"

    # Metamorphic high-grade (gneiss / amphibolite / etc.) — checked BEFORE
    # intrusive_felsic because orthogneiss is metamorphic, not igneous.
    # Also before ultramafic so "gneiss with amphibolite and ultramafic units"
    # (White Gold) gets the dominant host (gneiss) rather than the accessory.
    if _hit(("gneiss", "amphibolite", "granulite", "migmatite",
              "high-grade metamorphic"), host_blob, aux_blob):
        return "metamorphic_high_grade"

    # Metamorphic low-grade (greenstone, schist)
    if _hit(("greenstone", "schist", "phyllite", "slate", "metavolcanic",
              "metasediment", "metabasalt", "metaintrusive"), host_blob, aux_blob):
        return "metamorphic_low_grade"

    # Intrusive mafic (gabbro, diorite, monzodiorite) — Doubleview Hat, True North
    # Check before ultramafic because some texts say "ultramafic intrusion" loosely.
    if _hit(("gabbro", "diorite", "monzodiorite", "monzonite",
              "norite", "anorthosite"), host_blob, aux_blob):
        return "intrusive_mafic"

    # Ultramafic (only narrow terms — peridotite, dunite, serpentinite)
    if _hit(("peridotite", "dunite", "serpentinite"), host_blob, aux_blob):
        return "ultramafic"
    # Bare "ultramafic" mention — only if no other host class hit
    if "ultramafic" in host_blob:
        return "ultramafic"

    # Intrusive felsic (granite, syenite)
    if _hit(("granite", "granodiorite", "monzogranite", "leucogranite",
              "syenite", "alaskite", "trachyte porphyry"), host_blob, aux_blob):
        return "intrusive_felsic"

    # Volcanic — felsic vs mafic
    if _hit(("rhyolite", "dacite", "felsic tuff", "felsic volcanic",
              "dacitic", "rhyolitic"), host_blob, aux_blob):
        return "volcanic_felsic"
    if _hit(("basalt", "andesite", "mafic volcanic", "mafic flow",
              "komatiite", "pillow lava"), host_blob, aux_blob):
        return "volcanic_mafic"

    # Clastic sediment (sandstone, siltstone, shale)
    if _hit(("sandstone", "siltstone", "shale", "mudstone",
              "conglomerate", "arkose", "greywacke", "redbed"), host_blob, aux_blob):
        return "clastic_sediment"

    # Last-resort family inference
    dep = _norm(deposit_type)
    if "carlin" in dep:
        return "carbonate_sediment"
    if "porphyry" in dep:
        return "intrusive_felsic"
    if "bif" in dep:
        return "bif"
    if "laterite" in dep:
        return "ultramafic"
    return None


def detect_pattern(
    mineralization_style: Optional[str] = None,
    mining_method: Optional[str] = None,
    processing_method: Optional[str] = None,
    deposit_type: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[str]:
    """
    Detect the orebody geometry pattern. This is what determines mining method
    and resource model shape — vein-hosted vs disseminated-bulk vs replacement
    are mined and modelled completely differently even within the same
    deposit subtype.

    Returns one of MINERALIZATION_PATTERNS or None if no confident match.
    """
    blob = " ".join(filter(None, [
        _norm(mineralization_style), _norm(mining_method),
        _norm(processing_method), _norm(deposit_type),
        _norm(description),
    ]))
    if not blob:
        return None

    # Strong signals first
    if "placer" in blob or "alluvial" in blob:
        return "placer"
    if "reef" in blob and ("merensky" in blob or "ug2" in blob or "ug-2" in blob
                            or "platreef" in blob or "pgm" in blob or "pge" in blob):
        return "reef"
    if ("massive sulphide" in blob or "massive sulfide" in blob
            or "vms" in blob or "vhms" in blob):
        return "massive_sulphide"
    if ("oxide blanket" in blob or "supergene blanket" in blob or "laterite" in blob
            or ("supergene" in blob and "blanket" in blob)):
        return "blanket"

    # Replacement (CRD, manto, Trixie-style polymetallic)
    if ("carbonate replacement" in blob or "crd" in blob
            or "manto" in blob or "replacement" in blob):
        # Distinguish replacement-style from breccia-hosted within sediment-hosted
        if "vein" in blob and "replacement" not in blob:
            return "vein_hosted"
        return "replacement"

    # Vein-hosted — narrow, structurally controlled. Triggers on explicit
    # vein/shear language, free gold + pyrite. Excludes when "stockwork" or
    # "disseminated" dominates.
    has_vein = any(k in blob for k in (
        "vein", "quartz-carbonate vein", "quartz vein", "fault-fill",
        "shear-hosted", "shear zone", "lode gold", "fissure",
    ))
    has_dissem = any(k in blob for k in (
        "disseminated", "bulk tonnage", "bulk low-grade", "bulk low grade",
        "low-grade halo", "halo-dominated", "low-grade bulk",
    ))
    has_stockwork = any(k in blob for k in ("stockwork", "stringer", "veinlet"))
    has_breccia = "breccia" in blob

    # If multiple signals: prefer the dominant style indicated by the words used
    if has_vein and not has_dissem and not has_stockwork:
        return "vein_hosted"
    if has_dissem and not has_vein:
        return "disseminated_bulk"
    if has_stockwork and not has_vein:
        return "stockwork"
    if has_breccia and not has_vein and not has_dissem:
        return "breccia_hosted"
    # Mixed signals: the more aggressive geometry wins for mining-style purposes
    if has_dissem:
        return "disseminated_bulk"
    if has_vein:
        return "vein_hosted"
    if has_stockwork:
        return "stockwork"
    if has_breccia:
        return "breccia_hosted"

    # Last-resort by deposit family inference
    dep = _norm(deposit_type)
    if "porphyry" in dep:
        return "stockwork"
    if "orogenic" in dep or "lode" in dep:
        return "vein_hosted"
    if "carlin" in dep or "sediment hosted" in dep:
        return "disseminated_bulk"
    if "iocg" in dep:
        return "breccia_hosted"
    if "epithermal" in dep:
        return "vein_hosted"
    return None


def detect_stage_class(
    project_stage: Optional[str] = None,
    has_mre: Optional[bool] = None,
    description: Optional[str] = None,
) -> Optional[str]:
    """Map the freeform project_stage string onto a PROJECT_STAGES slug."""
    s = _norm(project_stage)
    blob = " ".join(filter(None, [s, _norm(description)]))
    if not blob:
        return None
    if any(k in blob for k in ("production", "operating", "in operation", "producing mine")):
        return "production"
    if any(k in blob for k in ("construction", "under construction")):
        return "construction"
    # Check pre-feasibility first — "feasibility study" appears in "pre-feasibility study"
    if any(k in blob for k in ("pre-feasibility", "prefeasibility", " pfs", "pre feasibility")):
        return "pfs"
    if any(k in blob for k in ("feasibility study", "bankable", "definitive feasibility",
                                  " bfs", " dfs", " fs ")):
        return "feasibility"
    if any(k in blob for k in ("pea", "preliminary economic assessment", "scoping study",
                                 "preliminary assessment", "economic assessment")):
        return "pea"
    if any(k in blob for k in ("care and maintenance", "care & maintenance",
                                 "suspended", "on care")):
        return "care_maintenance"
    if any(k in blob for k in ("closed", "rehabilitated", "decommissioned")):
        return "closed"
    if any(k in blob for k in ("measured and indicated", "m+i", "m&i", "indicated resource",
                                 "measured resource", "ni 43-101 resource",
                                 "compliant resource")):
        return "resource_m_and_i"
    if any(k in blob for k in ("inferred resource", "initial resource", "maiden resource")):
        return "resource_inferred"
    if any(k in blob for k in ("exploration", "grassroots", "drilling", "advanced exploration",
                                 "early exploration", "target")):
        return "exploration"
    return None


def detect_mining_method_class(
    mining_method: Optional[str] = None,
    processing_method: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[str]:
    """Map freeform mining-method strings to a MINING_METHOD_CLASSES slug."""
    blob = " ".join(filter(None, [
        _norm(mining_method), _norm(processing_method), _norm(description),
    ]))
    if not blob:
        return None

    if "iscr" in blob or "in-situ copper recovery" in blob or "in situ recovery" in blob:
        return "iscr_in_situ"
    if "in-situ leach" in blob or "in situ leach" in blob or "isl" in blob:
        return "iscr_in_situ"
    if "block cave" in blob or "block-cave" in blob or "panel cave" in blob:
        return "block_cave"
    if "dredging" in blob or "dredge" in blob or "alluvial" in blob:
        return "dredging"
    if "heap leach" in blob or "heap-leach" in blob:
        return "heap_leach_pad"
    if "solution mining" in blob or "brine" in blob:
        return "solution_mining"
    if "highwall" in blob:
        return "highwall"
    has_ug = any(k in blob for k in ("underground", "ug ", " ug,", "shaft", "decline",
                                        "sublevel", "stoping", "cut and fill", "longhole"))
    has_op = any(k in blob for k in ("open pit", "open-pit", "open cut", "open-cut",
                                        "strip mine"))
    has_vein_signal = any(k in blob for k in ("vein", "narrow-vein", "shear-hosted",
                                                  "narrow vein", "lode"))
    has_bulk_signal = any(k in blob for k in ("bulk", "low-grade", "low grade",
                                                  "high tonnage", "stockwork", "disseminated"))

    if has_ug and has_vein_signal:
        return "underground_vein"
    if has_ug and has_bulk_signal:
        return "underground_bulk"
    if has_ug:
        # default UG to vein when no scale signal — narrow-vein is the most
        # common UG mining mode globally
        return "underground_vein"
    if has_op and has_bulk_signal:
        return "open_pit_bulk"
    if has_op:
        return "open_pit_selective"
    return None


def detect_resource_category_class(
    resource_category: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[str]:
    """Map freeform resource_category strings to a RESOURCE_CATEGORY_CLASSES slug."""
    blob = " ".join(filter(None, [_norm(resource_category), _norm(description)]))
    if not blob:
        return None
    has_meas = "measured" in blob
    has_ind  = "indicated" in blob
    has_inf  = "inferred" in blob
    has_prov = "proven" in blob or "proved" in blob
    has_prob = "probable" in blob
    if has_prov and has_prob:
        return "reserve_p_and_p"
    if has_prov:
        return "reserve_proven"
    if has_prob:
        return "reserve_probable"
    if has_meas and has_ind and has_inf:
        return "m_and_i_plus_inferred"
    if has_meas and has_ind:
        return "m_and_i"
    if has_meas:
        return "measured"
    if has_ind:
        return "indicated"
    if has_inf:
        return "inferred"
    if "exploration target" in blob:
        return "exploration_target"
    if "historical" in blob or "non-compliant" in blob:
        return "historical"
    return None


def detect_resource_compliance(
    resource_category: Optional[str] = None,
    description: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Optional[str]:
    """Map text mentions of NI 43-101 / JORC / etc. to a compliance slug."""
    blob = " ".join(filter(None, [
        _norm(resource_category), _norm(description), _norm(source_url),
    ]))
    if not blob:
        return None
    if "ni 43-101" in blob or "ni43-101" in blob or "ni-43-101" in blob or "43-101" in blob:
        return "ni_43_101"
    if "jorc" in blob:
        return "jorc"
    if "sk-1300" in blob or "sk 1300" in blob or "s-k 1300" in blob or "sk1300" in blob:
        return "sk_1300"
    if "samrec" in blob:
        return "samrec"
    if "historical" in blob or "non-compliant" in blob or "non compliant" in blob:
        return "historical"
    if "press release" in blob or "company announcement" in blob:
        return "press_release"
    if "internal" in blob and "resource" in blob:
        return "internal"
    return None


# ── Compatibility checks ─────────────────────────────────────────────────────

def recovery_compatible(target: Optional[str], candidate: Optional[str]) -> bool:
    """
    True when target and candidate recovery methods are compatible (or either is unknown).
    Used as a hard filter — incompatible recovery means the analog's metallurgy doesn't
    transfer to the target's processing route.
    """
    if not target or not candidate:
        return True  # unknown — let through (analog finder defaults to permissive)
    if target == candidate:
        return True
    incompatible = RECOVERY_INCOMPATIBILITY.get(target, frozenset())
    return candidate not in incompatible


def subtype_compatible(target: Optional[str], candidate: Optional[str]) -> bool:
    """
    True when target and candidate sub-types are compatible (same sub-type, or
    either is unknown). Sub-types within the same family but different slugs are
    treated as INCOMPATIBLE — that's the whole point of sub-type filtering.

    Example: 'alkalic_porphyry' vs 'laramide_porphyry' → False
              'alkalic_porphyry' vs None → True (unknown candidate, pass through)
    """
    if not target or not candidate:
        return True
    return target == candidate


def mode_compatible(target: Optional[str], candidate: Optional[str]) -> bool:
    """
    True when mineralization modes are compatible. Mixed mode is compatible with
    either pure sulfide or pure oxide (transitional zones often appear in both).
    Earlier code wrote `"mixed" in (target, candidate)` which is *tuple
    membership* and never matches the compound slug `mixed_oxide_sulfide` —
    substring check is what we actually want.
    """
    if not target or not candidate:
        return True
    if target == candidate:
        return True
    if "mixed" in target or "mixed" in candidate:
        return True
    return False


def stage_compatible(target: Optional[str], candidate: Optional[str]) -> bool:
    """
    True when project stage classes are compatible per STAGE_COMPATIBILITY.
    Unknown stages on either side are permissive (the cascade won't strict-fail
    on missing metadata; the rule's required_stages list does that).
    """
    if not target or not candidate:
        return True
    if target == candidate:
        return True
    return candidate in STAGE_COMPATIBILITY.get(target, frozenset())


def mining_method_compatible(target: Optional[str], candidate: Optional[str]) -> bool:
    """True when mining methods can substitute for each other in resource modelling."""
    if not target or not candidate:
        return True
    if target == candidate:
        return True
    incompatible = MINING_METHOD_INCOMPATIBILITY.get(target, frozenset())
    return candidate not in incompatible


def resource_category_at_least(
    candidate: Optional[str], minimum: Optional[str],
) -> bool:
    """
    True when the candidate's resource category meets or exceeds the rule's
    minimum (using RESOURCE_CATEGORY_RANK). Unknowns are permissive — the
    rule's structured required_categories list does strict gating.
    """
    if not minimum or not candidate:
        return True
    c_rank = RESOURCE_CATEGORY_RANK.get(candidate, 0)
    m_rank = RESOURCE_CATEGORY_RANK.get(minimum, 0)
    return c_rank >= m_rank


def compliance_acceptable(
    candidate: Optional[str], min_year: Optional[int] = None,
    vintage_year: Optional[int] = None,
) -> bool:
    """
    True when a candidate's resource compliance standard is acceptable for
    modelling. Historical / press_release / internal are non-compliant — drop
    unconditionally. Compliant standards (NI 43-101, JORC, SK-1300, SAMREC,
    PERA) must also satisfy `min_year` if both are provided.
    """
    if candidate is None:
        return True  # unknown — let through, other gates handle it
    if candidate in {"historical", "press_release", "internal"}:
        return False
    if min_year is not None and vintage_year is not None:
        return vintage_year >= min_year
    return True
