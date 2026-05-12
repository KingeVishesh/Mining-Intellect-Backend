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

# Deposit sub-types — finer-grained than family. Keys are the family, values are
# the allowed sub-types within that family.
DEPOSIT_SUBTYPES: Dict[str, List[str]] = {
    "porphyry":          ["alkalic", "calc_alkalic", "laramide", "high_sulfidation_lithocap"],
    "iocg":              ["oxide", "sulfide", "hybrid"],
    "epithermal":        ["low_sulfidation", "high_sulfidation", "intermediate_sulfidation"],
    "orogenic":          ["greenstone", "turbidite_hosted", "bif_hosted"],
    "vms":               ["bimodal_mafic", "bimodal_felsic", "mafic_ultramafic", "siliciclastic_felsic"],
    "skarn":             ["cu_au_skarn", "fe_skarn", "zn_pb_skarn", "w_mo_skarn"],
    "sediment_hosted":   ["sedex", "kupferschiefer_style", "manto", "crd", "mvt", "redbed_cu"],
    "carlin":            ["jasperoid", "decalcified", "sedimentary_breccia"],
    "magmatic_sulphide": ["komatiite", "intrusion_hosted", "conduit_hosted"],
    "laterite":          ["limonite", "saprolite", "smectite"],
    "bif":               ["high_grade_hematite", "magnetite_taconite", "channel_iron"],
    "unconformity":      ["egress", "ingress"],
    "rollfront":         ["sandstone_hosted"],
    "pgm_reef":          ["merensky", "ug2", "platreef"],
    "oxide_iscr":        ["supergene_blanket"],  # in-situ copper recovery — distinct from family
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
        "regions": ["nevada", "carlin", "battle mountain", "eureka"],
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
        "countries": ["zambia", "drc", "congo", "democratic republic of the congo"],
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
        "regions": ["fennoscandian", "baltic shield"],
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

    # Sediment-hosted Cu — includes stratabound/stratiform language and the
    # Central African Copperbelt (Kamoa-Kakula, Tenke-Fungurume) style which
    # is rarely tagged with the "sediment-hosted" keyword in technical reports.
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

    # Carlin
    if "carlin" in blob:
        return "carlin_general"

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
    """Map country + region/district to a tectonic belt slug. Returns None if no match."""
    c = _norm(country)
    region_blob = " ".join(filter(None, [_norm(region), _norm(district)]))

    # First, look for an explicit region match across all belts (more specific than country)
    for belt, criteria in TECTONIC_BELTS.items():
        for r in criteria.get("regions", []):
            if r and r in region_blob:
                # Validate country if specified for this belt
                belt_countries = criteria.get("countries", [])
                if not belt_countries or c in belt_countries:
                    return belt

    # No region hit — look at country-only belts (single-country belts only)
    if c:
        for belt, criteria in TECTONIC_BELTS.items():
            belt_countries = criteria.get("countries", [])
            # Only attribute by country alone if the belt has a single country
            # (avoids mis-attributing all-Canada projects to Quesnel/Stikine)
            if len(belt_countries) == 1 and c in belt_countries and not criteria.get("regions"):
                return belt
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
    """
    if not target or not candidate:
        return True
    if target == candidate:
        return True
    if "mixed" in (target, candidate):
        return True
    return False
