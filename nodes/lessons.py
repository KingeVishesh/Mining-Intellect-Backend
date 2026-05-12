"""
Lessons Learned — first-class data, single source of truth.

Each lesson is an identifier referenced by analog_selection rules in
`scripts/seed_analog_rules.py` (the `applies_lessons` field) and surfaced
verbatim in:
  * `_cascading_match` reason strings → LangSmith traces
  * `report_analogs.lessons` JSON column → frontend "Why was this picked?" UI
  * the Pydantic schema validator → ensures rules can only cite known lessons

Add a new lesson by appending an entry here. The Pydantic schema's
`@field_validator("applies_lessons")` will then accept its ID.

Each entry has:
  commodity   — which commodity's lessons doc it comes from
  title       — short headline
  text        — the exact lesson principle in plain language
  source_doc  — citation for traceability
"""
from __future__ import annotations
from typing import Dict, TypedDict


class Lesson(TypedDict):
    commodity: str
    title: str
    text: str
    source_doc: str


LESSONS: Dict[str, Lesson] = {
    # ── COPPER (numeric IDs from the original Copper Lessons Learned doc) ──
    "L36": {
        "commodity": "copper",
        "title": "Unbiased pre-MRE analog selection",
        "text": (
            "Pre-MRE independent resource modelling requires analog selection "
            "free of confirmation bias toward the project's own narrative. "
            "Use only analogs that share primary geological identity — not "
            "merely the same commodity."
        ),
        "source_doc": "Copper Lessons Learned, L36",
    },
    "L83": {
        "commodity": "copper",
        "title": "Oxide ISCR is a distinct modelling regime",
        "text": (
            "Supergene oxide blankets processed via in-situ copper recovery "
            "(ISCR) have a fundamentally different resource model, cut-off, "
            "and recovery curve from primary sulfide porphyries. They must "
            "be analogged separately."
        ),
        "source_doc": "Copper Lessons Learned, L83",
    },
    "L86": {
        "commodity": "copper",
        "title": "≥95% geological similarity threshold",
        "text": (
            "Analog selection requires ≥95% match across primary "
            "mineralization style, deposit type, tectonic terrane, "
            "alteration assemblage, metal suite, and recovery route. "
            "Family-level matches (e.g. all 'porphyry') are insufficient."
        ),
        "source_doc": "Copper Lessons Learned, L86",
    },
    "L92": {
        "commodity": "copper",
        "title": "Oxide vs sulfide recovery sensitivity",
        "text": (
            "Recovery sensitivity tables for oxide ISCR (40–75%) cannot be "
            "applied to sulfide flotation projects (85–93%) and vice versa. "
            "Match recovery method when modelling cut-off grade sensitivity."
        ),
        "source_doc": "Copper Lessons Learned, L92",
    },
    "L101": {
        "commodity": "copper",
        "title": "Primary mineralization style must match",
        "text": (
            "Primary sulfide chalcopyrite-bornite mineralization and supergene "
            "oxide chrysocolla-malachite mineralization are not interchangeable "
            "analogs even within the same deposit-type family."
        ),
        "source_doc": "Copper Lessons Learned, L101",
    },
    "L102": {
        "commodity": "copper",
        "title": "Alteration zonation match",
        "text": (
            "Analog deposits must share the alteration zonation pattern "
            "(potassic core / phyllic shell / propylitic halo for alkalic "
            "porphyry; Na-Ca + hematite for IOCG; advanced argillic for HS "
            "epithermal). Different alteration ⇒ different deposit model."
        ),
        "source_doc": "Copper Lessons Learned, L102",
    },
    "L118": {
        "commodity": "copper",
        "title": "ISCR domains and oxide processing",
        "text": (
            "ISCR domains use distinct block-model parameters from underlying "
            "sulfide root zones. Do not blend oxide and sulfide resource "
            "estimates from analogs that mine both."
        ),
        "source_doc": "Copper Lessons Learned, L118",
    },
    "L124": {
        "commodity": "copper",
        "title": "Sub-type specificity (alkalic vs Laramide vs IOCG)",
        "text": (
            "Alkalic Cu-Au porphyries (BC Quesnel/Stikine, Lachlan) have "
            "different metal ratios, alteration, and host intrusives from "
            "calc-alkalic Laramide porphyries (Arizona/Sonora/Chile). They "
            "are not interchangeable analogs."
        ),
        "source_doc": "Copper Lessons Learned, L124",
    },
    "L138": {
        "commodity": "copper",
        "title": "Recovery and metallurgy must transfer",
        "text": (
            "Metallurgical recovery assumptions can only be borrowed from "
            "analogs that share the same processing route (flotation, ISCR, "
            "SX-EW, heap-leach). Cross-route analog borrowing introduces "
            "systematic resource estimation error."
        ),
        "source_doc": "Copper Lessons Learned, L138",
    },

    # ── GOLD ───────────────────────────────────────────────────────────────
    "L_ORO_01": {
        "commodity": "gold",
        "title": "Orogenic gold requires craton/greenstone setting",
        "text": (
            "Orogenic / lode gold deposits require the same craton or "
            "greenstone-belt setting as the analog (Abitibi, Yilgarn, "
            "Birimian, Tanzanian Archean). Cross-craton analogs introduce "
            "structural and metamorphic-grade mismatch."
        ),
        "source_doc": "Gold Lessons Learned, Orogenic 01",
    },
    "L_ORO_02": {
        "commodity": "gold",
        "title": "Exclude porphyry, epithermal, Carlin for orogenic targets",
        "text": (
            "Orogenic gold modelling cannot borrow analogs from porphyry, "
            "epithermal (LS/HS), or Carlin systems — vein geometry, "
            "continuity, and recovery curves differ fundamentally."
        ),
        "source_doc": "Gold Lessons Learned, Orogenic 02",
    },
    "L_EPI_01": {
        "commodity": "gold",
        "title": "Low-sulfidation epithermal — adularia-sericite signature",
        "text": (
            "Low-sulfidation epithermal Au-Ag analogs must share adularia-"
            "sericite alteration and boiling-zone depth. Au:Ag ratios cluster "
            "1:5 to 1:50; deviations indicate a different deposit class."
        ),
        "source_doc": "Gold Lessons Learned, Epithermal 01",
    },
    "L_EPI_02": {
        "commodity": "gold",
        "title": "LS vs HS epithermal must not cross",
        "text": (
            "Low-sulfidation and high-sulfidation epithermal deposits form "
            "from different fluid chemistries, leave different cap rocks, "
            "and have different vein geometries. They are not interchangeable "
            "analogs."
        ),
        "source_doc": "Gold Lessons Learned, Epithermal 02",
    },
    "L_EPI_03": {
        "commodity": "gold",
        "title": "High-sulfidation — advanced argillic (alunite-dickite)",
        "text": (
            "HS epithermal analogs must share advanced argillic alteration "
            "(alunite, pyrophyllite, dickite, enargite) and a near-surface "
            "oxidation/enrichment profile. Cu/As/Sb pathfinder ratios must "
            "match the target."
        ),
        "source_doc": "Gold Lessons Learned, Epithermal 03",
    },
    "L_CAR_01": {
        "commodity": "gold",
        "title": "Carlin — Great Basin extensional setting",
        "text": (
            "Carlin-type disseminated gold requires Nevada Basin-and-Range "
            "extensional tectonics or an analogous decalcified-carbonate "
            "setting. Refractory sulfide mineralogy mandates autoclave or "
            "roaster metallurgy."
        ),
        "source_doc": "Gold Lessons Learned, Carlin 01",
    },
    "L_CAR_02": {
        "commodity": "gold",
        "title": "Carlin — exclude vein-hosted analogs",
        "text": (
            "Carlin-style disseminated mineralization cannot be modelled with "
            "vein-hosted gold analogs (orogenic, epithermal). Structural "
            "controls and grade continuity differ fundamentally."
        ),
        "source_doc": "Gold Lessons Learned, Carlin 02",
    },
    "L_HL_01": {
        "commodity": "gold",
        "title": "Heap-leach gold — oxide ore + arid climate",
        "text": (
            "Heap-leach gold analogs require oxide-dominant mineralization "
            "(>70% leach recovery) and an arid climate. Refractory sulfide "
            "or wet-climate projects produce incompatible economics."
        ),
        "source_doc": "Gold Lessons Learned, Heap-Leach 01",
    },
    "L_AU_POR_01": {
        "commodity": "gold",
        "title": "Porphyry Au-Cu — alteration + Cu-Au-Mo ratios",
        "text": (
            "Porphyry gold-copper analogs must match alteration zonation "
            "(potassic core, phyllic shell, argillic cap) and the project's "
            "Cu:Au:Mo ratio (Cu-Au porphyry vs Cu-Mo porphyry are distinct)."
        ),
        "source_doc": "Gold Lessons Learned, Porphyry 01",
    },

    # ── SILVER ─────────────────────────────────────────────────────────────
    "L_CRD_01": {
        "commodity": "silver",
        "title": "CRD silver — carbonate replacement, polymetallic",
        "text": (
            "Carbonate Replacement Deposit (CRD) silver analogs require "
            "limestone/dolomite host at intrusive contact, polymetallic "
            "Pb-Zn-Ag ratios, and the same replacement geometry (mantle / "
            "pipe / blanket). Exclude epithermal vein analogs."
        ),
        "source_doc": "Silver Lessons Learned, CRD 01",
    },
    "L_MAN_01": {
        "commodity": "silver",
        "title": "Manto silver — stratiform sediment-hosted",
        "text": (
            "Manto silver requires stratiform mineralization in a sedimentary "
            "or volcanic host sequence. Match Pb/Zn/Cu by-product ratios and "
            "basin type (back-arc vs fore-arc)."
        ),
        "source_doc": "Silver Lessons Learned, Manto 01",
    },

    # ── URANIUM ────────────────────────────────────────────────────────────
    "L_U_RF_01": {
        "commodity": "uranium",
        "title": "Roll-front uranium — ISR mandatory",
        "text": (
            "Roll-front uranium analogs require permeable sandstone host, "
            "shallow depth (<300m), and in-situ recovery (ISR) metallurgy. "
            "Underground or open-pit U analogs have incompatible economics."
        ),
        "source_doc": "Uranium Lessons Learned, Roll-Front 01",
    },
    "L_U_UN_01": {
        "commodity": "uranium",
        "title": "Unconformity uranium — Athabasca-style high grade",
        "text": (
            "Unconformity-related uranium (Athabasca, McArthur River, Cigar "
            "Lake) requires Proterozoic unconformity setting, high grade "
            "(1–15% U3O8), and underground mining. Grade and tonnage scales "
            "are incompatible with roll-front or intrusive U."
        ),
        "source_doc": "Uranium Lessons Learned, Unconformity 01",
    },
    "L_U_INT_01": {
        "commodity": "uranium",
        "title": "Intrusive-related uranium — Rössing-style",
        "text": (
            "Alaskite/leucogranite-hosted uranium (Rössing, Husab) requires "
            "Proterozoic mobile-belt setting and bulk-tonnage open-pit "
            "geometry at 0.02–0.05% U3O8. Sandstone-hosted and unconformity "
            "U are not valid analogs."
        ),
        "source_doc": "Uranium Lessons Learned, Intrusive 01",
    },

    # ── NICKEL ─────────────────────────────────────────────────────────────
    "L_NI_SUL_01": {
        "commodity": "nickel",
        "title": "Magmatic Ni-Cu-PGE sulphide — flotation regime",
        "text": (
            "Magmatic Ni-Cu-PGE sulphide deposits (Sudbury, Voisey's Bay, "
            "Thompson, Kambalda) require mafic/ultramafic intrusion hosts, "
            "primary sulphide ore, and flotation recovery. Laterite Ni "
            "analogs use HPAL/atmospheric leach and are incompatible."
        ),
        "source_doc": "Nickel Lessons Learned, Sulphide 01",
    },
    "L_NI_LAT_01": {
        "commodity": "nickel",
        "title": "Nickel laterite — HPAL/RKEF metallurgy",
        "text": (
            "Nickel laterite (limonite vs saprolite zones) requires tropical "
            "weathering profile and HPAL or RKEF metallurgy. Magmatic "
            "sulphide analogs cannot transfer — wrong mineralogy, wrong "
            "processing route."
        ),
        "source_doc": "Nickel Lessons Learned, Laterite 01",
    },

    # ── PGM ────────────────────────────────────────────────────────────────
    "L_PGM_MR_01": {
        "commodity": "pgm",
        "title": "Merensky Reef — Pt-dominant ratios",
        "text": (
            "Merensky Reef PGM analogs require Bushveld Complex (or direct "
            "equivalent) setting and Pt-dominant ratios (~Pt:Pd:Rh:Au "
            "60:30:5:5). Pothole losses (15–50%) must match analog frequency."
        ),
        "source_doc": "PGM Lessons Learned, Merensky 01",
    },
    "L_PGM_UG2_01": {
        "commodity": "pgm",
        "title": "UG2 chromitite — Pd-dominant + chromite credit",
        "text": (
            "UG2 chromitite reef PGM analogs have Pd-dominant ratios (~Pd:Pt "
            "55:35) and chromite by-product credit. Thinner reef (0.3–1.0m) "
            "requires mechanised narrow-reef mining."
        ),
        "source_doc": "PGM Lessons Learned, UG2 01",
    },
    "L_PGM_PR_01": {
        "commodity": "pgm",
        "title": "Platreef — bulk-tonnage northern Bushveld",
        "text": (
            "Platreef analogs require thick (5–30m) PGM reef with elevated "
            "Ni-Cu base metal credit. Mining method is large-scale trackless "
            "or open-pit, distinct from Merensky/UG2 narrow-reef."
        ),
        "source_doc": "PGM Lessons Learned, Platreef 01",
    },

    # ── IRON ───────────────────────────────────────────────────────────────
    "L_FE_BIF_01": {
        "commodity": "iron",
        "title": "BIF hematite — craton-specific matching",
        "text": (
            "BIF-hosted hematite iron ore analogs must share craton (Pilbara, "
            "Carajás, Hamersley, Yilgarn). Goethite content >20% triggers "
            "M&I tonnage reduction. DSO vs beneficiation pathways are "
            "distinct."
        ),
        "source_doc": "Iron Lessons Learned, BIF 01",
    },
    "L_FE_MAG_01": {
        "commodity": "iron",
        "title": "Magnetite — beneficiation required to 65%+ Fe",
        "text": (
            "Magnetite iron analogs require crushing and magnetic separation "
            "to produce 65–70% Fe concentrate. Davis Tube Recovery (DTR) and "
            "mass-recovery curves must match. DSO hematite analogs have "
            "incompatible processing economics."
        ),
        "source_doc": "Iron Lessons Learned, Magnetite 01",
    },

    # ── COPPER (rule-specific lessons beyond the numeric ones above) ───────
    "L_VMS_01": {
        "commodity": "copper",
        "title": "VMS — volcanic-belt + seafloor-spreading setting",
        "text": (
            "VMS Cu analogs require the same volcanic-belt and seafloor-"
            "spreading tectonic setting (Iberian Pyrite, Abitibi, "
            "Fennoscandian). Host volcanic composition (mafic / bimodal / "
            "felsic) must match."
        ),
        "source_doc": "Copper Lessons Learned, VMS 01",
    },
    "L_SKN_01": {
        "commodity": "copper",
        "title": "Copper skarn — carbonate-intrusive contact",
        "text": (
            "Cu skarn analogs require carbonate host at intrusive contact, "
            "endoskarn-vs-exoskarn geometry match, and dense drill spacing "
            "(<25 m) due to complex geometry."
        ),
        "source_doc": "Copper Lessons Learned, Skarn 01",
    },
    "L_SED_01": {
        "commodity": "copper",
        "title": "Sediment-hosted Cu — Kupferschiefer vs Copperbelt",
        "text": (
            "Sediment-hosted Cu analogs must distinguish Kupferschiefer-style "
            "(European Permian) from Central African Copperbelt redbed style. "
            "Co/Ni ratios and stratigraphy thickness must match."
        ),
        "source_doc": "Copper Lessons Learned, Sediment-Hosted 01",
    },
}


def get_lesson(lesson_id: str) -> Lesson:
    """Return the full lesson dict for a given ID, or raise KeyError."""
    return LESSONS[lesson_id]


def resolve_lesson_ids(ids: list[str]) -> list[dict]:
    """Resolve a list of lesson IDs to their full dicts, dropping unknowns."""
    return [
        {"id": lesson_id, **LESSONS[lesson_id]}
        for lesson_id in ids
        if lesson_id in LESSONS
    ]
