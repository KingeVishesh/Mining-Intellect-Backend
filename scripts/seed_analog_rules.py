"""
seed_analog_rules.py — Seed analog_selection, model_adjustment, and confidence_adjustment
rules for all commodities based on Lessons Learned documents.

Creates new rules in the compiled_rules table. Safe to re-run (upserts on rule_id).

Commodities: gold, silver, gold_silver, copper, nickel, uranium, pgm, iron

Usage:
    python scripts/seed_analog_rules.py
    python scripts/seed_analog_rules.py --commodity gold
    python scripts/seed_analog_rules.py --dry-run
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Analog Selection Rules ─────────────────────────────────────────────────────
# Each entry: (rule_id, source_material, deposit_type, grade_min, grade_max, grade_unit,
#              tonnage_min_mt, tonnage_max_mt, drilling_stage, title, description,
#              analog_criteria, tonnage_multiplier, grade_multiplier)

ANALOG_SELECTION_RULES = [

    # ── GOLD ──────────────────────────────────────────────────────────────────
    {
        "rule_id": "analog_sel_gold_orogenic",
        "source_material": "gold",
        "deposit_type": "orogenic",
        "grade_min": 1.5, "grade_max": 20.0, "grade_unit": "g/t Au",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Orogenic Gold Analog Selection",
        "description": "Select analogs for orogenic (lode) gold deposits hosted in shear zones, typically underground mining with narrow high-grade veins.",
        "analog_criteria": [
            "Same craton or orogenic belt setting (e.g. Yilgarn, Superior, West African)",
            "Similar structural hosting: shear zone, fault-controlled veins",
            "Comparable plunge depth and underground mining method",
            "Match gold grade band: 2-8 g/t Au preferred, >1.5 g/t Au required",
            "Similar continuity ratio (>0.85) along strike",
            "Match resource classification level (M&I vs Inferred)",
            "Exclude epithermal or porphyry analogs — deposit controls are fundamentally different",
            "Prefer analogs with similar width-to-strike ratio",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_gold_epithermal_ls",
        "source_material": "gold",
        "deposit_type": "epithermal-LS",
        "grade_min": 0.5, "grade_max": 10.0, "grade_unit": "g/t Au",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Low-Sulfidation Epithermal Gold Analog Selection",
        "description": "Select analogs for LS epithermal gold — adularia-sericite alteration, bonanza vein potential, volcanic arc setting.",
        "analog_criteria": [
            "Same volcanic arc setting (e.g. Pacific Rim, Caribbean Arc)",
            "Low-sulfidation alteration style (adularia-sericite, silica sinter)",
            "Similar Au:Ag ratio (typically 1:5 to 1:50 for LS epithermal)",
            "Similar vein density and bonanza grade distribution",
            "Drill spacing <50m for M&I; 50-100m for Inferred",
            "Match depth to boiling zone (~200-400m below paleo-water table)",
            "Exclude HS epithermal analogs — different fluid chemistry and cap rock",
            "Match oxidation depth — oxide vs sulfide processing implications",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_gold_epithermal_hs",
        "source_material": "gold",
        "deposit_type": "epithermal-HS",
        "grade_min": 0.3, "grade_max": 8.0, "grade_unit": "g/t Au",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "High-Sulfidation Epithermal Gold Analog Selection",
        "description": "Select analogs for HS epithermal gold — advanced argillic alteration (alunite-dickite), steam-heated blankets.",
        "analog_criteria": [
            "High-sulfidation alteration (alunite, pyrophyllite, dickite, enargite)",
            "Same volcanic belt or porphyry system proximity",
            "Similar oxidation and enrichment profile",
            "Match depth to sulfide zone (typically 0-300m below surface oxidation cap)",
            "Consider Cu/As/Sb as pathfinder ratios for analog validity",
            "Match project stage — HS deposits often have variable oxide cap thickness",
            "Exclude LS epithermal analogs",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_gold_porphyry",
        "source_material": "gold_silver",
        "deposit_type": "porphyry",
        "grade_min": 0.1, "grade_max": 2.0, "grade_unit": "g/t Au",
        "tonnage_min_mt": 50.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Porphyry Gold-Copper Analog Selection",
        "description": "Select analogs for porphyry Au-Cu systems — bulk tonnage, low grade, large footprint.",
        "analog_criteria": [
            "Match Cu-Au-Mo ratios (porphyry type: Cu-Au vs Cu-Mo)",
            "Similar halo-to-core delineation (core: >0.5% Cu; halo: 0.1-0.5% Cu)",
            "Same tectonic setting (continental arc vs oceanic arc)",
            "Match mining method: open pit for most, block cave for very large deep systems",
            "Similar depth to top of mineralization (<500m preferred for OP)",
            "10-40% tonnage in core zone, 60-80% in halo — match analog zone geometry",
            "Drill spacing 50-100m for M&I in porphyry systems",
            "Match alteration zonation (potassic, phyllic, argillic caps)",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_gold_carlin",
        "source_material": "gold",
        "deposit_type": "carlin",
        "grade_min": 0.5, "grade_max": 5.0, "grade_unit": "g/t Au",
        "tonnage_min_mt": 5.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Carlin-Type Gold Analog Selection",
        "description": "Select analogs for Carlin-type disseminated gold — Nevada Basin and Range, carbonate dissolution, invisible gold in arsenian pyrite.",
        "analog_criteria": [
            "Nevada Basin and Range or analogous extensional tectonic setting",
            "Sediment-hosted disseminated gold in decalcified carbonates",
            "Similar host stratigraphy (silty limestone, calcareous siltstone)",
            "Refractory sulfide mineralization (arsenian pyrite) — similar processing route",
            "Match structural complexity (windows, horsts, graben)",
            "Heap leach or CIL processing implied by grade band",
            "Exclude structurally-hosted vein analogs",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_gold_heap_leach",
        "source_material": "gold",
        "deposit_type": "heap-leach",
        "grade_min": 0.12, "grade_max": 0.8, "grade_unit": "g/t Au",
        "tonnage_min_mt": 20.0, "tonnage_max_mt": None,
        "drilling_stage": "dense",
        "title": "Heap Leach Gold Analog Selection",
        "description": "Select analogs for low-grade bulk tonnage open-pit heap leach gold operations.",
        "analog_criteria": [
            "Low-grade bulk tonnage open-pit operation (0.12-0.6 g/t Au)",
            "Oxide-dominant mineralization for heap leach recoveries (>70%)",
            "Similar climate and terrain (aridity affects leach efficiency)",
            "Match stripping ratio and pit shell geometry",
            "Similar recovery rate (typically 60-80% oxide, 40-60% transitional)",
            "Comparable infrastructure requirements (distance to road, power, water)",
            "Match project stage — heap leach analogs should have completed PFS or feasibility",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },

    # ── SILVER ────────────────────────────────────────────────────────────────
    {
        "rule_id": "analog_sel_silver_crd",
        "source_material": "silver",
        "deposit_type": "CRD",
        "grade_min": 50.0, "grade_max": 2000.0, "grade_unit": "g/t Ag",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Carbonate Replacement (CRD) Silver Analog Selection",
        "description": "Select analogs for CRD silver — polymetallic Pb-Zn-Ag in carbonate replacement, Mexico, Peru, Bolivia.",
        "analog_criteria": [
            "Carbonate-hosted replacement style (limestone, dolomite)",
            "Polymetallic Pb-Zn-Ag ratios — match Ag:Pb:Zn proportions",
            "Same geological province (Mexican Silver Belt, Peruvian Andes, Bolivian Tin Belt)",
            "Similar replacement geometry (mantle vs pipe vs blanket CRD)",
            "Match depth and oxidation profile",
            "Processing: flotation for sulfides, leach for oxides — match mineral assemblage",
            "Exclude epithermal silver vein analogs — different geometry and grade distribution",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_silver_manto",
        "source_material": "silver",
        "deposit_type": "manto",
        "grade_min": 30.0, "grade_max": 600.0, "grade_unit": "g/t Ag",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Manto Silver Analog Selection",
        "description": "Select analogs for stratiform manto silver deposits in sediment-hosted settings.",
        "analog_criteria": [
            "Stratiform silver in sedimentary/volcanic host sequence",
            "Similar Pb/Zn/Cu by-product ratios",
            "Match host stratigraphy thickness and dip",
            "Same basin type (back-arc, fore-arc)",
            "Exclude CRD analogs — different controls on geometry",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },

    # ── COPPER ────────────────────────────────────────────────────────────────
    {
        "rule_id": "analog_sel_copper_porphyry",
        "source_material": "copper",
        "deposit_type": "porphyry",
        "grade_min": 0.2, "grade_max": 2.0, "grade_unit": "% Cu",
        "tonnage_min_mt": 50.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Porphyry Copper Analog Selection",
        "description": "Select analogs for porphyry copper-gold-molybdenum systems.",
        "analog_criteria": [
            "Match Cu-Au-Mo ratios (Cu-Mo vs Cu-Au porphyry subtypes)",
            "Halo vs core grade split (core >0.5% Cu; halo 0.1-0.5% Cu)",
            "Same tectonic belt (Andean, Pacific Rim arc)",
            "Similar alteration zonation (potassic core, phyllic shell, propylitic halo)",
            "Match mining method and depth (open pit vs block cave)",
            "Drill spacing 50-100m for M&I; acceptable <200m for Inferred",
            "290+ porphyry analogs available — filter to 3-4 with 95%+ match",
            "ESG weighting: +10-35% for high-restriction jurisdictions (Peru, Chile water risk)",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_copper_vms",
        "source_material": "copper",
        "deposit_type": "VMS",
        "grade_min": 0.5, "grade_max": 6.0, "grade_unit": "% Cu",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "dense",
        "title": "VMS Copper Analog Selection",
        "description": "Select analogs for volcanogenic massive sulfide copper deposits.",
        "analog_criteria": [
            "Same volcanic belt and seafloor spreading setting",
            "Match Cu-Zn-Pb ratios (Cu-rich vs Zn-rich VMS)",
            "Massive vs stringer stockwork zone geometry",
            "Similar host volcanic composition (mafic vs bimodal vs felsic)",
            "Underground mining assumed for high-grade VMS",
            "Drill spacing <25m for M&I in VMS (high grade variability)",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_copper_iocg",
        "source_material": "copper",
        "deposit_type": "IOCG",
        "grade_min": 0.2, "grade_max": 3.0, "grade_unit": "% Cu",
        "tonnage_min_mt": 20.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "IOCG Copper Analog Selection",
        "description": "Select analogs for Iron Oxide Copper-Gold (IOCG) deposits — Olympic Dam style.",
        "analog_criteria": [
            "Fe-oxide association (magnetite/hematite dominant gangue)",
            "Cu-Au-U element assemblage typical of IOCG",
            "Same craton (Gawler, São Francisco, Central Andes)",
            "Match depth — shallow IOCG vs deep basement-hosted",
            "Similar alteration style (Na-Ca, K-Fe)",
            "86-94% match acceptable for IOCG given limited global analog pool",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_copper_skarn",
        "source_material": "copper",
        "deposit_type": "skarn",
        "grade_min": 0.4, "grade_max": 5.0, "grade_unit": "% Cu",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "dense",
        "title": "Copper Skarn Analog Selection",
        "description": "Select analogs for copper skarn deposits at carbonate-intrusive contacts.",
        "analog_criteria": [
            "Carbonate host (limestone, dolomite) at intrusive contact",
            "Cu-Au skarn — match endoskarn vs exoskarn geometry",
            "Similar intrusive composition and emplacement depth",
            "Prograde vs retrograde alteration assemblage",
            "Drill spacing <25m needed for skarn geometry (complex geometry requires dense drilling)",
            "Exclude porphyry analogs — skarn controls differ fundamentally",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_copper_sediment_hosted",
        "source_material": "copper",
        "deposit_type": "sediment-hosted",
        "grade_min": 0.5, "grade_max": 4.0, "grade_unit": "% Cu",
        "tonnage_min_mt": 10.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Sediment-Hosted Copper Analog Selection",
        "description": "Select analogs for stratiform sediment-hosted copper (Kupferschiefer, Central African Copperbelt).",
        "analog_criteria": [
            "Match sediment host (Kupferschiefer-style vs Central African red-bed style)",
            "Similar stratigraphy thickness, dip, and continuity",
            "Same basin type and redox boundary controls",
            "Match Co/Ni ratios (Central African Copperbelt analogs expected higher Co)",
            "Drill spacing 100m acceptable for stratiform — good continuity assumed",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },

    # ── URANIUM ───────────────────────────────────────────────────────────────
    {
        "rule_id": "analog_sel_uranium_rollfront",
        "source_material": "uranium",
        "deposit_type": "roll-front",
        "grade_min": 0.02, "grade_max": 0.15, "grade_unit": "% U3O8",
        "tonnage_min_mt": 5.0, "tonnage_max_mt": None,
        "drilling_stage": "dense",
        "title": "Roll-Front Uranium Analog Selection",
        "description": "Select analogs for roll-front uranium in permeable sandstones — ISR mining method.",
        "analog_criteria": [
            "In-situ recovery (ISR) mining method required",
            "Same basin type (Wyoming-type vs Kazakh-type redox boundary)",
            "Match U3O8 grade: 0.02-0.08% typical for ISR roll-fronts",
            "Similar host sandstone permeability and porosity",
            "Shallow depth: <300m for ISR operations",
            "Exclude underground or open-pit uranium analogs — ISR economics differ fundamentally",
            "Match groundwater chemistry (oxidising vs reducing boundary)",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_uranium_unconformity",
        "source_material": "uranium",
        "deposit_type": "unconformity",
        "grade_min": 0.5, "grade_max": 15.0, "grade_unit": "% U3O8",
        "tonnage_min_mt": 1.0, "tonnage_max_mt": None,
        "drilling_stage": "dense",
        "title": "Unconformity Uranium Analog Selection",
        "description": "Select analogs for unconformity-related uranium — Athabasca Basin, McArthur River, Cigar Lake style.",
        "analog_criteria": [
            "Proterozoic unconformity setting (Athabasca-type)",
            "High grade required: typically 1-15% U3O8",
            "Underground mining method (depth 300-700m typically)",
            "Graphitic pelite fault conductor association",
            "Match clay alteration halo (bleaching, illite/chlorite)",
            "EM/resistivity conductor required for analog validation",
            "McArthur River (4 Mt @ 5%) and Cigar Lake (5 Mt @ 10%) as benchmark analogs",
            "Exclude roll-front or intrusive analogs — grade/tonnage scales are incompatible",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_uranium_intrusive",
        "source_material": "uranium",
        "deposit_type": "intrusive",
        "grade_min": 0.02, "grade_max": 0.15, "grade_unit": "% U3O8",
        "tonnage_min_mt": 50.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Intrusive Uranium Analog Selection",
        "description": "Select analogs for intrusive-related uranium — Rössing, Husab style alaskite-hosted.",
        "analog_criteria": [
            "Alaskite or leucogranite hosted uranium",
            "Namibian Damaran Belt or similar Proterozoic mobile belt setting",
            "Open pit mining for bulk low-grade (0.02-0.05% U3O8, >100 Mt)",
            "Rössing (200 Mt @ 0.04%) and Husab as primary benchmark analogs",
            "Match leucogranite intrusion volume and U mineralogy (uraninite vs davidite)",
            "Exclude sandstone-hosted or unconformity analogs",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },

    # ── PGM ───────────────────────────────────────────────────────────────────
    {
        "rule_id": "analog_sel_pgm_merensky",
        "source_material": "pgm",
        "deposit_type": "Merensky",
        "grade_min": 2.0, "grade_max": 12.0, "grade_unit": "g/t PGE",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Merensky Reef PGM Analog Selection",
        "description": "Select analogs for Merensky Reef PGM — Bushveld Complex thin chromitite reef.",
        "analog_criteria": [
            "Bushveld Complex Merensky Reef or direct equivalent",
            "Match Pt:Pd:Rh:Au grade ratios (Merensky: Pt-dominant, ~60:30:5:5)",
            "Reef-specific thickness: 0.5-2.0m typical",
            "Pothole loss adjustments: 15-50% — match analog pothole frequency",
            "Underground trackless mining or conventional stoping",
            "Depletion caps for producing mines (15-25%)",
            "Compare Cu and Ni by-product credits (+8-18% Cu, +5-12% Ni value)",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_pgm_ug2",
        "source_material": "pgm",
        "deposit_type": "UG2",
        "grade_min": 3.0, "grade_max": 10.0, "grade_unit": "g/t PGE",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "UG2 Chromitite Reef PGM Analog Selection",
        "description": "Select analogs for UG2 chromitite reef — higher Pd:Pt ratio vs Merensky, chromite by-product.",
        "analog_criteria": [
            "UG2 chromitite or equivalent chromitite reef",
            "Match Pd:Pt ratio (UG2: Pd-dominant, ~55:35:7:3 Pd:Pt:Rh:Au)",
            "Chromite by-product credit relevant",
            "Thinner reef: 0.3-1.0m typical",
            "Match mining method (mechanised narrow reef vs conventional)",
            "Acid-insoluble residue (AIR) correction for chromite dilution",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_pgm_platreef",
        "source_material": "pgm",
        "deposit_type": "Platreef",
        "grade_min": 1.5, "grade_max": 6.0, "grade_unit": "g/t PGE",
        "tonnage_min_mt": 50.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Platreef PGM Analog Selection",
        "description": "Select analogs for Platreef — thick, lower-grade, bulk tonnage northern Bushveld.",
        "analog_criteria": [
            "Platreef or equivalent thick (5-30m) PGM reef",
            "Bulk tonnage potential — match footprint scale",
            "Base metal enrichment (Ni, Cu) higher than Merensky",
            "Match mining method: large-scale trackless, potential open pit for shallow sections",
            "Waterberg and Platreef analogous projects in Limpopo",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },

    # ── NICKEL ────────────────────────────────────────────────────────────────
    {
        "rule_id": "analog_sel_nickel_magmatic_sulphide",
        "source_material": "nickel",
        "deposit_type": "magmatic sulphide",
        "grade_min": 0.3, "grade_max": 5.0, "grade_unit": "% Ni",
        "tonnage_min_mt": None, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Magmatic Nickel Sulphide Analog Selection",
        "description": "Select analogs for magmatic Ni-Cu-PGE sulphide deposits — Norilsk, Voisey's Bay, Thompson Belt style.",
        "analog_criteria": [
            "Mafic/ultramafic intrusion hosting (komatiite, gabbro, peridotite)",
            "Match Ni/Cu ratio (>1.0 for Ni-dominant, <1.0 for Cu-dominant)",
            "Rift-conduit deposits: Ni/Cu >1.5, >10km strike — uplift potential",
            "Gabbro-hosted: +3-5% tonnage uplift for piercement zones (from lessons)",
            "Ultramafic-hosted: apply 50-75% of gabbro uplift",
            "PGE stability check: Pd+Pt >0.5 g/t validates ultramafic type",
            "EM/IP continuity >95% required for M&I classification",
            "Thompson Nickel Belt, Kamoa-Kakula, Kabanga as benchmark analogs",
            "Drill spacing 50-100m for M&I magmatic Ni",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_nickel_laterite",
        "source_material": "nickel",
        "deposit_type": "laterite",
        "grade_min": 0.8, "grade_max": 3.0, "grade_unit": "% Ni",
        "tonnage_min_mt": 10.0, "tonnage_max_mt": None,
        "drilling_stage": "dense",
        "title": "Nickel Laterite Analog Selection",
        "description": "Select analogs for saprolite/limonite nickel laterite deposits.",
        "analog_criteria": [
            "Tropical weathering profile (saprolite vs limonite vs bedrock zones)",
            "Match Ni/Co ratio (laterites: Co-rich limonite vs Ni-rich saprolite)",
            "Same processing route: HPAL (limonite) vs RKEF (saprolite) — fundamentally different economics",
            "Match depth to water table and saprolite thickness",
            "Similar parent rock (peridotite, dunite, serpentinite)",
            "Same country or tropical belt (Philippines, Indonesia, New Caledonia, Cuba)",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },

    # ── IRON ORE ──────────────────────────────────────────────────────────────
    {
        "rule_id": "analog_sel_iron_bif_hematite",
        "source_material": "iron",
        "deposit_type": "BIF hematite",
        "grade_min": 50.0, "grade_max": 68.0, "grade_unit": "% Fe",
        "tonnage_min_mt": 50.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "BIF Hematite Iron Ore Analog Selection",
        "description": "Select analogs for supergene BIF-hosted hematite iron ore — Pilbara, Carajás, Hamersley style.",
        "analog_criteria": [
            "Craton-specific matching: Pilbara, Carajás, Hamersley, Gawler, Transvaal, Yilgarn",
            "Match BIF host (Hamersley Group, Itabira Group, Transvaal Supergroup)",
            "Hematite vs magnetite mineralogy — critical for processing and grade",
            "Goethite content: >20% goethite causes M&I tonnage reduction (-12 to -1.5%)",
            "Match plateau vs ridge-hosted geometry",
            "Depletion caps for producing mines (15-25%)",
            "Match mine life and production rate for analog validity",
            "DSO (direct shipping ore) vs beneficiation — match processing requirement",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
    {
        "rule_id": "analog_sel_iron_magnetite",
        "source_material": "iron",
        "deposit_type": "magnetite",
        "grade_min": 20.0, "grade_max": 45.0, "grade_unit": "% Fe",
        "tonnage_min_mt": 100.0, "tonnage_max_mt": None,
        "drilling_stage": "moderate",
        "title": "Magnetite Iron Ore Analog Selection",
        "description": "Select analogs for magnetite iron ore requiring beneficiation to 65%+ Fe concentrate.",
        "analog_criteria": [
            "Magnetite BIF or skarn-hosted (requires crushing and magnetic separation)",
            "Match Davis Tube Recovery (DTR) and magnetic separation efficiency",
            "Target concentrate grade 65-70% Fe at 30-35% mass recovery",
            "Nugget effect 6-8% for magnetite — match with analogous variography",
            "Non-craton magnetite: +1-3% for >70-80% supergene DSO outside cratons",
            "Match strip ratio and beneficiation OPEX profile",
        ],
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0, "confidence_modifier": 0,
    },
]


# ── Confidence Adjustment Rules ────────────────────────────────────────────────

CONFIDENCE_RULES = [
    # These apply to ALL commodities — source_material = 'all'
    {
        "rule_id": "conf_adj_early_exploration",
        "source_material": "gold",  # duplicated per commodity in seed loop
        "deposit_type": None,
        "project_stage_filter": "early exploration",
        "title": "Early Exploration Confidence Reduction",
        "description": "Early exploration projects have sparse drilling, high geological uncertainty.",
        "analog_criteria": [],
        "confidence_modifier": -25.0,
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0,
    },
    {
        "rule_id": "conf_adj_advanced_exploration",
        "source_material": "gold",
        "deposit_type": None,
        "project_stage_filter": "advanced exploration",
        "title": "Advanced Exploration Confidence Reduction",
        "description": "Advanced exploration: more drilling but still pre-resource.",
        "analog_criteria": [],
        "confidence_modifier": -15.0,
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0,
    },
    {
        "rule_id": "conf_adj_pea",
        "source_material": "gold",
        "deposit_type": None,
        "project_stage_filter": "pea",
        "title": "PEA Stage Confidence",
        "description": "PEA stage: initial resource estimate completed, moderate confidence.",
        "analog_criteria": [],
        "confidence_modifier": -5.0,
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0,
    },
    {
        "rule_id": "conf_adj_prefeasibility",
        "source_material": "gold",
        "deposit_type": None,
        "project_stage_filter": "pre-feasibility",
        "title": "Pre-Feasibility Confidence",
        "description": "Pre-feasibility: M&I resource defined, drilling density adequate.",
        "analog_criteria": [],
        "confidence_modifier": 0.0,
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0,
    },
    {
        "rule_id": "conf_adj_feasibility",
        "source_material": "gold",
        "deposit_type": None,
        "project_stage_filter": "feasibility",
        "title": "Feasibility Stage Confidence Boost",
        "description": "Feasibility: high drilling density, measured resources, bankable study completed.",
        "analog_criteria": [],
        "confidence_modifier": 10.0,
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0,
    },
    {
        "rule_id": "conf_adj_production",
        "source_material": "gold",
        "deposit_type": None,
        "project_stage_filter": "production",
        "title": "Production Stage Confidence Boost",
        "description": "Producing mine: actual production reconciles MRE, high confidence.",
        "analog_criteria": [],
        "confidence_modifier": 15.0,
        "tonnage_multiplier": 1.0, "grade_multiplier": 1.0,
    },
]

# Commodities to replicate confidence rules for
ALL_COMMODITIES = ["gold", "silver", "gold_silver", "copper", "nickel", "uranium", "pgm", "iron"]


def build_rows(rules_list: list, rule_type: str, extra_commodities: list | None = None) -> list:
    """Convert rule dicts to Supabase rows."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    source_rules = rules_list

    # For confidence rules, replicate across all commodities
    if extra_commodities:
        expanded = []
        for r in rules_list:
            for commodity in extra_commodities:
                new_r = dict(r)
                new_r["rule_id"] = r["rule_id"].replace("gold", commodity) if "gold" in r["rule_id"] else f"{r['rule_id']}_{commodity}"
                new_r["source_material"] = commodity
                expanded.append(new_r)
        source_rules = expanded

    for r in source_rules:
        rows.append({
            "id": str(uuid4()),
            "rule_id": r["rule_id"],
            "source_material": r["source_material"],
            "source_lesson": f"lessons_learned_{r['source_material']}",
            "rule_type": rule_type,
            "deposit_type": r.get("deposit_type"),
            "analog_criteria": r.get("analog_criteria") or [],
            "grade_min": r.get("grade_min"),
            "grade_max": r.get("grade_max"),
            "grade_unit": r.get("grade_unit"),
            "tonnage_min_mt": r.get("tonnage_min_mt"),
            "tonnage_max_mt": r.get("tonnage_max_mt"),
            "project_stage_filter": r.get("project_stage_filter"),
            "drilling_stage": r.get("drilling_stage"),
            "title": r.get("title", ""),
            "description": r.get("description", ""),
            "weight": 0.8,
            "confidence_modifier": float(r.get("confidence_modifier", 0)),
            "model_effects_json": {
                "tonnage_multiplier": r.get("tonnage_multiplier", 1.0),
                "grade_multiplier": r.get("grade_multiplier", 1.0),
                "reasoning": r.get("description", ""),
            },
            "active": True,
            "compiled_at": now,
            "compiler_version": "v3",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Seed analog_selection rules from lessons learned")
    parser.add_argument("--commodity", default=None, help="Limit to one commodity")
    parser.add_argument("--dry-run", action="store_true", help="Print counts, don't write")
    args = parser.parse_args()

    if not settings.supabase_url:
        logger.error("SUPABASE_URL not set")
        sys.exit(1)

    analog_rows = build_rows(ANALOG_SELECTION_RULES, "analog_selection")
    conf_rows = build_rows(CONFIDENCE_RULES, "confidence_adjustment", extra_commodities=ALL_COMMODITIES)

    if args.commodity:
        analog_rows = [r for r in analog_rows if r["source_material"] == args.commodity]
        conf_rows = [r for r in conf_rows if r["source_material"] == args.commodity]

    all_rows = analog_rows + conf_rows
    logger.info(f"Prepared {len(analog_rows)} analog_selection + {len(conf_rows)} confidence_adjustment rules")

    if args.dry_run:
        for r in all_rows[:5]:
            logger.info(f"  Sample: {r['rule_id']} ({r['source_material']}, {r['rule_type']})")
        logger.info(f"  ... {len(all_rows)} total")
        return

    client = get_client()
    saved = 0
    for i in range(0, len(all_rows), 20):
        batch = all_rows[i:i+20]
        try:
            client.table("compiled_rules").upsert(batch, on_conflict="rule_id").execute()
            saved += len(batch)
            logger.info(f"  Upserted {saved}/{len(all_rows)} rules")
        except Exception as e:
            logger.error(f"  Error at batch {i}: {e}")

    logger.info(f"\n=== Done: {saved} rules seeded ===")


if __name__ == "__main__":
    main()
