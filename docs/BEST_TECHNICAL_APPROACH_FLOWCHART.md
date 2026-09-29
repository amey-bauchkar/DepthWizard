# DepthWizard — Best Technical Approach Architecture Flowchart

This document contains the definitive, jury-grade architecture flowchart for DepthWizard's **Technical Approach Slide** (designed for ISRO SIH 26175).

---

## 1. Master Flowchart (Mermaid)

> **Copy-paste this directly** into any Mermaid renderer (Markdown, GitHub, Notion, Canva, Mermaid Live Editor, or PowerPoint Mermaid plugin).

```mermaid
flowchart LR
    %% Modern High-Contrast Style Definitions
    classDef input fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#ffffff;
    classDef ai fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ffffff;
    classDef geo fill:#1e293b,stroke:#60a5fa,stroke-width:2px,color:#ffffff;
    classDef fusion fill:#2a1b08,stroke:#f59e0b,stroke-width:2.5px,color:#ffffff;
    classDef twin fill:#2e1065,stroke:#c084fc,stroke-width:2px,color:#ffffff;
    classDef disaster fill:#3b0717,stroke:#fb7185,stroke-width:2px,color:#ffffff;

    subgraph S1 ["<b>STAGE 1: INPUTS</b>"]
        direction TB
        IMG["<b>📷 2D Optical Image</b><br/>Satellite / Drone RGB (0.5m GSD)<br/><i>Mode A: Relative · Mode B: Metric</i>"]:::input
        DEM["<b>🌍 Coarse 30m DEM</b><br/>Copernicus GLO-30 / CartoDEM<br/><i>Macro terrain baseline</i>"]:::input
    end

    subgraph S2 ["<b>STAGE 2: PARALLEL PROCESSING</b>"]
        direction TB
        AI["<b>🤖 AI Height Model</b><br/>Depth Anything V2 (ViT-S/14)<br/><i>Fine-tuned on LiDAR → nDSM (m)</i>"]:::ai
        GEO["<b>📐 C-1 Geodetic Datum Guard</b><br/>Ellipsoid to EGM2008 Geoid<br/><i>Offline PROJ · Zero datum error</i>"]:::geo
    end

    subgraph S3 ["<b>STAGE 3: CORE INNOVATION</b>"]
        direction TB
        FUSION["<b>⚡ TL-CSM Detail Fusion</b><br/><b>DSM = Base DEM + AI Heights</b><br/>• Low-Freq (≥30m): Stable DEM terrain<br/>• High-Freq (&lt;30m): AI building/tree detail<br/><i>No bias added at 30 m · Tiers R, H, T, A</i>"]:::fusion
    end

    subgraph S4 ["<b>STAGE 4: 3D DIGITAL TWIN</b>"]
        direction TB
        MESH["<b>🏙️ 3D Terrain &amp; LoD-1 City</b><br/>Adaptive RTIN mesh + extruded buildings<br/><i>Aerial texture drape + anti-smear facade</i>"]:::twin
        VIEW["<b>📦 Offline Standalone Package</b><br/>1.7m walk, drone orbit, measurement tools<br/><i>Single 8MB standalone HTML (Zero server)</i>"]:::twin
    end

    subgraph S5 ["<b>STAGE 5: DISASTER OPERATIONS</b>"]
        direction TB
        HLZ["<b>🚁 Helicopter Landing Zones (HLZ)</b><br/>FM 3-21.38 military standard<br/><i>16-bearing 10:1 obstacle clearance</i>"]:::disaster
        FLOOD["<b>🌊 Flood Inundation Model</b><br/>HAND valley drainage hydrology<br/><i>Per-building water depth &amp; exposure</i>"]:::disaster
        DAMAGE["<b>🏚️ Collapse Damage Screening</b><br/>Pre vs post disaster nDSM difference<br/><i>71–88% precision (24-flag audit, Islahiye)</i>"]:::disaster
    end

    %% Pipeline Flow
    IMG --> AI
    DEM --> GEO
    AI --> FUSION
    GEO --> FUSION
    FUSION --> MESH
    MESH --> VIEW
    MESH --> HLZ
    MESH --> FLOOD
    MESH --> DAMAGE
```

---

## 2. Text / ASCII Schematic (For Quick Notes & Slide Layout)

```
[ STAGE 1: INPUTS ]             [ STAGE 2: PROCESSING ]        [ STAGE 3: CORE FUSION ]        [ STAGE 4: 3D TWIN ]         [ STAGE 5: DISASTER OPS ]
┌───────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│ 📷 2D Optical Image       ├──►│ 🤖 AI Height Model        ├──►│ ⚡ TL-CSM Fusion          ├──►│ 🏙️ 3D Terrain + City   ├──►│ 🚁 Helicopter HLZ      │
│ (Satellite / Drone RGB)   │   │ (Depth Anything V2 LiDAR) │   │ DSM = Base DEM + AI Detail│   │ (LoD-1 Extruded Blocks)│   │ (FM 3-21.38 10:1 Clearance)│
└───────────────────────────┘   └───────────────────────────┘   │ • ≥30m: Base DEM terrain  │   └───────────┬────────────┘   ├────────────────────────┤
                                                                │ • <30m: AI object heights │               │                │ 🌊 HAND Flood Model    │
┌───────────────────────────┐   ┌───────────────────────────┐   │ No bias added at 30 m     │               ▼                │ (Per-building depth)   │
│ 🌍 Coarse 30m DEM         ├──►│ 📐 Geodetic Datum Guard   ├──►│ Tiers: R · H · T · A      │   ┌────────────────────────┐   ├────────────────────────┤
│ (Copernicus GLO-30)       │   │ (EGM2008 Geoid Alignment) │   └───────────────────────────┘   │ 📦 Offline 8MB HTML    │   │ 🏚️ Collapse Screening  │
└───────────────────────────┘   └───────────────────────────┘                                   │ (Walk/Fly, zero server)│   │ (Pre/post damage 71-88%)│
                                                                                                └────────────────────────┘   └────────────────────────┘
```

---

## 3. The 45-Second Judge Presentation Script

When this slide is on the screen, deliver these 4 points:

1. **The Problem & Inputs:**
   > *"A single 2D satellite photo gives sharp visual details but has zero elevation data. Coarse 30m DEMs give regional terrain but miss every building and tree. DepthWizard takes both as inputs."*

2. **Parallel Processing:**
   > *"In parallel, our fine-tuned Depth Anything V2 model predicts object heights in metres, while our Geodetic Datum Guard locks the base DEM to the global EGM2008 geoid datum to eliminate vertical shift errors."*

3. **The Core Innovation (TL-CSM Fusion):**
   > *"Instead of hallucinating heights, our TL-CSM multiscale fusion preserves the reliable base DEM for macro terrain (≥30m) and injects the AI model's sharp structural details (<30m). It adds no bias at the DEM's 30 m scale, and against LiDAR it cut the DSM error by 14–40 % on every Swiss test tile; on Sikkim it improved the DSM on 5 of 6 sites."*

4. **Deliverables & Tactical Disaster Operations:**
   > *"From this fused DSM, we extrude LoD-1 3D city blocks and package everything into an offline 8MB HTML file that responders can run on any laptop without internet. On top of this digital twin, we execute three critical disaster tools: automated **helicopter landing zone screening**, **HAND flood inundation modeling**, and **post-disaster building collapse detection**."*
