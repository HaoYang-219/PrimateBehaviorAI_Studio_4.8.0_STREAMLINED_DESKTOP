# PrimateBehaviorAI Studio 4.1 Architecture

```text
PySide6 / Qt UI
    ↓
Viewpoint-aware calibration
  Right profile / Left profile / Oblique / Frontal
  Visible / Partial / Invisible per region
    ↓
Original-FPS multi-region feature extraction
    ↓
Dynamic polygon tracking + visibility gating
  Whole body / Head / Torso / 4 limbs
  Left eye / Right eye follow Head
  Invisible regions => confidence = 0, never alarm
    ↓
Global-motion vs local-residual decomposition
    ↓
Multi-scale temporal representation
  ~0.4 s  focal transient
  ~2 s    rhythmic/repetitive motion
  ~10 s   behavioral state / arousal context
    ↓
Baseline behavior-state model
  KMeans state discovery
  state-conditioned kNN deviation percentile
    ↓
Specific event channels
  Twitch-like / Tremor-like / Convulsion-like / Sway-like / Eye change
  Hypoactivity + Unknown novelty
    ↓
Behavioral endpoint layer
  Arousal/drowsiness proxy
  activity/immobility
  posture stability
  bilateral asymmetry (only when both sides are visible)
  state occupancy
    ↓
Quality-control gate
    ↓
10 s / 30 s / minute / session statistics
    ↓
Before / After comparison
    ↓
Chinese HTML report + CSVs + review dashboard
    ↓
SQLite expert-review database
```

## Core design rules

1. `movement != abnormality`.
2. A hidden or poorly visible body region must return unavailable, not abnormal.
3. Side-view videos are analyzed using visibility priors; far-side eyes/limbs are not treated as symmetric observations.
4. Arousal/drowsiness is a multi-signal behavioral proxy, not a diagnosis.
5. Before/After differences are descriptive; causal claims require Sham/repeated sessions and experimental controls.

## Tracking backend

4.1 uses the verified dynamic polygon tracker with periodically refreshed templates and explicit viewpoint/visibility gating. SAM2 remains a future pluggable backend and is **not** reported as active unless it is actually integrated into the main analysis path.
