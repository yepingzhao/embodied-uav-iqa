# Round 3 Re-Evaluation Bundle

I revised the proposal based on your Round 2 feedback. Three minor changes only:

1. **VLA accessibility pre-start gate**: Added Month-0 verification step. If UAV-Track VLA / CognitiveDrone-R1 / Qwen-VLA are unavailable, backup plan: OpenVLA fine-tuned on AirCopBench (sim-to-real transfer validated in AirCopBench paper). If all unavailable, flag as limitation with 2-VLA annotator fallback.

2. **Execution layer per-task scope**: Tracking + Inspection use CARLA-Air SITL (well-supported). Delivery + SAR use VLA-only annotation (no execution ground truth, higher reported uncertainty). Execution calibration applies to tracking+inspection only (~4,500 pairs — still sufficient for linear calibration).

3. **Sequential real-UAV validation**: Adaptive strategy — start with 2 distortion types, analyze power, expand. Total flights ~50 but allocation is adaptive with explicit SRCC thresholds for proceed/refine/flag decisions.

Revised proposal path (read this file yourself): /home/uesr/zhao/github/embodied-uav-iqa/refine-logs/round-2-refinement.md

(The full proposal is in round-1-refinement.md with these 3 localized changes — see round-2-refinement.md for the change descriptions.)

Please re-score the same 7 dimensions. These are minor implementation de-risking changes — the core method is unchanged from Round 2.
