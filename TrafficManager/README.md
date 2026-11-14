# VL-Adversary simulation component

The main adversarial entry point is `sim_manager_t_attack.py`; `attack_framework/` contains selection, planning, and injection modules.

Brake-check, cut-in, and opposite-direction intrusion all rerun both LLM
planning stages during eligible runtime updates, at least 0.5 s of simulation
time apart. Updates use the latest vehicle states and keep the selected
vehicle and attack type fixed. They stop when the braking vehicle has stopped,
the cut-in vehicle has joined the ego lane, or the oncoming vehicle has passed
the ego vehicle, respectively. If an LLM update fails, cut-in uses its
rule-based fallback and the other maneuvers keep their current segment.
Set `attack.llm_replanning: false` to use rule-based cut-in updates and the
initial LLM-generated segments for the other maneuvers.

Run commands from the project root. See the [English guide](../docs/QUICKSTART.md) or [中文指南](../docs/QUICKSTART_CN.md) for environment preparation, service startup, model configuration, and scene examples.

For custom road networks, see the inherited [map guide](docs/custommap.md) and [中文地图指南](docs/custommap_CN.md).
