# VL-Adversary simulation component

The main adversarial entry point is `sim_manager_t_attack.py`; `attack_framework/` contains selection, planning, and injection modules.

During an unfinished cut-in, eligible runtime updates (at least 0.5 s of
simulation time apart) rerun both LLM planning stages with the latest vehicle
states. The selected vehicle and attack type stay fixed, and the simulator
waits for the replanned segment before advancing. Failed LLM updates use the
rule-based replanner. Set `attack.llm_cut_in_replanning: false` to use the
rule-based replanner for all cut-in updates.

Run commands from the project root. See the [English guide](../docs/QUICKSTART.md) or [中文指南](../docs/QUICKSTART_CN.md) for environment preparation, service startup, model configuration, and scene examples.

For custom road networks, see the inherited [map guide](docs/custommap.md) and [中文地图指南](docs/custommap_CN.md).
