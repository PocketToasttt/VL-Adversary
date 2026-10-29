# VL-Adversary

**Vision-Language-Driven Safety-Critical Trajectory Generation for Autonomous Driving**

VL-Adversary generates safety-critical interactions for autonomous driving evaluation. A vision-language model reasons over bird's-eye-view (BEV) images, vehicle states, and road topology to select an adversarial vehicle and its attack intention. A language model then generates executable programs for hazardous endpoint selection and constrained trajectory planning. The selected vehicle's future motion is updated using the evolving traffic state and ego-vehicle response.

[中文指南](docs/QUICKSTART_CN.md) · [Setup and running](docs/QUICKSTART.md) · [Third-party acknowledgments](THIRD_PARTY_NOTICES.md)

## Method

![VL-Adversary method overview](docs/images/method-overview.png)

*Overview of VL-Adversary: multimodal attacker selection, two-stage executable planning, and closed-loop trajectory updates using ego feedback.*

1. **Observe:** LimSim provides a BEV image, vehicle states, and road topology.
2. **Select:** the VLM selects a surrounding vehicle and an attack intention.
3. **Plan:** the LLM generates an endpoint function followed by a trajectory function, with road geometry, kinematics, and motion continuity included in the prompts.
4. **Interact:** TrafficManager executes the generated trajectory. DreamForge renders six camera views for UniAD.
5. **Update:** ego feedback and the latest scene state guide subsequent endpoint and trajectory updates for the selected vehicle.

The three supported intentions are **brake-check**, **cut-in**, and **opposite-direction intrusion**. Prompted constraints and runtime checks support trajectory feasibility; they do not constitute a formal guarantee for every model output.

## Demonstrations

### Brake-check

![Brake-check demonstration](docs/videos/brake_check_007.gif)

### Cut-in

![Cut-in demonstration](docs/videos/cut_in_003.gif)

### Opposite-direction intrusion

![Opposite-direction intrusion demonstration](docs/videos/opposite_direction_intrusion_002.gif)

## Running

The main entry point is `TrafficManager/sim_manager_t_attack.py`. The default configuration uses `qwen3-vl-plus` for selection and `qwen-coder-plus-1106` for endpoint and trajectory generation. Model availability depends on the API provider and account.

Prepare the Linux/WSL simulation environment, SUMO, model weights, and DreamForge/UniAD services using the [setup guide](docs/QUICKSTART.md). From the project root, after activating the simulation environment and setting `DASHSCOPE_API_KEY`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -u TrafficManager/batch_test.py --mode attack --per-type-limit 1
```

This runs a small batch from the included Boston-Seaport scenario collection. It performs real model requests and simulation, and requires the external services to be running. It is not an installation check.

## Code layout

| Directory | Purpose |
| --- | --- |
| `TrafficManager/attack_framework/` | Scene analysis, target selection, two-stage planning, and trajectory injection |
| `TrafficManager/networkFiles/` | Road networks and initial traffic configurations |
| `WorldDreamer/DreamForge/` | Multi-view generative rendering service |
| `DrivingAgents/UniAD/` | Ego driving model and its service |
| `docs/` | English and Chinese setup guides |

The public snapshot contains source code and scenario inputs. Model checkpoints, datasets, API credentials, and experiment outputs are not included. This repository does not yet provide a verified reproduction of the paper's quantitative tables.

## License and credits

See [LICENSE.txt](LICENSE.txt) and [third-party notices](THIRD_PARTY_NOTICES.md). Bundled components retain their original licenses and notices. VL-Adversary describes the adversarial generation method; the simulation, rendering, and driving components are credited separately.
