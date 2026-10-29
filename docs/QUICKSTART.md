# VL-Adversary setup and running

## Prerequisites

Use Linux or WSL with an appropriate graphics environment for the simulator. The simulation and GPU services use separate Python environments; do not install all component requirements into one environment.

- Simulation: SUMO 1.19-compatible tools, Python 3.8, and the dependencies in `TrafficManager/requirements.txt`. The supplied legacy requirements include PyTorch 1.9.1 and MMCV 1.4.0; their CUDA wheels must match the environment.
- Rendering: follow the bundled [DreamForge installation guide](../WorldDreamer/DreamForge/doc/INSTALL.md) and [data/weight preparation guide](../WorldDreamer/DreamForge/doc/GETTING_STARTED.md). These describe a separate CUDA 11.8 / Python 3.9 environment.
- Ego driving: follow the bundled [UniAD installation and weight instructions](../DrivingAgents/UniAD/README.md). This service also needs CAMixerSR and its checkpoint.
- Language models: access to the configured OpenAI-compatible API and both model IDs.

The component documents are inherited dependency instructions. This snapshot has not been installed and tested from scratch on a clean GPU machine.

## Prepare the simulation environment

From the `VL-Adversary` root in Linux/WSL:

```bash
conda create -n vl_adversary python=3.8
conda activate vl_adversary
pip install -r TrafficManager/requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Install SUMO separately, make its executables available, and configure `SUMO_HOME` for your installation. The small runtime template is supplied at `TrafficManager/data/data_template.pth`; it is copied from the included UniAD template and is not a model checkpoint.

## Start the GPU services

In the rendering environment, from `WorldDreamer/DreamForge`, after preparing weights and data:

```bash
python tools/run_fastapi.py --model_single ./pretrained/dreamforge-s --model ./pretrained/dreamforge-t --host 127.0.0.1 --port 11000
```

The service also loads `data/canbus_infos_trainval.pkl`; supply this metadata according to the component data preparation instructions. Do not use `--debug` for closed-loop simulation: that mode reads dataset samples instead of the simulator inputs.

In the ego driving environment, from `DrivingAgents/UniAD`:

```bash
python demo/uniad_fast_api.py --checkpoint ckpts/uniad_base_e2e.pth --host 127.0.0.1 --port 11001
```

Prepare `ckpts/CAMixerSRx4_DF.pth`, the CAMixerSR code dependency, and any data/configuration assets required by UniAD before starting. For remote services, forward these ports to the simulation machine or change the service addresses in `TrafficManager/config-attack.yaml`.

## Configure the models

The default configuration is `TrafficManager/config-attack.yaml`:

| Setting | Default |
| --- | --- |
| `servers.diffusion` | `http://127.0.0.1:11000/` |
| `servers.driver` | `http://127.0.0.1:11001/` |
| `selector.mode` / `selector.model` | `vlm` / `qwen3-vl-plus` |
| `llm.model` | `qwen-coder-plus-1106` |
| `llm.api_key_env` | `DASHSCOPE_API_KEY` |
| `simulation.use_agent_path` | `true` for UniAD feedback |

Set `DASHSCOPE_API_KEY` in the simulation terminal. For an interactive Bash terminal, this avoids putting the key in command history:

```bash
read -rsp 'Model API key: ' DASHSCOPE_API_KEY
echo
export DASHSCOPE_API_KEY
```

The selector inherits the trajectory model's service address and credentials unless separately configured. When using a different selector provider, specify its own `api_base` and `api_key_env`. Do not commit keys or local credentials. Changing model IDs changes the experimental setup.

## Run a scene or a small batch

Run from the project root with the simulation environment active. This single-scene example leaves vehicle and intention selection to the VLM:

```bash
python -u TrafficManager/sim_manager_t_attack.py \
  --config TrafficManager/config-attack.yaml \
  --scene_name brake_check_001 \
  --sumo_cfg networkFiles/boston-seaport/scenarios_diverse_3types/brake_check_001/brake_check_001.sumocfg \
  --sumo_rou networkFiles/boston-seaport/scenarios_diverse_3types/brake_check_001/brake_check_001.rou.xml
```

Scenario labels describe initial configurations; they do not force the VLM's selected intention. For one initial scene per category:

```bash
python -u TrafficManager/batch_test.py --mode attack --per-type-limit 1
```

The batch runner defaults to `TrafficManager/networkFiles/boston-seaport/scenarios_diverse_3types/scene_index.json`. Batch outputs are written under `test_results/attack/`. Single-scene outputs are written under `results/`. Full closed-loop runs need both GPU services and real model requests; no full GPU/API run was performed when packaging this snapshot.
