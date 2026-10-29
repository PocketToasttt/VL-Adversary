# Third-party notices

VL-Adversary adds vision-language adversarial selection, two-stage executable trajectory generation, and closed-loop trajectory updating to an existing simulation stack.

The simulation integration is derived from [DriveArena](https://github.com/PJLab-ADG/DriveArena). The release snapshot was prepared from the local main-method development checkout at revision `92616ff`. The project is presented under the VL-Adversary research-method name and is not the upstream project's official repository.

The included components also build on LimSim, DreamForge, UniAD, VAD, BaseDreamer, and their dependencies. Their original copyright headers, component documentation, and license files are retained. Dependency documentation may therefore use original project names.

- [Top-level Apache License 2.0](LICENSE.txt)
- [LimSim source](TrafficManager/LimSim/)
- [DreamForge](WorldDreamer/DreamForge/README.md)
- [UniAD](DrivingAgents/UniAD/README.md)
- [VAD](DrivingAgents/VAD/README.md)

The top-level license does not replace licenses applicable to individual bundled components. Dataset and pretrained-model terms apply separately; model checkpoints and complete datasets are not redistributed in this snapshot.
