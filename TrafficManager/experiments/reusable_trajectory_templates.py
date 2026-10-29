"""Generate one pair of parameterized planner functions per attack, shared by all runs."""
import ast
import hashlib
import json
from pathlib import Path


def check_function(code, name):
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1 or len(functions[0].args.args) != 3:
        raise ValueError("Expected a three-argument function: " + name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(m.split(".")[0] not in ("math", "numpy") for m in modules):
                raise ValueError("Planner imports outside math/numpy are prohibited")
        if isinstance(node, ast.Name) and node.id in ("exec", "eval", "open", "compile", "__import__", "globals", "locals", "getattr", "setattr"):
            raise ValueError("Unsupported planner operation")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in ("load", "save", "loadtxt", "savetxt", "fromfile", "tofile", "ctypeslib")):
            raise ValueError("Unsupported planner attribute")


def prepare_templates(directory):
    # Lightweight imports keep preparation independent from the SUMO runtime.
    from TrafficManager import run_selector_only_comparison as selection
    import importlib
    prompts = importlib.import_module(selection.PACKAGE + ".prompts")
    models = importlib.import_module(selection.PACKAGE + ".models")
    base = selection.service_config()
    model = "qwen-coder-plus-1106"
    config = selection.config_module.LLMServiceConfig(True, base.api_key, base.api_base, model)
    client = selection.client_module.LLMClient(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generic = "\n这是跨场景复用的通用函数，不提供具体场景。所有车辆状态、车道编号、宽度及限速必须在函数执行时读取参数；禁止写死场景坐标、车辆ID或车道ID。"
    hashes = {}
    for attack in models.AttackType:
        for stage, name, prompt in (
            ("goal", "calculate_goal_position", prompts.build_goal_function_prompt(attack, {})),
            ("path", "plan_attack_trajectory", prompts.build_path_planning_prompt({}, {}, {}, attack)),
        ):
            path = directory / (attack.value + "_" + stage + ".py")
            if not path.exists():
                code = client.create_code(prompt + generic)
                check_function(code, name)
                path.write_text(code, encoding="utf-8")
            check_function(path.read_text(encoding="utf-8"), name)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(dict(model=model, api_base=base.api_base,
        code_sha256=hashes, policy="one goal/path pair per attack type; frozen across all scenes and arms"), indent=2), encoding="utf-8")


class TemplateClient:
    """TwoStageTrajectoryGenerator still applies its original goal/dynamics constraints."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.attack = None
        self.stage = 0

    def create_code(self, prompt, **kwargs):
        stage = ("goal", "path")[self.stage]
        self.stage += 1
        return (self.directory / (self.attack + "_" + stage + ".py")).read_text(encoding="utf-8")


def use_templates(generator, directory):
    client = TemplateClient(directory)
    generator.llm_client = client
    original = generator.generate

    def generate(attack_type, *args, **kwargs):
        client.attack = attack_type.value
        client.stage = 0
        return original(attack_type, *args, **kwargs)

    generator.generate = generate
